"""End-to-end tests for the paste-file SHA verify + auto-repair loop.

These don't need a live host: we mock the webcam (any synthetic frame
is fine — the test patches the OCR step) and the OCR pass with a
script of pre-canned responses. The HID side is hit via an in-memory
fake-Pi that records every keystroke / keystroke-combo / text request
so we can assert the right host commands were typed in the right
order.

What's being validated:

1. **Happy path** — first SHA OCR matches local → single round, no
   chunk reads, no repair typing.
2. **OCR retry** — first OCR pass returns garbage; second OCR of the
   same SHA print returns the right thing. Should still report match
   without escalating to chunk repair.
3. **Single-chunk repair** — first SHA mismatch, chunks OCR shows
   one bad index; we type ONE base64+dd command; second SHA matches.
4. **Multi-chunk repair in one round** — three bad indices, all
   rewritten in one repair round.
5. **Max rounds exceeded** — SHA never converges; endpoint returns
   match=False with the per-round audit trail.
6. **Total OCR collapse** — neither SHA nor chunks parse anywhere;
   endpoint aborts cleanly with abort_reason populated.

These are the failure modes that justify the auto-repair design over
the previous SequenceMatcher heuristic.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from terminaleyes.commandcenter import paste_protocol as pp
from terminaleyes.commandcenter.frame_store import FrameStore
from terminaleyes.commandcenter.log_bus import LogBus
from terminaleyes.commandcenter.server import create_app


# A "settings" stub that satisfies the endpoint's attribute lookups.
class _CaptureCfg:
    device_index = 0
    resolution_width = None
    resolution_height = None


class _CommanderCfg:
    pi_base_url = "http://127.0.0.1:9"  # unused, kb mocked
    transport = "usb"
    screen_width = 1920
    screen_height = 1080


class _Settings:
    capture = _CaptureCfg()
    commander = _CommanderCfg()


@pytest.fixture
def watch_dir(tmp_path):
    d = tmp_path / "watch"
    d.mkdir()
    return d


@pytest.fixture
def store(watch_dir):
    # max_frames=10 is plenty for these tests.
    return FrameStore(watch_dir=watch_dir, max_frames=10)


@pytest.fixture
def bus():
    return LogBus()


@pytest.fixture
def kb_log() -> list[tuple[str, dict]]:
    """List that the mocked keyboard appends every call to."""
    return []


@pytest.fixture
def mock_kb(kb_log):
    """A keyboard mock that records every call. Returned object is
    the class to patch HttpKeyboardOutput with — instantiating it
    yields the same recorder so the test can assert the call order.
    """
    class _RecordingKb:
        def __init__(self, *a, **kw):
            pass

        async def connect(self):
            kb_log.append(("connect", {}))

        async def disconnect(self):
            kb_log.append(("disconnect", {}))

        async def send_text(self, text, *, warmup=True):
            kb_log.append(("text", {"text": text, "warmup": warmup}))

        async def send_keystroke(self, key):
            kb_log.append(("key", {"key": key}))

        async def send_key_combo(self, modifiers, key):
            kb_log.append(("combo", {"modifiers": list(modifiers), "key": key}))

    return _RecordingKb


@pytest.fixture
def mock_capture():
    """Returns a class that mimics WebcamCapture with a constant frame."""
    class _ConstantCapture:
        def __init__(self, *a, **kw):
            pass

        async def open(self):
            return None

        async def close(self):
            return None

        async def capture_frame(self):
            # Smallest plausible BGR frame — image content is irrelevant
            # since OCR is patched.
            from terminaleyes.domain.models import CapturedFrame
            return CapturedFrame(
                image=np.zeros((16, 16, 3), dtype=np.uint8),
                frame_number=1,
            )

    return _ConstantCapture


@contextmanager
def patched_runtime(mock_kb_cls, mock_capture_cls, ocr_responses: list[str]):
    """Patch the runtime imports used inside the paste-file endpoint.

    ``ocr_responses`` is consumed in FIFO order — each ``_ocr_now``
    invocation gets the next one. If the test under-supplies, the
    final response is repeated to avoid IndexError obscuring the real
    assertion.
    """
    ocr_iter = iter(ocr_responses)
    last = [""]

    def _ocr_next(*_args, **_kw):
        try:
            last[0] = next(ocr_iter)
        except StopIteration:
            pass
        return last[0]

    # The endpoint scatters asyncio.sleep() calls so the host's
    # terminal has time to render between commands. They add ~3-5s
    # per round in real time — irrelevant to the logic under test.
    async def _instant_sleep(_seconds, *a, **kw):
        return None

    with patch(
        "terminaleyes.keyboard.http_backend.HttpKeyboardOutput",
        mock_kb_cls,
    ), patch(
        "terminaleyes.capture.webcam.WebcamCapture",
        mock_capture_cls,
    ), patch(
        # The endpoint imports pytesseract lazily inside _ocr_now.
        "pytesseract.image_to_string",
        side_effect=_ocr_next,
    ), patch(
        "terminaleyes.commandcenter.server.asyncio.sleep",
        side_effect=_instant_sleep,
    ):
        yield


def _build_client(store, bus):
    async def _factory():
        # paste-file doesn't run the controller path, so the factory
        # only matters if someone hits /api/run during the test. Stub
        # it with no-op AsyncMocks.
        return AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()

    app = create_app(
        _factory, frame_store=store, bus=bus, settings=_Settings(),
    )
    return TestClient(app)


def _sha_block(content: bytes) -> str:
    """Synthesize what the host's framed SHA print would render as."""
    return (
        f"$ shasum -a 256 ...\n"
        f"{pp.SHA_OPEN}\n{pp.file_sha256(content)}\n{pp.SHA_CLOSE}\n$ "
    )


def _wrong_sha_block() -> str:
    return f"{pp.SHA_OPEN}\n{'b' * 64}\n{pp.SHA_CLOSE}\n"


def _chunks_block(content: bytes, *, bad_indices: list[int] = ()) -> str:
    """Synthesize chunks-print output where ``bad_indices`` carry a
    different hash, simulating those chunks having a dropped char."""
    good = pp.chunk_hashes(content)
    lines = []
    for i, h in enumerate(good):
        if i in bad_indices:
            # Flip a hex digit to differ from the local hash.
            h = "0" * 32 if h[0] != "0" else "f" * 32
        lines.append(f"{i} {h}")
    body = "\n".join(lines)
    return f"{pp.CHUNKS_OPEN}\n{body}\n{pp.CHUNKS_CLOSE}\n"


# ─────────────────────────── tests ────────────────────────────

def test_happy_path_first_sha_matches(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """First SHA OCR matches → exactly one verify round, no repair."""
    content = "hello\nworld\n"
    cb = content.encode()
    responses = [_sha_block(cb)]
    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content,
            "path": "/tmp/x.txt",
            "platform": "macos",
            "maximize": False,
            "verify": True,
        })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    v = data["verify"]
    assert v["match"] is True
    assert v["local_sha"] == pp.file_sha256(cb)
    assert len(v["rounds"]) == 1
    assert v["rounds"][0]["match"] is True

    # Behaviour check on HID side: no chunks-print, no overwrite-dd.
    kb_texts = [c[1]["text"] for c in kb_log if c[0] == "text"]
    assert any(c.startswith("cat > /tmp/x.txt") for c in kb_texts)
    assert any(pp.SHA_OPEN in c for c in kb_texts)  # SHA print
    assert not any(pp.CHUNKS_OPEN in c for c in kb_texts)
    assert not any("seek=" in c for c in kb_texts)


def test_ocr_retry_recovers_without_escalating_to_chunks(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """First SHA OCR returns garbage; the inner OCR-retry should
    re-print and parse on the second attempt. No chunk repair."""
    content = "the quick brown fox\n"
    cb = content.encode()
    responses = [
        "GARBAGE-no-framing",       # OCR retry 1
        _sha_block(cb),             # OCR retry 2
    ]
    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content, "maximize": False, "verify": True,
        })
    assert r.status_code == 200
    v = r.json()["verify"]
    assert v["match"] is True
    assert len(v["rounds"]) == 1
    # No CHUNKS print should have been sent.
    kb_texts = [c[1]["text"] for c in kb_log if c[0] == "text"]
    assert not any(pp.CHUNKS_OPEN in c for c in kb_texts)


def test_single_chunk_repair_then_match(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """SHA mismatch → chunks show one bad index → one base64+dd
    overwrite typed → second SHA matches."""
    content = "A" * (pp.CHUNK_SIZE * 2 + 50)  # 3 chunks
    cb = content.encode()
    responses = [
        _wrong_sha_block(),                                # round 0
        _chunks_block(cb, bad_indices=[1]),                # diff
        _sha_block(cb),                                    # round 1
    ]
    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content, "maximize": False, "verify": True,
        })
    assert r.status_code == 200
    v = r.json()["verify"]
    assert v["match"] is True
    assert len(v["rounds"]) == 2
    assert v["rounds"][0]["match"] is False
    assert v["rounds"][0]["bad_indices"] == [1]
    assert v["rounds"][1]["match"] is True

    # Exactly one overwrite-dd typed at seek=1.
    overwrites = [
        c[1]["text"] for c in kb_log
        if c[0] == "text" and "seek=" in c[1]["text"]
    ]
    assert len(overwrites) == 1
    assert "seek=1" in overwrites[0]
    assert f"bs={pp.CHUNK_SIZE}" in overwrites[0]
    assert "conv=notrunc" in overwrites[0]


def test_multi_chunk_repair_one_round(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """Three bad indices, all rewritten in a single round."""
    content = "X" * (pp.CHUNK_SIZE * 4)  # 4 chunks
    cb = content.encode()
    responses = [
        _wrong_sha_block(),
        _chunks_block(cb, bad_indices=[0, 2, 3]),
        _sha_block(cb),
    ]
    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content, "maximize": False, "verify": True,
        })
    assert r.status_code == 200
    v = r.json()["verify"]
    assert v["match"] is True
    assert sorted(v["rounds"][0]["bad_indices"]) == [0, 2, 3]

    overwrites = [
        c[1]["text"] for c in kb_log
        if c[0] == "text" and "seek=" in c[1]["text"]
    ]
    # One per bad index — order preserved (sorted).
    seeks = [int(o.split("seek=")[1].split()[0]) for o in overwrites]
    assert seeks == [0, 2, 3]


def test_max_rounds_exceeded_reports_clean_failure(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """SHA never converges across the configured 3 repair rounds →
    match=False, audit trail populated, endpoint returns 200 so the
    operator can inspect the rounds payload."""
    content = "Z" * pp.CHUNK_SIZE
    cb = content.encode()
    # Pattern: SHA mismatch + chunks(bad) repeated.
    responses = [
        _wrong_sha_block(),     # round 0 SHA
        _chunks_block(cb, bad_indices=[0]),
        _wrong_sha_block(),     # round 1 SHA
        _chunks_block(cb, bad_indices=[0]),
        _wrong_sha_block(),     # round 2 SHA
        _chunks_block(cb, bad_indices=[0]),
        _wrong_sha_block(),     # round 3 SHA — final attempt
    ]
    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content, "maximize": False, "verify": True,
        })
    assert r.status_code == 200
    v = r.json()["verify"]
    assert v["match"] is False
    # 4 SHA reads = rounds 0..3 (initial + 3 repair attempts).
    assert len(v["rounds"]) == 4
    assert all(r["match"] is False for r in v["rounds"])


def test_sha_disagrees_but_chunks_all_clean_aborts_with_reason(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """A specific paradox the loop has to guard against: OCR
    misreads the SHA line as a mismatch, but the per-chunk OCR
    cleanly shows every chunk matching local. There's nothing the
    repair loop can action — retransmitting "nothing" wouldn't
    change anything. Code MUST abort with a reason rather than
    spin to max rounds."""
    content = "Y" * (pp.CHUNK_SIZE * 2)
    cb = content.encode()
    responses = [
        _wrong_sha_block(),                # round 0 SHA: mismatch
        _chunks_block(cb, bad_indices=[]),  # chunks all match local
    ]
    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content, "maximize": False, "verify": True,
        })
    assert r.status_code == 200
    v = r.json()["verify"]
    assert v["match"] is False
    last = v["rounds"][-1]
    assert "abort_reason" in last
    assert "OCR" in last["abort_reason"]
    # No overwrites typed — there were no actionable bad chunks.
    assert not any(
        c[0] == "text" and "seek=" in c[1]["text"]
        for c in kb_log
    )


def test_total_ocr_collapse_repairs_defensively_until_max_rounds(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """When the chunks OCR can't be parsed at all, every chunk is
    treated as "unknown" and overwritten defensively. If SHA still
    won't converge after that, we exhaust MAX_REPAIR_ROUNDS and
    report failure with all rounds=match=False."""
    content = "Z" * (pp.CHUNK_SIZE * 2)
    responses = [
        _wrong_sha_block(),                 # round 0 SHA
        "noise no framing",                  # chunks fail 3x
        "noise",
        "noise",
        _wrong_sha_block(),                 # round 1 SHA
        "noise", "noise", "noise",
        _wrong_sha_block(),                 # round 2 SHA
        "noise", "noise", "noise",
        _wrong_sha_block(),                 # round 3 SHA (final)
    ]
    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content, "maximize": False, "verify": True,
        })
    assert r.status_code == 200
    v = r.json()["verify"]
    assert v["match"] is False
    # 4 SHA rounds (0..3) — initial + 3 repair attempts.
    assert len(v["rounds"]) == 4
    assert all(r["match"] is False for r in v["rounds"])
    # All chunks unknown → all repaired each round.
    nchunks = v["n_chunks"]
    for round_info in v["rounds"][:-1]:
        # Each repair round (rounds 0..2) records unknown_indices.
        if "unknown_indices" in round_info:
            assert sorted(round_info["unknown_indices"]) == list(range(nchunks))


def test_unknown_chunks_are_repaired_defensively(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """When OCR returns hashes for only SOME chunks, the unread ones
    are treated as bad and overwritten — they may or may not actually
    be wrong, but we'd rather pay the retransmit than declare a false
    success based on partial information."""
    content = "Y" * (pp.CHUNK_SIZE * 3)
    cb = content.encode()
    local = pp.chunk_hashes(cb)
    # OCR returns only chunk 0 (correct). 1 and 2 are "unknown".
    partial_block = f"{pp.CHUNKS_OPEN}\n0 {local[0]}\n{pp.CHUNKS_CLOSE}\n"
    responses = [
        _wrong_sha_block(),
        partial_block,
        _sha_block(cb),
    ]
    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content, "maximize": False, "verify": True,
        })
    assert r.status_code == 200
    v = r.json()["verify"]
    assert v["match"] is True
    bad = v["rounds"][0]["bad_indices"]
    # unknowns 1 and 2 should be retransmitted.
    assert sorted(bad) == [1, 2]

    seeks = sorted(
        int(c[1]["text"].split("seek=")[1].split()[0])
        for c in kb_log
        if c[0] == "text" and "seek=" in c[1]["text"]
    )
    assert seeks == [1, 2]


def test_409_when_runner_busy(store, bus, mock_kb, mock_capture):
    """A controller run holding the device must lock out the manual
    paste path — exactly the same gate as /api/mouse/* etc."""
    with patched_runtime(mock_kb, mock_capture, [_sha_block(b"hi\n")]):
        client = _build_client(store, bus)
        # Synthesize a busy state on the runner.
        client.app.state.runner._active = MagicMock(run_id="fake")
        try:
            r = client.post("/api/paste-file", json={
                "content": "hi\n", "maximize": False, "verify": False,
            })
            assert r.status_code == 409
        finally:
            client.app.state.runner._active = None


def test_body_readback_more_pagination_matches_sent(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """body_readback=True drives `more PATH`, captures each page via
    OCR, and reports a similarity score against the sent content.
    With perfect OCR (we feed back the original content split into
    pages), similarity should be ~1.0."""
    # Multi-page synthetic content with deterministic line shape.
    lines = [f"line {i:03d} hello world" for i in range(60)]
    content = "\n".join(lines) + "\n"
    cb = content.encode()

    # OCR responses: first the SHA happy-path frame, then one OCR
    # response per `more` page. Split content into 30-line pages.
    page_size = 30
    pages = [
        "\n".join(lines[i : i + page_size])
        for i in range(0, len(lines), page_size)
    ]
    # pages_budget for 60 lines = (60//30)+2 = 4. We supply 2 real
    # pages then 2 empty pages (post-EOF the screen is just the
    # shell prompt — OCR returns near-nothing).
    responses = [_sha_block(cb)] + pages + ["$ ", "$ "]

    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content,
            "maximize": False,
            "verify": True,
            "body_readback": True,
        })
    assert r.status_code == 200
    data = r.json()
    assert data["verify"]["match"] is True
    rb = data["body_readback"]
    # 4 pages budgeted for 60 lines (60//30 + 2).
    assert rb["pages"] == 4
    # OCR was perfect for the first 2 pages → similarity should be high.
    assert rb["similarity"] >= 0.95, rb

    # HID side: the readback section MUST type `clear`, `more PATH`,
    # then exactly `pages_budget` Spaces, then a defensive `q`.
    kb_texts = [c[1]["text"] for c in kb_log if c[0] == "text"]
    assert "clear" in kb_texts
    assert any(c.startswith("more /tmp/cc_paste.txt") for c in kb_texts)
    spaces = sum(1 for t in kb_texts if t == " ")
    assert spaces == 4
    assert "q" in kb_texts


def test_body_readback_off_skips_more(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """body_readback=False → no `more`, no Space, no `q`."""
    content = "hello\n"
    cb = content.encode()
    with patched_runtime(mock_kb, mock_capture, [_sha_block(cb)]):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content,
            "maximize": False,
            "verify": True,
            "body_readback": False,
        })
    assert r.status_code == 200
    assert "body_readback" not in r.json()
    kb_texts = [c[1]["text"] for c in kb_log if c[0] == "text"]
    assert not any(t.startswith("more ") for t in kb_texts)
    assert " " not in kb_texts


def test_body_readback_lossy_ocr_reports_lower_similarity(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """When OCR mangles the page content (substitutions, drops),
    similarity should still be reported — just lower. This is the
    user-visible signal that the visual readback isn't perfect even
    when SHA reports a clean match."""
    lines = [f"line {i:02d} foo bar baz" for i in range(30)]
    content = "\n".join(lines) + "\n"
    cb = content.encode()

    # Page budget for 30 lines = (30//30) + 2 = 3.
    # Provide one badly OCR'd page (random text), then empty pages.
    mangled = "compI3tely diff3r3nt text the OCR" * 5
    responses = [_sha_block(cb), mangled, "", ""]
    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content,
            "maximize": False,
            "verify": True,
            "body_readback": True,
        })
    assert r.status_code == 200
    data = r.json()
    # SHA verdict is independent and still True.
    assert data["verify"]["match"] is True
    rb = data["body_readback"]
    assert rb["similarity"] < 0.5, rb  # OCR completely mangled


def test_body_readback_page_budget_caps_at_60(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """Very large content shouldn't burn unbounded HID time on
    pagination. The page budget caps at 60 pages regardless."""
    # 2000 lines of 24 chars (48 KB, under the 50 KB request cap).
    # Naive budget = (2000//30)+2 = 68 → capped at 60.
    content = ("x" * 23 + "\n") * 2000
    cb = content.encode()
    # Feed one valid SHA response and many empty OCR pages.
    responses = [_sha_block(cb)] + [""] * 120
    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content,
            "maximize": False,
            "verify": True,
            "body_readback": True,
        })
    assert r.status_code == 200
    rb = r.json()["body_readback"]
    assert rb["pages"] == 60   # hard cap
    spaces = sum(
        1 for c in kb_log if c[0] == "text" and c[1]["text"] == " "
    )
    assert spaces == 60


def test_architecture_md_end_to_end_with_perfect_ocr(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """Drive the full pipeline with the real ARCHITECTURE.md from the
    repo: paste, SHA verify, `more` body readback paginated across
    multiple pages. With perfectly faithful OCR (we replay the actual
    file content split into pages) similarity should be very high.

    Establishes a baseline for the body-readback similarity score that
    a real run on hardware should approach (subject to webcam OCR
    noise)."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[3]
    md = repo_root / "ARCHITECTURE.md"
    if not md.exists():
        pytest.skip("ARCHITECTURE.md not in repo root — test skipped")
    content = md.read_text()
    cb = content.encode()

    # Split into 30-line pages (matches the endpoint's page-size
    # heuristic) so the "OCR" sees what a real `more` would render.
    all_lines = content.split("\n")
    page_size = 30
    pages = [
        "\n".join(all_lines[i : i + page_size])
        for i in range(0, len(all_lines), page_size)
    ]
    pages_budget = max(2, (content.count("\n") + 1) // 30 + 2)
    pages_budget = min(pages_budget, 60)
    # Pad with empty OCR responses for any extra pages the endpoint
    # captures past EOF.
    responses = [_sha_block(cb)] + pages + [""] * pages_budget

    with patched_runtime(mock_kb, mock_capture, responses):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content,
            "maximize": False,
            "verify": True,
            "body_readback": True,
        })

    assert r.status_code == 200, r.text
    data = r.json()
    # SHA path: deterministic.
    assert data["verify"]["match"] is True
    # Body readback: should converge to near-1 with perfect OCR.
    rb = data["body_readback"]
    assert rb["pages"] == pages_budget
    assert rb["similarity"] >= 0.95, rb
    # Command sequence sanity: cat redirect → SHA print → clear →
    # more → spaces → q.
    kb_texts = [c[1]["text"] for c in kb_log if c[0] == "text"]
    assert any(t.startswith("cat > /tmp/cc_paste.txt") for t in kb_texts)
    assert any(pp.SHA_OPEN in t for t in kb_texts)
    assert "clear" in kb_texts
    assert any(t.startswith("more /tmp/cc_paste.txt") for t in kb_texts)
    spaces = sum(1 for t in kb_texts if t == " ")
    assert spaces == pages_budget


def test_disabled_verify_skips_all_readback(
    store, bus, mock_kb, mock_capture, kb_log,
):
    """verify=False → no SHA / chunks commands typed at all."""
    content = "hello\n"
    with patched_runtime(mock_kb, mock_capture, []):
        client = _build_client(store, bus)
        r = client.post("/api/paste-file", json={
            "content": content, "maximize": False, "verify": False,
        })
    assert r.status_code == 200
    data = r.json()
    assert "verify" not in data
    kb_texts = [c[1]["text"] for c in kb_log if c[0] == "text"]
    assert not any(pp.SHA_OPEN in c for c in kb_texts)
    assert not any(pp.CHUNKS_OPEN in c for c in kb_texts)
