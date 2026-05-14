"""Tests for the paste-file integrity / repair protocol.

No live host: these only exercise the pure-logic side — hash math,
host-command-string construction, and OCR-text parsing. The actual
HID typing + webcam OCR is covered (by hand) when the target Mac is
connected.
"""
from __future__ import annotations

import hashlib

import pytest

from terminaleyes.commandcenter.paste_protocol import (
    CHUNK_SIZE,
    CHUNKS_CLOSE,
    CHUNKS_OPEN,
    SHA_CLOSE,
    SHA_OPEN,
    ChunkDiff,
    chunk_hashes,
    cmd_chunks_print,
    cmd_overwrite_chunk,
    cmd_sha_print,
    diff_chunks,
    file_sha256,
    n_chunks,
    parse_chunks_from_ocr,
    parse_sha_from_ocr,
)


# ───────────────────────── digests ─────────────────────────

def test_file_sha256_is_lowercase_hex_64():
    h = file_sha256(b"hello\n")
    assert h == hashlib.sha256(b"hello\n").hexdigest()
    assert len(h) == 64 and h == h.lower()


def test_chunk_hashes_split_at_chunk_size():
    data = b"A" * (CHUNK_SIZE * 2 + 5)  # 2 full chunks + 5-byte tail
    hashes = chunk_hashes(data)
    assert len(hashes) == 3
    assert hashes[0] == hashlib.md5(b"A" * CHUNK_SIZE).hexdigest()
    assert hashes[2] == hashlib.md5(b"A" * 5).hexdigest()


def test_chunk_hashes_empty_input_is_empty_list():
    assert chunk_hashes(b"") == []


def test_n_chunks_rounds_up():
    assert n_chunks(0) == 0
    assert n_chunks(1) == 1
    assert n_chunks(CHUNK_SIZE) == 1
    assert n_chunks(CHUNK_SIZE + 1) == 2


# ─────────────────── host-side command shape ──────────────

def test_cmd_sha_print_has_framing_and_path():
    c = cmd_sha_print("/tmp/foo.txt")
    assert SHA_OPEN in c and SHA_CLOSE in c
    assert "shasum -a 256 /tmp/foo.txt" in c


def test_cmd_chunks_print_has_framing_and_loop():
    c = cmd_chunks_print("/tmp/foo.txt", n=4)
    assert CHUNKS_OPEN in c and CHUNKS_CLOSE in c
    assert "n=4" in c and f"N={CHUNK_SIZE}" in c
    assert "openssl md5" in c
    assert "/tmp/foo.txt" in c


def test_cmd_overwrite_chunk_is_base64_dd_seek():
    payload = b"helloworld"
    c = cmd_overwrite_chunk("/tmp/foo.txt", idx=3, payload=payload)
    # Base64 carrier — no shell escaping needed for any payload.
    import base64
    assert base64.b64encode(payload).decode() in c
    assert "base64 -d" in c
    assert f"of=/tmp/foo.txt bs={CHUNK_SIZE} seek=3" in c
    assert "conv=notrunc" in c    # MUST avoid truncation


# ─────────────────────── OCR parsing ──────────────────────

def test_parse_sha_extracts_64_hex_from_framed_output():
    ocr = (
        "some terminal noise here\n"
        f"{SHA_OPEN}\n"
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08\n"
        f"{SHA_CLOSE}\n"
        "more noise\n"
    )
    assert parse_sha_from_ocr(ocr) == (
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    )


def test_parse_sha_returns_none_on_missing_framing():
    assert parse_sha_from_ocr("no framing here") is None
    # Wrong length — must reject (not a real SHA-256).
    assert parse_sha_from_ocr(f"{SHA_OPEN} abc {SHA_CLOSE}") is None


def test_parse_sha_lowercases_uppercase_hex():
    # Tesseract sometimes upcases — we should normalise.
    h = "A" * 64
    ocr = f"{SHA_OPEN} {h} {SHA_CLOSE}"
    out = parse_sha_from_ocr(ocr)
    assert out == "a" * 64


def test_parse_chunks_extracts_index_hash_pairs():
    ocr = (
        "noise above\n"
        f"{CHUNKS_OPEN}\n"
        "0 5d41402abc4b2a76b9719d911017c592\n"
        "1 7d793037a0760186574b0282f2f435e7\n"
        "2 6e809cbda0732ac4845916a59016f954\n"
        f"{CHUNKS_CLOSE}\n"
        "noise below\n"
    )
    out = parse_chunks_from_ocr(ocr)
    assert out == {
        0: "5d41402abc4b2a76b9719d911017c592",
        1: "7d793037a0760186574b0282f2f435e7",
        2: "6e809cbda0732ac4845916a59016f954",
    }


def test_parse_chunks_skips_malformed_lines_silently():
    ocr = (
        f"{CHUNKS_OPEN}\n"
        "0 5d41402abc4b2a76b9719d911017c592\n"
        "broken-line-no-hash\n"
        "2 6e809cbda0732ac4845916a59016f954\n"
        f"{CHUNKS_CLOSE}\n"
    )
    out = parse_chunks_from_ocr(ocr)
    assert set(out.keys()) == {0, 2}


def test_parse_chunks_returns_empty_when_block_missing():
    assert parse_chunks_from_ocr("just noise") == {}


# ─────────────────────── diff logic ───────────────────────

def test_diff_chunks_identifies_bad_and_unknown():
    local = ["aaa", "bbb", "ccc", "ddd"]
    host = {0: "aaa", 1: "BAD", 2: "ccc"}  # 1 bad, 3 missing
    d = diff_chunks(local, host)
    assert d.bad_indices == [1]
    assert d.unknown_indices == [3]


def test_diff_chunks_clean_match():
    local = ["aaa", "bbb"]
    host = {0: "aaa", 1: "bbb"}
    d = diff_chunks(local, host)
    assert d == ChunkDiff(bad_indices=[], unknown_indices=[])


# ─────────────── end-to-end: parse what we'd emit ─────────

def test_parse_sha_ignores_command_echo_with_no_inner_hex():
    """The typed command's own echo contains the framing tokens but
    no 64-char hex between them. Parser must skip past it."""
    real_hash = "a" * 64
    ocr = (
        f"$ {cmd_sha_print('/x')}\n"
        f"{SHA_OPEN}\n{real_hash}\n{SHA_CLOSE}\n$ "
    )
    assert parse_sha_from_ocr(ocr) == real_hash


def test_parse_chunks_ignores_command_echo_block():
    """The shell echoes the chunks loop on one line, which has both
    framing tokens but no real chunk lines between them. Parser must
    aggregate framed blocks so the real one wins."""
    ocr = (
        f"$ {cmd_chunks_print('/x', 2)}\n"   # echo: framing, no body
        f"{CHUNKS_OPEN}\n"
        "0 5d41402abc4b2a76b9719d911017c592\n"
        "1 7d793037a0760186574b0282f2f435e7\n"
        f"{CHUNKS_CLOSE}\n$ "
    )
    out = parse_chunks_from_ocr(ocr)
    assert out == {
        0: "5d41402abc4b2a76b9719d911017c592",
        1: "7d793037a0760186574b0282f2f435e7",
    }


def test_chunks_command_output_round_trips_through_parser():
    """Sanity: if we mimic what the host loop would emit for known
    payloads, our parser recovers the same dict the producer wrote."""
    payloads = [b"first chunk", b"second chunk", b"third"]
    expected = {
        i: hashlib.md5(p).hexdigest() for i, p in enumerate(payloads)
    }
    # Build a fake OCR transcript of the framed block.
    body = "\n".join(f"{i} {h}" for i, h in expected.items())
    ocr = f"prompt$ {cmd_chunks_print('/x', 3)}\n{CHUNKS_OPEN}\n{body}\n{CHUNKS_CLOSE}\n$ "
    assert parse_chunks_from_ocr(ocr) == expected
