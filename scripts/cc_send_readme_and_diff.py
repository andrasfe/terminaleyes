#!/usr/bin/env python3
"""Operator-facing validation: send README.md to host, read it back
via `more`, run a real diff against the original, print the verdict.

Drives the running Command Center on ``--cc-url`` (default
http://127.0.0.1:8765). The cc must be configured to talk to the
real Pi/host — this script doesn't mock anything.

Two independent identity proofs are reported:

1. **SHA-256 match** (cryptographic) — proves the file on the host
   is byte-identical to what we sent.
2. **`more` round-trip diff** — reconstructs the file from the
   webcam OCR of `more PATH` and runs ``diff -u`` against the
   original.

OCR is lossy on body text, so (2) will likely show *some*
differences on real hardware. The authoritative answer is (1).

Usage:
  python scripts/cc_send_readme_and_diff.py
  python scripts/cc_send_readme_and_diff.py --path ~/Downloads/README.md
  python scripts/cc_send_readme_and_diff.py --cc-url http://10.0.0.5:8765
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import requests


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    default_src = here / "README.md"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", type=Path, default=default_src,
        help="local file to send (default: repo README.md)",
    )
    ap.add_argument(
        "--path", default="~/Downloads/README.md",
        help="host path to write to (default: ~/Downloads/README.md)",
    )
    ap.add_argument(
        "--cc-url", default="http://127.0.0.1:8765",
        help="base URL of the running Command Center",
    )
    ap.add_argument(
        "--no-maximize", action="store_true",
        help="skip the Cmd+Ctrl+F terminal maximise step",
    )
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    if not src.exists():
        print(f"!! source file not found: {src}", file=sys.stderr)
        return 2
    content = src.read_text()

    body = {
        "content": content,
        "path": args.path,
        "platform": "macos",
        "maximize": not args.no_maximize,
        "verify": True,
        "body_readback": True,
    }
    print(f"== posting paste-file → {args.cc_url}/api/paste-file")
    print(f"   src: {src}  ({len(content):,} bytes / "
          f"{content.count(chr(10))+1} lines)")
    print(f"   path on host: {args.path}")
    print(f"   (this takes a while — BT HID is ~30–50 cps)")

    try:
        r = requests.post(
            f"{args.cc_url}/api/paste-file",
            json=body, timeout=1800,
        )
    except requests.RequestException as e:
        print(f"!! request failed: {e}", file=sys.stderr)
        return 3
    if not r.ok:
        print(f"!! {r.status_code}: {r.text}", file=sys.stderr)
        return 4
    data = r.json()

    # (1) SHA verdict — the cryptographic proof.
    v = data.get("verify") or {}
    sha_match = bool(v.get("match"))
    print()
    print("── SHA-256 verdict ──")
    if sha_match:
        print(f"  ✓ MATCH — host file is byte-identical")
        print(f"    local : {v.get('local_sha', '?')}")
        rounds = v.get("rounds", [])
        if rounds:
            print(f"    after {len(rounds)} round(s)")
    else:
        print("  ✗ MISMATCH")
        for rd in v.get("rounds", []):
            tag = "✓" if rd.get("match") else "✗"
            line = f"    round {rd.get('round')}: {tag}"
            if rd.get("bad_indices"):
                line += f"  bad={rd['bad_indices'][:12]}"
            if rd.get("abort_reason"):
                line += f"  abort: {rd['abort_reason']}"
            print(line)

    # (2) `more` round-trip diff.
    rb = data.get("body_readback") or {}
    recovered = rb.get("recovered_text", "")
    print()
    print("── `more` round-trip ──")
    if not recovered:
        print("  (no body readback — was body_readback disabled?)")
        return 0 if sha_match else 5
    print(f"  pages: {rb.get('pages')}")
    print(f"  similarity: {rb.get('similarity')}")
    print(f"  expected={rb.get('expected_chars')}c, "
          f"ocr={rb.get('ocr_chars')}c")

    # Run a real diff against a normalised version of the source
    # so we're comparing apples to apples (the endpoint normalises
    # the OCR side; we must too).
    import re
    _MORE = re.compile(
        r"^\s*-{1,3}\s*More\s*-{1,3}\s*\(\s*\d+\s*%\s*\)\s*$",
        re.IGNORECASE,
    )

    def _norm(s: str) -> str:
        return "\n".join(
            ln.rstrip()
            for ln in s.replace("\r", "").split("\n")
            if ln.strip() and not _MORE.match(ln.rstrip())
        ).strip()

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        orig_n = td_p / "original.norm"
        recv_n = td_p / "recovered.norm"
        orig_n.write_text(_norm(content))
        recv_n.write_text(recovered)
        proc = subprocess.run(
            ["diff", "-u", str(orig_n), str(recv_n)],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            print("  ✓ ZERO DIFF — body recovered identically.")
        else:
            print("  ≈ differences below "
                  "(OCR noise — SHA verdict is authoritative):")
            tail = proc.stdout
            if len(tail) > 4000:
                tail = tail[:4000] + "\n…(truncated)…"
            print(tail)

    # Exit code mirrors the cryptographic verdict (the real one).
    return 0 if sha_match else 1


if __name__ == "__main__":
    sys.exit(main())
