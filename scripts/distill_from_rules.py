#!/usr/bin/env python3
"""distill_from_rules.py — bootstrap a training corpus by running the
working rules+LLM planner over a diverse intent list.

The step logger writes a row to ``<output_dir>/steps.jsonl`` on every
agent call regardless of planner, so simply iterating a list of
intents through ``terminaleyes do`` produces a labelled dataset for
free. This is functionally distillation from the existing planner:
the rule planner + LM Studio LLM emit the gold actions, and the
trained VLA adapter learns to mimic them.

Caveats baked into the default intent list:
  * No `__EXEC_SCRIPT__` envelopes — those pollute the model's
    format prior (build_ml_dataset.py filters them out anyway).
  * No `lock the screen` mid-batch unless the next intent is an
    `unlock` with --vault desktop — DPMS suspends the monitor and
    LoginAgent's wake step doesn't reliably catch it inside a
    scripted run.

Usage::

    # Use built-in intent list (~100 intents)
    python scripts/distill_from_rules.py

    # Custom list
    python scripts/distill_from_rules.py --intents path/to/list.txt

    # Dry run — show what would be sent
    python scripts/distill_from_rules.py --print-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ────────────────────────────────────────────────────────────────
# Built-in intent list. Designed for breadth across the agent
# REGISTRY without triggering the known failure modes (envelope
# intents, mid-batch lock cycles, etc.).
# ────────────────────────────────────────────────────────────────
DEFAULT_INTENTS: list[str] = [
    # launch (~15) — covers app aliases
    "open a terminal",
    "open the terminal",
    "open a new terminal",
    "open the calculator",
    "open the calc app",
    "open libreoffice writer",
    "open writer",
    "open libreoffice calc",
    "open firefox",
    "open the firefox browser",
    "open google chrome",
    "open the files app",
    "open file manager",
    "open nautilus",
    "open the text editor",

    # keys close-window (~10) — pairs with launches above
    "close the terminal window",
    "close the calculator window",
    "close the libreoffice writer window",
    "close the firefox window",
    "close the files window",
    "close the calc window",
    "close the current window",
    "close this window",

    # shell_run (~30) — varied commands, always paired with launch
    "open a terminal and run pwd",
    "open a terminal and run whoami",
    "open a terminal and run hostname",
    "open a terminal and run date",
    "open a terminal and run uname -r",
    "open a terminal and run uname -a",
    "open a terminal and run uname -m",
    "open a terminal and run echo hello",
    "open a terminal and run echo world",
    "open a terminal and run echo hello world",
    "open a terminal and run ls /tmp",
    "open a terminal and run ls /etc | head -5",
    "open a terminal and run ls -la ~ | head -5",
    "open a terminal and run ls /usr/bin | head",
    "open a terminal and run cat /etc/os-release | head -3",
    "open a terminal and run cat /proc/cpuinfo | head -3",
    "open a terminal and run df -h /",
    "open a terminal and run free -h | head",
    "open a terminal and run uptime",
    "open a terminal and run who",
    "open a terminal and run id",
    "open a terminal and run echo $HOME",
    "open a terminal and run echo $SHELL",
    "open a terminal and run env | head -5",
    "open a terminal and run ps aux | head -5",
    "open a terminal and run printenv USER",
    "open a terminal and run ls -la /var/log | head",
    "open a terminal and run wc -l /etc/passwd",
    "open a terminal and run head -3 /etc/hostname",
    "open a terminal and run echo done && date",

    # script (~10)
    "open a terminal and run this script: echo hello",
    "open a terminal and run this script: pwd",
    "open a terminal and run this script: date",
    "open a terminal and run this script: uname -a",
    "open a terminal and run this script: ls /tmp | head",
    "open a terminal and run this script: echo a && echo b",
    "open a terminal and run this script: for i in 1 2; do echo $i; done",
    "open a terminal and run this script: hostname",
    "open a terminal and run this script: id",
    "open a terminal and run this script: echo done",

    # set_prompt (~6)
    "open a terminal and change the bash prompt to mini1",
    "open a terminal and change the bash prompt to dev",
    "open a terminal and change the bash prompt to box1",
    "open a terminal and change the bash prompt to host",
    "open a terminal and set the bash prompt to work",
    "open a terminal and rename the bash prompt to local",

    # navigate (~15) — bare domains; the URL-bar verifier passes
    # these cleanly more often than path-bearing URLs.
    "go to example.com",
    "go to wikipedia.org",
    "go to news.ycombinator.com",
    "go to duckduckgo.com",
    "go to github.com",
    "go to xkcd.com",
    "go to apple.com",
    "navigate to example.com",
    "navigate to wikipedia.org",
    "navigate to duckduckgo.com",
    "navigate to github.com",
    "open reddit.com in the browser",
    "open httpbin.org",
    "open archive.org",
    "open kernel.org",

    # read (~10) — common question shapes; outputs vary, so the
    # gold labels will too. That's fine for distillation.
    "what is on the screen right now?",
    "what is the title of the current window?",
    "what URL is in the address bar?",
    "what user am I logged in as?",
    "what is the kernel version on this machine?",
    "what is the current working directory?",
    "what time is it on this machine?",
    "what is the hostname of this machine?",
    "read the contents of the current page",
    "tell me what apps are visible on the screen",
]


def run_intent(
    intent: str, *, route: str, vault: str | None = None,
) -> bool:
    """Fire one intent through ``terminaleyes do``. Returns True iff
    the run logged a success. Errors are tolerated — failed runs
    still produce step records useful for SFT."""
    if vault:
        cmd = [
            str(Path(__file__).resolve().parent / "te-secrets"),
            "run", vault, intent,
        ]
    else:
        cmd = [
            str(Path(__file__).resolve().parent / "te-secrets"),
            "exec", "terminaleyes", "do", intent,
        ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return False
    tail = (out.stdout or "").strip().splitlines()[-3:]
    blob = "\n".join(tail)
    return "✓ cc run succeeded" in blob or "✓ Controller succeeded" in blob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--intents", type=Path, default=None,
        help="Path to a file of intents (one per line). Defaults to "
             "the built-in DEFAULT_INTENTS list.",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Cap the number of intents to fire (0 = all).",
    )
    ap.add_argument(
        "--sleep", type=float, default=2.0,
        help="Sleep between intents to let cc settle.",
    )
    ap.add_argument(
        "--print-only", action="store_true",
        help="Just print the intent list and exit.",
    )
    args = ap.parse_args()

    if args.intents:
        intents = [
            ln.strip() for ln in args.intents.read_text("utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    else:
        intents = list(DEFAULT_INTENTS)
    if args.limit:
        intents = intents[: args.limit]

    if args.print_only:
        for i, it in enumerate(intents, 1):
            print(f"{i:>3}. {it}")
        return 0

    print(f"distilling over {len(intents)} intent(s)\n")
    n_ok = n_fail = 0
    for i, intent in enumerate(intents, 1):
        printable = intent[:78]
        sys.stdout.write(
            f"  [{i:>3}/{len(intents)}] {printable:<78} "
        )
        sys.stdout.flush()
        ok = run_intent(intent, route="auto")
        if ok:
            n_ok += 1; sys.stdout.write("✓\n")
        else:
            n_fail += 1; sys.stdout.write("✗\n")
        sys.stdout.flush()
        time.sleep(args.sleep)

    print()
    print(f"distill done: {n_ok} ✓ / {n_fail} ✗")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
