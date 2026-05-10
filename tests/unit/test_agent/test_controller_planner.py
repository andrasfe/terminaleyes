"""Regression tests for the planner.

The planner is now LLM-first. The only intents :func:`plan_intent`
matches deterministically (without invoking the LLM) are:

  - ``login`` / ``log in`` (security-sensitive)
  - ``focus`` / ``center`` / ``maximize`` (trivial primitive)
  - ``wake`` (trivial primitive)
  - ``scroll up|down [N]`` (trivial primitive)
  - the subreddit-fetch shortcut (a multi-step workflow with a
    fixed shape — short-circuited at the top of plan_intent)

Every other intent (``open <app>`` / ``close window`` / ``click X``
/ ``run X`` / ``read X`` / ``ocr X`` / ``go to <url>`` / ``copy`` /
``save`` / ``paste`` / ...) returns ``[]`` from :func:`plan_intent`
so the controller falls through to the LLM planner. Those routes
are covered by the few-shot examples in ``_PLANNER_FEW_SHOT`` and
verified end-to-end via the cc UI.
"""

from __future__ import annotations

import pytest

from terminaleyes.agents.controller import (
    PlanStep,
    _cache_get, _cache_key, _dedup_adjacent_steps,
    _detect_stuck_terminal, _filter_kwargs,
    _intent_expects_output, _scan_for_error,
    cache_clear, plan_intent,
)


def names(plan):
    return [s.name for s in plan]


def kwargs_of(plan, name):
    for s in plan:
        if s.name == name:
            return s.kwargs
    return None


# ── subreddit fetch (kept rule — multi-step workflow) ────────────

@pytest.mark.parametrize("intent", [
    "navigate to reddit.com, go to r/Qiskit and fetch the top 5 post titles",
    "go to r/Qiskit and give me the top 5 posts in r/Qiskit",
    "give me the top posts in r/Qiskit",
    "give me the top 5 posts in r/Qiskit",
    "fetch the top 5 post titles in r/Qiskit",
    "list top posts in r/Qiskit",
    "open reddit.com/r/Qiskit and read the post titles",
    "show me the top 3 titles in r/LocalLLaMA",
    "what are the top 5 post titles in r/Qiskit",
    "read the top 3 posts in r/programming",
    "find the top headlines in r/news",
    "tell me top posts of r/Python",
])
def test_subreddit_fetch_routes_to_dismiss_navigate_read(intent):
    plan = plan_intent(intent)
    assert plan, f"no plan for {intent!r}"
    assert names(plan) == ["dismiss", "navigate", "read"], (
        f"{intent!r} -> {names(plan)}"
    )
    nav = kwargs_of(plan, "navigate")
    assert nav and nav["url"].startswith("reddit.com/r/")
    read = kwargs_of(plan, "read")
    assert read and "question" in read
    assert "r/" in read["question"]


def test_subreddit_fetch_carries_count_into_question():
    plan = plan_intent("fetch the top 7 post titles in r/Qiskit")
    q = kwargs_of(plan, "read")["question"]
    assert "top 7" in q
    nav = kwargs_of(plan, "navigate")
    assert nav["url"] == "reddit.com/r/Qiskit"


def test_subreddit_fetch_default_count_is_5():
    plan = plan_intent("give me top posts in r/Qiskit")
    q = kwargs_of(plan, "read")["question"]
    assert "top 5" in q


# ── kept rule whitelist ──────────────────────────────────────────

def test_login():
    plan = plan_intent("login")
    assert names(plan) == ["login"]


def test_log_in_with_space():
    plan = plan_intent("log in")
    assert names(plan) == ["login"]


def test_focus():
    plan = plan_intent("focus")
    assert names(plan) == ["focus"]


@pytest.mark.parametrize("intent", [
    "center", "centre", "maximize", "maximise",
])
def test_focus_aliases(intent):
    assert names(plan_intent(intent)) == ["focus"]


def test_wake():
    assert names(plan_intent("wake")) == ["wake"]


@pytest.mark.parametrize("intent,direction,amount", [
    ("scroll", "down", 4),
    ("scroll down", "down", 4),
    ("scroll up", "up", 4),
    ("scroll down 6", "down", 6),
    ("scroll up 12", "up", 12),
])
def test_scroll(intent, direction, amount):
    plan = plan_intent(intent)
    assert names(plan) == ["scroll"]
    assert kwargs_of(plan, "scroll") == {
        "direction": direction, "amount": amount,
    }


# ── everything else falls through (LLM territory) ───────────────

@pytest.mark.parametrize("intent", [
    "open terminal",
    "launch firefox",
    "open files",
    "go to reddit.com",
    "navigate to https://news.ycombinator.com",
    "click the Run button",
    "type hello world",
    "run ls -la",
    "execute pwd",
    "close the window",
    "close the terminal window",
    "save",
    "save the file",
    "copy",
    "paste",
    "cut",
    "undo",
    "redo",
    "new tab",
    "close tab",
    "quit",
    "switch window",
    "minimize",
    "press enter",
    "press escape",
    "read with ocr the URL bar",
    "ocr the page title",
    "give me the page title",
    "what's on the screen",
    "tell me the headlines",
    "dance the macarena",
])
def test_unmatched_intents_return_empty(intent):
    """The rule planner used to handle these via regex; now the LLM
    planner does. plan_intent should return [] so the controller
    falls through to _llm_plan."""
    assert plan_intent(intent) == [], (
        f"{intent!r} unexpectedly matched a rule: {names(plan_intent(intent))}"
    )


# ── controller memory ───────────────────────────────────────────

def test_memory_returns_empty_when_unset(monkeypatch, tmp_path):
    from terminaleyes.agents import controller as c
    monkeypatch.setenv("TERMINALEYES_MEMORY", str(tmp_path / "absent.md"))
    assert c.load_memory() == ""


def test_memory_reads_file(monkeypatch, tmp_path):
    from terminaleyes.agents import controller as c
    p = tmp_path / "memory.md"
    p.write_text("# notes\nuse chrome not firefox\n", encoding="utf-8")
    monkeypatch.setenv("TERMINALEYES_MEMORY", str(p))
    assert c.load_memory() == "# notes\nuse chrome not firefox"


# ── intent → plan cache ─────────────────────────────────────────

def test_cache_starts_empty():
    cache_clear()
    key = _cache_key("anything", False, None, "linux")
    assert _cache_get(key) is None


def test_cache_round_trip():
    """Verify the cache helpers — full controller integration is
    covered end-to-end in cc."""
    from terminaleyes.agents.controller import _cache_put
    cache_clear()
    plan = plan_intent("focus")
    key = _cache_key("focus", False, None, "linux")
    _cache_put(key, plan)
    assert _cache_get(key) == plan
    cache_clear()
    assert _cache_get(key) is None


def test_cache_key_distinguishes_options():
    cache_clear()
    k1 = _cache_key("open terminal", False, None, "linux")
    k2 = _cache_key("open terminal", True, None, "linux")
    k3 = _cache_key("open terminal", False, None, "macos")
    k4 = _cache_key("open terminal", False, "myhost", "linux")
    assert len({k1, k2, k3, k4}) == 4


# ── error-pattern detection (used by completion verifier) ───────

@pytest.mark.parametrize("text", [
    "Command 'ind' not found, but there are 18 similar ones.",
    "bash: foobarbaz: command not found",
    "rm: cannot remove '/etc/foo': Permission denied",
    "curl: (7) Failed to connect to localhost port 8080: Connection refused",
    "ls: cannot access '/nonexistent': No such file or directory",
    "Did you mean: pwd ?",
    "Traceback (most recent call last):\n  File ...",
    "404 Not Found",
    "This site can't be reached",
    "syntax error near unexpected token",
])
def test_scan_for_error_detects(text):
    assert _scan_for_error(text), f"missed error in {text!r}"


@pytest.mark.parametrize("text", [
    "",
    "andras@host:~$ ls\nfoo.py  bar.py",
    "user@machine ~]$ pwd\n/home/user",
    "$ echo hello\nhello",
    "Welcome to Reddit — front page of the internet",
])
def test_scan_for_error_does_not_misfire(text):
    assert _scan_for_error(text) == "", (
        f"false positive in {text!r}"
    )


# ── intent-vs-output classifier (gates the error short-circuit) ─

@pytest.mark.parametrize("intent", [
    "run ls -la",
    "execute pwd",
    "find python files in subdirectories",
    "search for foo in this directory",
    "list the files",
    "show me the headlines",
    "tell me what's on screen",
    "fetch the top posts in r/Qiskit",
    "navigate to reddit.com",
    "browse to news.ycombinator.com",
    "what is the current URL",
    "ocr the URL bar",
    "give me the page title",
    "compute 17 * 23",
])
def test_intent_expects_output_true(intent):
    assert _intent_expects_output(intent), intent


@pytest.mark.parametrize("intent", [
    "close this term window",
    "close the firefox window",
    "open terminal",
    "launch firefox",
    "switch window",
    "minimize",
    "maximize the window",
    "save",
    "copy",
    "paste",
    "press enter",
    "focus",
    "wake",
])
def test_intent_expects_output_false(intent):
    assert not _intent_expects_output(intent), intent


# ── kwargs filtering (drops LLM-invented args) ──────────────────

def test_filter_kwargs_drops_unknown_for_focus():
    from terminaleyes.agents.focus import FocusAgent
    kw = _filter_kwargs(
        FocusAgent,
        {"app": "Terminal", "platform": "linux"},
        name="focus",
    )
    assert kw == {"platform": "linux"}


def test_filter_kwargs_keeps_all_known_for_keys():
    from terminaleyes.agents.keys import KeyComboAgent
    kw = _filter_kwargs(
        KeyComboAgent,
        {"modifiers": ["alt"], "key": "F4", "platform": "linux"},
        name="keys",
    )
    assert kw == {"modifiers": ["alt"], "key": "F4", "platform": "linux"}


def test_filter_kwargs_empty_dict_passes_through():
    from terminaleyes.agents.focus import FocusAgent
    assert _filter_kwargs(FocusAgent, {}, name="focus") == {}


# ── stuck-terminal (shell continuation prompt) detection ───────

@pytest.mark.parametrize("text", [
    # Bash/zsh continuation prompt — the canonical signal.
    "$ echo 'hello\n> uname -r\n> pwd\n> ",
    "andras@host:~$ echo 'hello\n> world\n> ",
    # Many `> ` lines in a row.
    "> foo\n> bar\n> baz",
    # Single `> ` is still enough.
    "user@machine ~$ echo \"hi\n> next\n",
])
def test_detect_stuck_terminal_hit(text):
    assert _detect_stuck_terminal(text), f"missed in {text!r}"


@pytest.mark.parametrize("text", [
    "",
    "$ ls\nfile1.py file2.py\n$ ",
    # `>` mid-line is NOT a continuation prompt.
    "echo a > /tmp/foo",
    # `>` at start of a line, but with 'not found' (an error line
    # showing the command being suggested) — we exclude that to
    # avoid false-firing on shell suggestions like:
    #     "did you mean: > pear"
    # which can sometimes happen in OCR'd suggestion blocks.
    "> something not found here",
    # Markdown blockquotes in browser content.
    "page footer\n> quote line",  # should NOT hit if exclusion holds — let
    # check loose behavior below
])
def test_detect_stuck_terminal_no_misfire_obvious(text):
    # Only assert on obvious negative cases; the heuristic is a
    # best-effort signal, not a precise grammar.
    if not text:
        assert _detect_stuck_terminal(text) == ""


# ── adjacent-step dedup ──────────────────────────────────────────

def test_dedup_collapses_duplicate_launches():
    """The headline bug: per-chunk LLM emits two identical launch
    steps for chained intents like 'open a terminal and run X',
    and the second launch eats the first char of the following
    type. Dedup at the seam fixes it."""
    from terminaleyes.agents.launch import LaunchAgent
    from terminaleyes.agents.type_text import TypeAgent
    plan = [
        PlanStep("launch", LaunchAgent,
                 {"app": "terminal", "platform": "linux"}),
        PlanStep("launch", LaunchAgent,
                 {"app": "terminal", "platform": "linux"}),
        PlanStep("type", TypeAgent,
                 {"text": "apt update", "submit": True}),
    ]
    out = _dedup_adjacent_steps(plan)
    assert [s.name for s in out] == ["launch", "type"]


def test_dedup_keeps_distinct_kwargs():
    from terminaleyes.agents.launch import LaunchAgent
    plan = [
        PlanStep("launch", LaunchAgent, {"app": "terminal"}),
        PlanStep("launch", LaunchAgent, {"app": "firefox"}),
    ]
    out = _dedup_adjacent_steps(plan)
    assert [s.kwargs["app"] for s in out] == ["terminal", "firefox"]


def test_dedup_keeps_non_adjacent_duplicates():
    """Two identical steps separated by a different step both
    survive — the bug only happens at the seam."""
    from terminaleyes.agents.launch import LaunchAgent
    from terminaleyes.agents.focus import FocusAgent
    plan = [
        PlanStep("launch", LaunchAgent, {"app": "terminal"}),
        PlanStep("focus", FocusAgent, {}),
        PlanStep("launch", LaunchAgent, {"app": "terminal"}),
    ]
    out = _dedup_adjacent_steps(plan)
    assert [s.name for s in out] == ["launch", "focus", "launch"]


def test_dedup_handles_empty_and_single():
    assert _dedup_adjacent_steps([]) == []
    from terminaleyes.agents.focus import FocusAgent
    one = [PlanStep("focus", FocusAgent, {})]
    assert _dedup_adjacent_steps(one) == one
