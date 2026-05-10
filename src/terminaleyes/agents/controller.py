"""ControllerAgent — top-level orchestrator.

Takes a high-level intent (free-form English) and decomposes it into a
sequence of agent invocations. Two-phase planning:

  1. **Rule-based router** (default, fast, no LLM). Pattern-matches
     the intent against a small handful of common shapes:
       - ``login``                                → [LoginAgent]
       - ``focus`` / ``center``                   → [FocusAgent]
       - ``go to URL`` / ``navigate to URL``      → [FocusAgent, NavigateAgent]
       - ``open URL``                             → [FocusAgent, NavigateAgent]
       - ``click X``                              → [FocusAgent, SearchAgent]
       - ``type X``                               → [TypeAgent]
       - ``login and …``                          → [LoginAgent, then route the rest]
       - ``focus and …``                          → [FocusAgent, then route the rest]

  2. **LLM-planner fallback** (TODO). When no rule matches, ask the
     multimodal model to produce a plan referencing the registered
     agents. Validated against the registry; rejected if it names
     unknown actions. Not implemented in this commit.

Defaults that make the controller "safe":
  - Click-like steps are prefixed with FocusAgent unless the user
    passes ``no_focus=True`` (CLI ``--no-focus``).
  - Hard cap on total steps to prevent runaway planning.
  - Each step's :class:`Outcome` is collected; the final outcome
    surfaces the full audit trail in ``data['results']``.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.click import ClickAgent
from terminaleyes.agents.cursor import CursorAgent
from terminaleyes.agents.dismiss import DismissModalsAgent
from terminaleyes.agents.focus import FocusAgent
from terminaleyes.agents.keys import KeyComboAgent
from terminaleyes.agents.launch import LaunchAgent
from terminaleyes.agents.login import LoginAgent
from terminaleyes.agents.navigate import NavigateAgent
from terminaleyes.agents.ocr import OcrAgent
from terminaleyes.agents.read import ReadAgent
from terminaleyes.agents.scroll import ScrollAgent
from terminaleyes.agents.target import TargetAgent
from terminaleyes.agents.type_text import TypeAgent
from terminaleyes.agents.verify import VerifyAgent
from terminaleyes.agents.wake import WakeAgent

logger = logging.getLogger(__name__)


# Hard step cap so a runaway plan can't lock up the target.
MAX_STEPS = 12


# Few-shot examples appended to the LLM-planner prompt so a flaky
# model (Gemini Flash Lite, Nemotron-Nano, etc.) gets concrete
# patterns to imitate. Each block is a single user intent → JSON
# plan; keep them short, generic, and varied so the model
# generalises rather than parrots a specific verb.
_PLANNER_FEW_SHOT = (
    "Examples — study how each intent decomposes into 1–5 agent "
    "calls. Reply ONLY in the same JSON shape:\n\n"
    "Intent: open a terminal and run ls -la\n"
    'Reply: {"plan": [\n'
    '  {"name": "launch", "kwargs": {"app": "terminal", "platform": "linux"}},\n'
    '  {"name": "type",   "kwargs": {"text": "ls -la", "submit": true}}\n'
    "]}\n\n"
    "Intent: open the calculator and compute 17 * 23\n"
    'Reply: {"plan": [\n'
    '  {"name": "launch", "kwargs": {"app": "calculator", "platform": "linux"}},\n'
    '  {"name": "type",   "kwargs": {"text": "17*23", "submit": true}}\n'
    "]}\n\n"
    "Intent: close the firefox window\n"
    'Reply: {"plan": [\n'
    '  {"name": "keys", "kwargs": {"modifiers": ["alt"], "key": "F4"}}\n'
    "]}\n\n"
    "Intent: open files and search for the downloads folder\n"
    'Reply: {"plan": [\n'
    '  {"name": "launch", "kwargs": {"app": "files", "platform": "linux"}},\n'
    '  {"name": "keys",   "kwargs": {"modifiers": ["ctrl"], "key": "f"}},\n'
    '  {"name": "type",   "kwargs": {"text": "Downloads", "submit": true}}\n'
    "]}\n\n"
    "Intent: read what's in the URL bar\n"
    'Reply: {"plan": [\n'
    '  {"name": "ocr", "kwargs": {"region": "url_bar"}}\n'
    "]}\n\n"
    "Intent: open a terminal and check the kernel version\n"
    'Reply: {"plan": [\n'
    '  {"name": "launch", "kwargs": {"app": "terminal", "platform": "linux"}},\n'
    '  {"name": "type",   "kwargs": {"text": "uname -r", "submit": true}}\n'
    "]}\n\n"
    "Intent: navigate to news.ycombinator.com and tell me the top 3 headlines\n"
    'Reply: {"plan": [\n'
    '  {"name": "navigate", "kwargs": {"url": "news.ycombinator.com", "platform": "linux"}},\n'
    '  {"name": "read",     "kwargs": {"question": "List the top 3 article headlines on this page, one per line."}}\n'
    "]}\n\n"
    "Intent: close the terminal window\n"
    'Reply: {"plan": [\n'
    '  {"name": "keys", "kwargs": {"modifiers": ["alt"], "key": "F4"}}\n'
    "]}\n\n"
    "Intent: save the file\n"
    'Reply: {"plan": [\n'
    '  {"name": "keys", "kwargs": {"modifiers": ["ctrl"], "key": "s"}}\n'
    "]}\n\n"
    "Intent: click the Run button\n"
    'Reply: {"plan": [\n'
    '  {"name": "click", "kwargs": {"target": "the Run button"}}\n'
    "]}\n\n"
    "Intent: type hello world\n"
    'Reply: {"plan": [\n'
    '  {"name": "type", "kwargs": {"text": "hello world", "submit": false}}\n'
    "]}\n\n"
    "Intent: read what's in the URL bar via OCR\n"
    'Reply: {"plan": [\n'
    '  {"name": "ocr", "kwargs": {"region": "url_bar"}}\n'
    "]}\n"
)


# ─────────────── intent → plan cache ───────────────
#
# Keyed by ``(intent, no_focus, vault_name, platform)``. Stores
# (rule_or_LLM-validated) plans so a repeated intent skips the LLM
# round-trip on subsequent runs. Module-level so it survives across
# ControllerAgent instances (each cc run builds a fresh agent), but
# obviously not across cc restarts. Cap is generous; we expect at
# most a few hundred unique intents over a cc lifetime.
_PLAN_CACHE: dict[tuple, list[PlanStep]] = {}
_PLAN_CACHE_MAX = 256


def _cache_key(
    intent: str, no_focus: bool, vault_name: str | None, platform: str,
) -> tuple:
    return (
        intent.strip().lower(), bool(no_focus),
        vault_name or "", platform,
    )


def _cache_get(key: tuple) -> list[PlanStep] | None:
    return _PLAN_CACHE.get(key)


def _cache_put(key: tuple, plan: list[PlanStep]) -> None:
    if not plan:
        return
    if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
        # Drop the oldest entry — dicts preserve insertion order.
        try:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
        except StopIteration:
            pass
    _PLAN_CACHE[key] = plan


# ─────────────── error-pattern detection ───────────────
#
# Patterns that almost-always indicate a failed action when they
# appear in OCR'd screen text. Used by the final-state verifier
# to short-circuit to FAILURE before bothering the LLM — small
# vision models routinely overlook these in pattern-match mode.
# Patterns are case-insensitive and matched anywhere in the
# extracted screen text.
_ERROR_MARKERS: tuple[re.Pattern, ...] = (
    re.compile(r"command [^\n]{0,40}not found", re.I),
    re.compile(r"\b(?:no such file or directory|"
               r"permission denied|"
               r"connection refused|"
               r"connection timed out|"
               r"network is unreachable|"
               r"address already in use)\b", re.I),
    re.compile(r"\bdid you mean\b", re.I),
    re.compile(r"\bsimilar (?:commands?|programs?)\b", re.I),
    re.compile(r"\b(?:syntax error|parse error|"
               r"unexpected token|unrecognized argument)\b", re.I),
    re.compile(r"traceback \(most recent call last\)", re.I),
    re.compile(r"\bstack trace\b", re.I),
    re.compile(r"\b(?:404 not found|403 forbidden|"
               r"500 internal server error|502 bad gateway|"
               r"503 service unavailable|504 gateway timeout)\b",
               re.I),
    re.compile(r"\bthis site can.?t be reached\b", re.I),
    re.compile(r"\b(?:failed to|unable to|cannot) "
               r"(?:open|read|write|find|connect|load|launch|"
               r"start|execute)\b", re.I),
)


def _scan_for_error(text: str) -> str:
    """Return the first error-marker substring found in ``text``,
    or ``""`` if none. Case-insensitive."""
    if not text:
        return ""
    for pat in _ERROR_MARKERS:
        m = pat.search(text)
        if m:
            # Return up to ~80 chars of surrounding context for
            # the verdict's reason field.
            start = max(0, m.start() - 8)
            end = min(len(text), m.end() + 40)
            return text[start:end].strip()
    return ""


# Verbs that imply the action should produce VISIBLE OUTPUT on
# the screen (command output, page content, search results, an
# answer). For these intents, an error message on screen is a
# strong signal of failure. For NON-output intents (close /
# open / switch / minimise), an error message on screen is most
# likely from a PRIOR run and doesn't reflect on the current one.
_OUTPUT_VERBS = re.compile(
    r"\b(?:run|execute|exec|find|search|list|show|tell|fetch|get|"
    r"give|read|what(?:'s|s|\s+is|\s+are)?|check|print|ls|grep|"
    r"cat|navigate|browse|extract|ocr|summari[zs]e|"
    r"compute|calculate|count)\b",
    re.I,
)


def _intent_expects_output(intent: str) -> bool:
    """True when the user's intent implies the screen should show
    fresh output (command output, page content, search results).
    For these intents, a visible error decisively means failure.
    For non-output intents (close/open/switch/minimise/etc.), pre-
    existing error text on the screen is most likely from a prior
    run and is NOT load-bearing for the current verdict — let the
    LLM judge based on whether the requested state change happened.
    """
    return bool(_OUTPUT_VERBS.search(intent or ""))


def _dedup_adjacent_steps(plan: list[PlanStep]) -> list[PlanStep]:
    """Collapse adjacent steps with identical ``(name, kwargs)``.

    Used after stitching per-chunk LLM plans together: the LLM
    often emits a ``launch terminal`` step for both "open a
    terminal" and "run X in a terminal", which means the
    second launch reopens the app mid-plan and eats the first
    keystroke of the following type step (we observed `find` →
    `ind`, `apt update` → `pt update`, etc.). Dedup at the seam
    is the cleanest fix.
    """
    if len(plan) < 2:
        return list(plan)
    out: list[PlanStep] = []
    for s in plan:
        if out and out[-1].name == s.name and out[-1].kwargs == s.kwargs:
            logger.info(
                "Plan dedup: dropping duplicate adjacent %s step",
                s.name,
            )
            continue
        out.append(s)
    return out


def _filter_kwargs(
    agent_cls: type, kwargs: dict, *, name: str,
) -> dict:
    """Drop kwargs the agent's ``run()`` method doesn't accept.

    The LLM planner frequently invents arguments (e.g. ``focus
    {"app": "Terminal"}`` — :class:`FocusAgent` has no ``app``
    param) which would raise ``TypeError`` at dispatch time.
    Inspecting the signature once per call is cheap and keeps
    the planner generic — every agent benefits without per-name
    special-casing.

    A ``run()`` that accepts ``**kwargs`` is treated as accepting
    everything (rare in this codebase but cheap to honour).
    """
    import inspect
    if not isinstance(kwargs, dict) or not kwargs:
        return kwargs or {}
    try:
        sig = inspect.signature(agent_cls.run)
    except (TypeError, ValueError):
        return kwargs
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs
    accepted = {
        p.name for p in sig.parameters.values()
        if p.name != "self" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    extra = [k for k in kwargs if k not in accepted]
    if not extra:
        return kwargs
    logger.info(
        "Dropping %d unknown kwarg(s) from %s: %s",
        len(extra), name, extra,
    )
    return {k: v for k, v in kwargs.items() if k in accepted}


def cache_clear() -> None:
    """Drop every cached plan. Exposed for tests / a future
    ``terminaleyes plans clear`` CLI verb."""
    _PLAN_CACHE.clear()


# Long-lived notes the user wants the controller to remember across
# runs. Plain markdown — read on every run, never written by the
# controller itself. Path can be overridden with the env var.
DEFAULT_MEMORY_PATH = (
    Path.home() / ".local" / "share" / "terminaleyes" / "memory.md"
)


def _memory_path() -> Path:
    env = os.environ.get("TERMINALEYES_MEMORY")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_MEMORY_PATH


def load_memory() -> str:
    """Read the controller's memory file. Returns "" if missing or
    unreadable. Caller decides how to use the contents (typically
    injected into the LLM-planner prompt and printed at run start)."""
    p = _memory_path()
    try:
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.debug("could not read memory %s: %s", p, e)
    return ""


@dataclass
class PlanStep:
    name: str
    agent_cls: type
    kwargs: dict[str, Any] = field(default_factory=dict)
    # When True, a failure of this step logs a warning and the
    # controller continues to the next step instead of aborting.
    # Used for "soft" preconditions like the auto-prepended
    # FocusAgent before navigate/click — the next agent has its own
    # pre-flight (e.g. NavigateAgent's browser check) that handles
    # the same concern, so a strict stop on focus failure is wrong.
    best_effort: bool = False


@dataclass
class ControllerOutcome(Outcome):
    pass


# ───────────────── agent registry ─────────────────

REGISTRY: dict[str, tuple[type, str]] = {
    "wake":     (WakeAgent,     "wake the remote screen / dismiss screensaver"),
    "verify":   (VerifyAgent,   "ask a yes/no visual question about the screen"),
    "dismiss":  (DismissModalsAgent,
                 "detect and close modal dialogs / popups blocking the UI"),
    "focus":    (FocusAgent,    "centre and maximise the foreground app"),
    "launch":   (LaunchAgent,
                 "open a desktop app by name on the target. "
                 "kwargs: app (str — terminal/files/calculator/firefox/"
                 "chrome/...; aliases like 'the terminal' / 'shell' "
                 "are accepted; unknown names pass through verbatim), "
                 "platform (linux/macos). Verifies via top-bar OCR "
                 "with a multimodal-Verify fallback. Returns the "
                 "canonical typed name in data['app']."),
    "login":    (LoginAgent,    "wake + verify-login + type password from vault"),
    "type":     (TypeAgent,     "type text (optional secret + Enter)"),
    "keys":     (KeyComboAgent,
                 "send a keyboard shortcut to the target. "
                 "kwargs: modifiers (list of 'ctrl'/'alt'/'shift'/"
                 "'super'/'cmd'; empty for a bare key), key (the "
                 "non-modifier key, e.g. 'F4', 's', 'Tab', 'Up'; "
                 "empty for a modifier-only tap). PREFER this over "
                 "'click' for actions that have a known shortcut: "
                 "Alt+F4 to close a window, Ctrl+S to save, Ctrl+W "
                 "to close a tab, Ctrl+T new tab, Ctrl+C/V/X copy/"
                 "paste/cut, Ctrl+Z undo, Ctrl+Q quit, Alt+Tab "
                 "switch window, Super+Up maximise, Super+H "
                 "minimise."),
    "navigate": (NavigateAgent, "type a URL into a browser address bar (browser-aware)"),
    "click":    (ClickAgent,    "find a target by description; scroll-and-retry if not visible"),
    "scroll":   (ScrollAgent,   "scroll up/down via the mouse wheel"),
    "read":     (ReadAgent,
                 "ask the multimodal model an open question about the current screen "
                 "(returns the answer text in outcome.data['answer'])"),
    "ocr":      (OcrAgent,
                 "extract plain text from the screen via the OCR-"
                 "specialised vision model in ctx.ocr_model "
                 "(default 'nanonets-ocr-s' on LM Studio). "
                 "kwargs: region (preset: url_bar/title/top_bar/footer/...), "
                 "crop (explicit (x0,y0,x1,y1) fractions), "
                 "target (natural-language hint used ONLY to auto-pick a "
                 "region preset — never filters returned text). "
                 "Returns verbatim text in data['text'], lines in "
                 "data['lines'], and a legibility flag dict "
                 "(low_confidence/sparse/edge_clipped) so the caller "
                 "can decide whether to trust the text. Use this when "
                 "you want verbatim text out of a known region; use "
                 "'read' instead when you need natural-language Q&A."),
    "cursor":   (CursorAgent,   "locate the mouse cursor in the current frame"),
    "target":   (TargetAgent,   "locate a target by description (no click)"),
    # Aliases (kept for backwards compat).
    "search":   (ClickAgent,    "alias of 'click'"),
}


# ───────────────── rule-based planner ─────────────────

# Tokens we strip from the start of an intent to detect chained verbs.
_LEADING_PREP = ("then ", "and ", ", ")


def _strip_leading_prep(s: str) -> str:
    out = s.strip()
    for prep in _LEADING_PREP:
        if out.lower().startswith(prep):
            out = out[len(prep):].strip()
            break
    return out


def _split_chain(intent: str) -> list[str]:
    """Split a chained intent like 'login and open reddit.com' into
    ['login', 'open reddit.com']. Splits on ``" and "`` / ``" then "``
    / ``";"`` / sentence-level ``","``."""
    # Normalise sentence-level commas to "and" so a phrasing like
    # "navigate to reddit.com, go to r/Qiskit and fetch titles" yields
    # three planable chunks instead of one. Strip a trailing comma off
    # any token (shlex keeps "reddit.com," as one token otherwise).
    normalised = re.sub(r",\s+", " and ", intent)
    try:
        tokens = shlex.split(normalised, posix=True)
    except ValueError:
        # Unbalanced quotes — fall back to a naive split.
        tokens = normalised.split()
    chunks: list[list[str]] = [[]]
    for tok in tokens:
        # Detach a trailing comma if shlex left one attached.
        bare = tok.rstrip(",")
        if bare.lower() in ("and", "then") or bare == ";" or bare == "":
            if chunks[-1]:
                chunks.append([])
        else:
            chunks[-1].append(bare)
    return [" ".join(c) for c in chunks if c]


_FETCH_VERBS = (
    r"(?:give\s+me|show\s+me|tell\s+me|fetch|list|get|find|read|"
    r"display|extract|summari[zs]e|"
    r"what(?:'s|s|\s+is|\s+are)?(?:\s+the)?)"
)
_FETCH_NOUNS = (
    r"(?:posts?\s+titles?|titles?\s+of\s+(?:the\s+)?posts?|"
    r"blog\s+posts?|hot\s+posts?|"
    r"titles?|posts?|content|headlines?|threads?)"
)
# Optional "top [N]" prefix in front of the noun.
_TOP_PREFIX = r"(?:top\s+(?:(?P<n>\d+)\s+)?)?"
_SUB_PATH = (
    r"(?:r/(?P<sub_first>[A-Za-z0-9_]+)|"
    r"(?:[a-z0-9-]+\.[a-z]{2,}/)?r/(?P<sub_second>[A-Za-z0-9_]+))"
)
# Strict form: verb + (top N) + noun + (in|of|...) + r/<sub>, all
# adjacent. Catches the canonical phrasing in one shot.
_SUBREDDIT_FETCH_RE = re.compile(
    _FETCH_VERBS + r"\s+(?:the\s+)?" + _TOP_PREFIX
    + r"(?P<noun>" + _FETCH_NOUNS + r")"
    + r"(?:\s+(?:of|on|in|from))?\s+" + _SUB_PATH,
    re.IGNORECASE,
)
# Standalone search patterns used by the lenient compound scan: each
# half (verb, noun, subreddit, optional top-N) can appear anywhere in
# the intent in any order.
_VERB_ANY_RE = re.compile(_FETCH_VERBS, re.IGNORECASE)
_NOUN_ANY_RE = re.compile(_FETCH_NOUNS, re.IGNORECASE)
_SUB_ANY_RE = re.compile(r"r/([A-Za-z0-9_]+)", re.IGNORECASE)
_TOP_N_RE = re.compile(r"\btop\s+(\d+)\b", re.IGNORECASE)


def _match_subreddit_fetch(text: str) -> dict | None:
    """Pull (subreddit, top_n, noun) out of an intent if it asks for
    a list/read of subreddit content.

    Two-pass:

      1. **Strict form** — verb, count, noun, ``r/<sub>`` all adjacent
         within one phrase. Highest precision.
      2. **Compound form** — fall back to looking for each piece
         independently anywhere in the intent. Catches phrasings like
         *"navigate to reddit.com, go to r/Qiskit and fetch the top 5
         post titles"* where the subreddit and the fetch-shape live in
         different clauses.

    Used both inside ``_plan_one`` (single-phrase intents) and at the
    top of ``plan_intent`` to short-circuit chain-splitting.
    """
    m = _SUBREDDIT_FETCH_RE.search(text)
    if m:
        sub = m.group("sub_first") or m.group("sub_second") or ""
        if sub:
            raw_n = m.group("n")
            return {
                "sub": sub,
                "top_n": int(raw_n) if raw_n else 5,
                "noun": (m.group("noun") or "posts").lower(),
            }

    sub_m = _SUB_ANY_RE.search(text)
    if not sub_m:
        return None
    verb_m = _VERB_ANY_RE.search(text)
    noun_m = _NOUN_ANY_RE.search(text)
    if not (verb_m and noun_m):
        return None
    n_m = _TOP_N_RE.search(text)
    return {
        "sub": sub_m.group(1),
        "top_n": int(n_m.group(1)) if n_m else 5,
        "noun": noun_m.group(0).lower(),
    }


# NOTE: the planner is now LLM-first. The rule layer below handles
# only a small whitelist of trivial / security-sensitive intents
# that we want to dispatch in microseconds without involving the
# model:
#
#   - login / log in            (security-sensitive)
#   - focus / center / maximize (trivial primitive)
#   - wake                      (trivial primitive)
#   - scroll up|down [N]        (trivial primitive)
#   - subreddit-fetch shortcut  (a workflow with a fixed shape, not
#                                a phrasing variant — short-circuited
#                                at the top of plan_intent)
#
# Everything else (open/launch/close/save/copy/click/type/run/read/
# ocr/navigate/etc.) is routed to the LLM planner with few-shot
# examples. Repeated identical intents hit an in-memory cache so a
# second invocation skips the LLM round-trip.


def _subreddit_fetch_plan(
    *, subreddit: str, top_n: int, noun: str, platform: str,
) -> list[PlanStep]:
    """Build the canonical [dismiss, navigate, read] plan for a
    'fetch top N <noun> in r/<sub>' intent."""
    noun_norm = (
        "post titles" if "title" in noun
        else "posts" if "post" in noun
        else "headlines" if "headline" in noun
        else "threads" if "thread" in noun
        else "posts"
    )
    question = (
        f"List the top {top_n} {noun_norm} visible on this "
        f"r/{subreddit} page. Output ONE per line, numbered 1. 2. 3. "
        "etc., in the order they appear from top to bottom. "
        "Quote the exact text. Skip ads, sidebar widgets, "
        "navigation, and the subreddit description — only the "
        "main feed entries."
    )
    return [
        PlanStep(
            "dismiss", DismissModalsAgent,
            {"aggressive": True}, best_effort=True,
        ),
        PlanStep(
            "navigate", NavigateAgent,
            {"url": f"reddit.com/r/{subreddit}", "platform": platform},
        ),
        PlanStep(
            "read", ReadAgent,
            {
                "question": question,
                # Scroll the page until we've collected this many
                # titles (or run out of scroll budget). Reddit's
                # default layout shows ~2 posts above the fold, so
                # we need scrolls to satisfy "top 5".
                "scroll_collect": top_n,
                "scroll_amount": 5,
                "max_scrolls": 6,
            },
        ),
    ]


def _plan_one(
    intent: str,
    *,
    no_focus: bool = False,
    vault_name: str | None = None,
    platform: str = "linux",
) -> list[PlanStep]:
    """Tiny rule whitelist. Anything else → returns ``[]`` so the
    controller falls through to the LLM planner.

    Kept here on purpose:

      - ``login`` / ``log in`` (security-sensitive — never let the
        LLM rewrite this into a click on a fake login button).
      - ``focus`` / ``center`` / ``maximize`` (trivial primitive).
      - ``wake`` (trivial primitive).
      - ``scroll up|down [N]`` (trivial primitive — predictable
        param parsing makes the regex worth it).

    Subreddit-fetch is *not* matched here — it short-circuits at
    the top of :func:`plan_intent` because it's a multi-step
    workflow with a fixed shape, not a phrasing variant.
    """
    s = intent.strip()
    sl = s.lower()

    if sl == "login" or sl.startswith("log in"):
        return [PlanStep(
            "login", LoginAgent,
            {"vault_name": vault_name} if vault_name else {},
        )]

    if sl in ("focus", "center", "centre", "maximize", "maximise"):
        return [PlanStep("focus", FocusAgent, {"platform": platform})]

    if sl == "wake":
        return [PlanStep("wake", WakeAgent, {})]

    scroll_match = re.match(
        r"^scroll(?:\s+(up|down))?(?:\s+(\d+))?$", sl, re.IGNORECASE,
    )
    if scroll_match:
        direction = scroll_match.group(1) or "down"
        amount = int(scroll_match.group(2)) if scroll_match.group(2) else 4
        return [PlanStep(
            "scroll", ScrollAgent,
            {"direction": direction, "amount": amount},
        )]

    return []


def plan_intent(
    intent: str,
    *,
    no_focus: bool = False,
    vault_name: str | None = None,
    platform: str = "linux",
) -> list[PlanStep]:
    """Build a plan for ``intent``. Splits chained intents on
    ``" and "`` / ``" then "`` / sentence-level ``","``.

    A "subreddit fetch" intent (something containing ``r/<sub>`` AND a
    fetch verb/noun like ``top posts`` / ``post titles``) short-circuits
    the chain split: the whole phrase becomes a single
    ``[dismiss, navigate, read]`` plan. This handles compound phrasings
    like *"navigate to reddit.com, go to r/Qiskit and fetch the top 5
    post titles"* in one shot.
    """
    sub_q = _match_subreddit_fetch(intent)
    if sub_q:
        return _subreddit_fetch_plan(
            subreddit=sub_q["sub"],
            top_n=sub_q["top_n"],
            noun=sub_q["noun"],
            platform=platform,
        )

    plan, unresolved = _partial_plan(
        intent, no_focus=no_focus, vault_name=vault_name,
        platform=platform,
    )
    if unresolved:
        # Strict-mode behaviour preserved for tests: any chunk that
        # didn't rule-match means "no plan from rules alone". The
        # controller calls :func:`plan_intent_partial` instead so it
        # can ask the LLM to fill in just the unresolved bits.
        return []
    return plan


def plan_intent_partial(
    intent: str,
    *,
    no_focus: bool = False,
    vault_name: str | None = None,
    platform: str = "linux",
) -> tuple[list[PlanStep], list[tuple[int, str]]]:
    """Like :func:`plan_intent` but returns whatever the rule
    planner CAN handle, plus the chunks it couldn't.

    Returns ``(plan, unresolved)`` where ``unresolved`` is a list of
    ``(insertion_index, chunk_text)`` tuples — the controller asks
    the LLM to plan each chunk and splices the result back into
    ``plan`` at ``insertion_index``. Order across both lists is
    preserved so the final stitched plan keeps the user's original
    sequence.

    The subreddit-fetch short-circuit still applies — that case
    always plans whole-intent and never produces unresolved chunks.
    """
    sub_q = _match_subreddit_fetch(intent)
    if sub_q:
        return _subreddit_fetch_plan(
            subreddit=sub_q["sub"],
            top_n=sub_q["top_n"],
            noun=sub_q["noun"],
            platform=platform,
        ), []
    return _partial_plan(
        intent, no_focus=no_focus, vault_name=vault_name,
        platform=platform,
    )


def _partial_plan(
    intent: str,
    *,
    no_focus: bool,
    vault_name: str | None,
    platform: str,
) -> tuple[list[PlanStep], list[tuple[int, str]]]:
    """Shared rule-loop used by :func:`plan_intent` and
    :func:`plan_intent_partial`. Walks each chunk; chunks that
    don't match a rule are recorded as unresolved with the index
    they should be inserted at in the final plan."""
    parts = _split_chain(intent)
    plan: list[PlanStep] = []
    unresolved: list[tuple[int, str]] = []
    seen_names: list[str] = []
    for part in parts:
        text = _strip_leading_prep(part)
        steps = _plan_one(
            text,
            no_focus=no_focus,
            vault_name=vault_name,
            platform=platform,
        )
        if not steps:
            unresolved.append((len(plan), text))
            continue
        for s in steps:
            # Dedup adjacent identical Focus steps (login already
            # wakes/focuses for us; an explicit focus right before a
            # navigate after a login is wasteful).
            if (
                seen_names
                and seen_names[-1] == s.name == "focus"
            ):
                continue
            plan.append(s)
            seen_names.append(s.name)
    return plan, unresolved


# ───────────────── controller agent ─────────────────


class ControllerAgent(Agent):
    """Top-level orchestrator. Plans + executes."""

    name = "controller"

    async def run(
        self,
        *,
        intent: str,
        no_focus: bool = False,
        vault_name: str | None = None,
        platform: str = "linux",
        dry_run: bool = False,
        max_steps: int = MAX_STEPS,
        allow_llm_fallback: bool = True,
        final_settle_sec: float = 2.0,
        verify_completion: bool = True,
    ) -> ControllerOutcome:
        memory = load_memory()
        if memory:
            mem_path = _memory_path()
            print(f"Controller memory ({mem_path}):")
            for ln in memory.splitlines()[:30]:
                print(f"  | {ln}")
            extra = max(0, len(memory.splitlines()) - 30)
            if extra:
                print(f"  | ... ({extra} more line(s))")

        ck = _cache_key(intent, no_focus, vault_name, platform)
        cached = _cache_get(ck)
        if cached is not None:
            print(f"Plan (cache hit) — skipping rules + LLM")
            plan = list(cached)
            unresolved: list[tuple[int, str]] = []
            plan_source = "cache"
        else:
            plan, unresolved = plan_intent_partial(
                intent,
                no_focus=no_focus,
                vault_name=vault_name,
                platform=platform,
            )
            plan_source = "rules"
        if unresolved and allow_llm_fallback:
            # Some chunks didn't rule-match. Ask the LLM to plan
            # ONLY those chunks (with surrounding context), and
            # splice each result into the rule plan at the right
            # spot. This keeps the deterministic rule plan for
            # what we can match while still handling natural
            # phrasings for the rest.
            print(
                f"Rule-planned {len(plan)} step(s); "
                f"asking LLM for {len(unresolved)} unresolved "
                f"chunk(s): {[c for _, c in unresolved]!r}"
            )
            plan = await self._fill_unresolved(
                intent=intent, plan=plan, unresolved=unresolved,
                no_focus=no_focus, platform=platform,
                vault_name=vault_name, memory=memory,
            )
            plan_source = "rules+llm" if plan else "llm"
        elif unresolved and not allow_llm_fallback:
            # Strict-rules-only and we couldn't fully match — treat
            # as no plan so the existing "no rule matched" failure
            # path runs.
            plan = []
        if not plan and allow_llm_fallback:
            print(
                f"No rule matched {intent!r}; asking LLM planner..."
            )
            plan = await self._llm_plan(
                intent,
                no_focus=no_focus,
                platform=platform,
                vault_name=vault_name,
                memory=memory,
            )
            plan_source = "llm"
        if not plan:
            return ControllerOutcome(
                success=False,
                reason=(
                    f"no rule matched intent {intent!r}"
                    + ("" if allow_llm_fallback
                       else "; LLM fallback disabled")
                ),
                data={"intent": intent},
            )
        if len(plan) > max_steps:
            return ControllerOutcome(
                success=False,
                reason=f"plan too long ({len(plan)} > {max_steps})",
                data={"plan": [s.name for s in plan]},
            )

        # Cache the freshly-built plan so a repeat of the same intent
        # skips the LLM next time. Only on cache misses (would be a
        # no-op overwrite otherwise).
        if plan_source != "cache":
            _cache_put(ck, list(plan))

        print(f"Plan ({plan_source}):")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step.name} {step.kwargs or ''}")
        if dry_run:
            return ControllerOutcome(
                success=True,
                reason="dry-run; nothing executed",
                data={"plan": [s.name for s in plan]},
            )

        results: list[tuple[str, Outcome]] = []
        for i, step in enumerate(plan, 1):
            tag = " (best-effort)" if step.best_effort else ""
            print(f"\n[{i}/{len(plan)}] {step.name}{tag} ...")
            agent = step.agent_cls(self.ctx)
            try:
                outcome = await agent.run(**step.kwargs)
            except Exception as e:
                logger.exception("Agent %s raised", step.name)
                outcome = Outcome(
                    success=False, reason=f"exception: {e}",
                )
            results.append((step.name, outcome))
            mark = "✓" if outcome else "✗"
            print(f"   {mark} {step.name}: {outcome.reason}")
            if not outcome:
                if step.best_effort:
                    # Soft-fail: log + continue. The next step has its
                    # own pre-flight (e.g. NavigateAgent's browser
                    # check, ClickAgent's scroll-and-retry) that
                    # handles the same concern.
                    print(
                        f"   ↺ {step.name} is best-effort; "
                        "continuing despite failure"
                    )
                    continue
                completion = await self._final_capture_and_verify(
                    intent=intent,
                    final_settle_sec=final_settle_sec,
                    verify_completion=verify_completion,
                )
                return ControllerOutcome(
                    success=False,
                    reason=f"stopped at step {i} ({step.name})",
                    data={
                        "plan": [s.name for s in plan],
                        "results": [
                            (name, o.success, o.reason)
                            for name, o in results
                        ],
                        "completion": completion,
                    },
                )
        # Surface the answer text from any ReadAgent / OcrAgent step
        # on the final outcome so callers (CLI, Command Center) can
        # show it directly without having to scrape stdout. Last
        # such step wins — usually that's the most recent intent
        # action.
        answer = ""
        for name, o in results:
            if not (o.success and o.data):
                continue
            if name == "read":
                ans = str(o.data.get("answer", "")).strip()
                if ans:
                    answer = ans
            elif name == "ocr":
                # OcrAgent returns just the extracted text + legibility
                # signals; it never decides which line "answers" the
                # caller. Surface the full text and let downstream
                # consumers do their own filtering if needed.
                txt = str(o.data.get("text", "")).strip()
                if txt:
                    answer = txt
        completion = await self._final_capture_and_verify(
            intent=intent,
            final_settle_sec=final_settle_sec,
            verify_completion=verify_completion,
        )
        # Refine the top-line outcome with the verifier's verdict so
        # the cc UI / CLI summary line tells the user whether the
        # intent actually appears to have landed on screen.
        success = True
        reason = f"completed all {len(plan)} steps"
        if completion.get("verified") is False:
            success = False
            reason = (
                f"completed all {len(plan)} steps but visual "
                f"verification rejected the result: "
                f"{completion.get('reason', '')}"
            )
        elif completion.get("verified") is True:
            reason = (
                f"completed all {len(plan)} steps; visual "
                f"verification: {completion.get('reason', '')}"
            )
        return ControllerOutcome(
            success=success,
            reason=reason,
            data={
                "plan": [s.name for s in plan],
                "results": [
                    (name, o.success, o.reason) for name, o in results
                ],
                "answer": answer,
                "completion": completion,
            },
        )

    # ──────────────── final capture + completion verify ────────────────

    async def _final_capture_and_verify(
        self,
        *,
        intent: str,
        final_settle_sec: float,
        verify_completion: bool,
    ) -> dict[str, Any]:
        """Wait for the screen to settle, capture a ``final_state``
        frame, and (optionally) ask :class:`VerifyAgent` whether the
        intent looks like it landed.

        Always best-effort — never raises. Always records a frame
        when a capture device is wired so the cc UI's last
        screenshot reflects the actual end state, not whatever the
        last agent happened to capture mid-step.
        """
        import asyncio as _aio
        info: dict[str, Any] = {
            "captured": False,
            "verified": None,
            "reason": "",
        }
        if final_settle_sec > 0:
            try:
                await _aio.sleep(final_settle_sec)
            except Exception:
                pass
        if self.ctx.capture is None:
            info["reason"] = "no capture in context"
            return info
        try:
            frame = await self.ctx.capture.capture_frame()
            self.ctx.record_frame(frame.image, label="final_state")
            info["captured"] = True
        except Exception as e:
            logger.debug("final-state capture failed: %s", e)
            info["reason"] = f"final capture failed: {e}"
            return info

        if not verify_completion:
            info["reason"] = "verify disabled"
            return info
        if self.ctx.vision_client is None:
            info["reason"] = "no vision client; skipped verify"
            return info

        # Pre-OCR the frame so the verifier is given the literal
        # screen text rather than relying on the multimodal model
        # to read it. Vision models on small/cheap tiers (Nemotron-
        # Nano, Gemini Flash Lite) routinely pattern-match at a
        # high level and miss obvious error text like "Command
        # 'ind' not found" — the OCR pass forces the verdict to
        # be based on what's actually on screen.
        ocr_text = ""
        try:
            ocr_outcome = await OcrAgent(self.ctx).run(
                region="full", image=frame.image,
                record_label="final_ocr",
            )
            if ocr_outcome.success and ocr_outcome.data:
                ocr_text = (ocr_outcome.data.get("text") or "").strip()
        except Exception as e:
            logger.debug("final OCR failed: %s", e)

        # Hard heuristic: explicit error markers in the OCR text
        # short-circuit to FAILURE — but ONLY for intents that
        # imply the action should produce visible output (run /
        # find / fetch / navigate / etc.). For non-output intents
        # (close / open / switch / minimise), errors on screen are
        # almost always residue from a PRIOR action and don't
        # reflect on the current request — let the LLM judge based
        # on whether the requested state change actually happened.
        err_hit = _scan_for_error(ocr_text)
        if err_hit and _intent_expects_output(intent):
            info["verified"] = False
            info["reason"] = (
                f"visible error on screen: {err_hit!r}"
            )
            print(f"   final-state verify ✗ (error detected): {err_hit!r}")
            return info

        ocr_block = (
            f"\n\nThe OCR-extracted text on screen is verbatim:\n"
            f"-----\n{ocr_text[:1500]}\n-----\n\n"
            "Use the OCR text as the AUTHORITATIVE evidence of what is "
            "actually visible. Do not hallucinate output that isn't in "
            "the OCR text."
        ) if ocr_text else ""

        question = (
            "Look at the screen. The user asked the system to do "
            f"the following:\n\n    {intent}\n"
            + ocr_block
            + "\n\nDoes the screen now show evidence that this was "
            "SUCCESSFULLY accomplished?\n\n"
            "Strict rules — answer FALSE if ANY of these apply:\n"
            "  * The OCR text contains an error message — 'command "
            "not found', 'permission denied', 'no such file', "
            "'connection refused', 'did you mean', 'similar "
            "commands', '404', '500', stack-trace lines, etc.\n"
            "  * For a typed shell command, the OCR output below the "
            "command shows an error or a 'did you mean?' suggestion "
            "— that means the command was mistyped or doesn't exist, "
            "NOT that it succeeded.\n"
            "  * The action looks merely STARTED but not COMPLETED — "
            "the command's expected OUTPUT (file listing, search "
            "results, page content) isn't visible.\n"
            "  * The window/page/state didn't actually change as "
            "implied (asked to 'close X' but X is still visible; "
            "asked to 'open Y' but Y isn't foregrounded).\n\n"
            "Only answer TRUE when the visible end state reflects "
            "the requested OUTCOME, not just that a step was "
            "attempted. Quote the specific OCR snippet (or the "
            "visible error text) that justifies your verdict in "
            "your reason."
        )
        try:
            v = await VerifyAgent(self.ctx).run(
                question=question, visual_only=True,
                image=frame.image,
                record_label="final_verify",
            )
        except Exception as e:
            logger.debug("final verify failed: %s", e)
            info["reason"] = f"final verify errored: {e}"
            return info
        info["verified"] = bool(v)
        info["reason"] = v.reason
        mark = "✓" if v else "✗"
        print(f"   final-state verify {mark}: {v.reason}")
        return info

    # ───────────────────── per-chunk LLM filler ─────────────────────

    async def _fill_unresolved(
        self,
        *,
        intent: str,
        plan: list[PlanStep],
        unresolved: list[tuple[int, str]],
        no_focus: bool,
        platform: str,
        vault_name: str | None,
        memory: str,
    ) -> list[PlanStep]:
        """Resolve each unmatched chunk via the LLM and splice into
        ``plan`` at the index it was recorded with.

        We pass the LLM the surrounding context — the full original
        intent + the rule-matched plan so far — so it produces steps
        that compose with what's already there. If the LLM can't
        resolve any chunk, we fall back to whole-intent planning so
        the controller still has SOMETHING to try.
        """
        # Walk in source order. ``offset`` tracks cumulative
        # insertions so later splices land after earlier ones.
        # ``merged`` is the running plan; we pass it (NOT the
        # original rule plan) into each subsequent LLM call so
        # the LLM knows what's already been planned by previous
        # chunks. Without this, the per-chunk LLM cheerfully
        # re-emits a ``launch terminal`` for "check for updates"
        # because it never saw the prior chunk's ``launch
        # terminal`` for "open a terminal".
        merged = list(plan)
        offset = 0
        any_resolved = False
        for idx, chunk in unresolved:
            steps = await self._llm_plan_chunk(
                chunk=chunk,
                full_intent=intent,
                rule_plan=list(merged),
                no_focus=no_focus,
                platform=platform,
                vault_name=vault_name,
                memory=memory,
            )
            if steps:
                pos = idx + offset
                merged[pos:pos] = steps
                offset += len(steps)
                any_resolved = True
                print(
                    f"  LLM resolved chunk {chunk!r} → "
                    f"{[s.name for s in steps]!r}"
                )
            else:
                print(f"  LLM could not resolve chunk {chunk!r}")
        if any_resolved:
            # Belt-and-braces dedup: even with the cumulative-plan
            # context, models occasionally still emit duplicates.
            # An adjacent-pair dedup keyed on (name, kwargs) fixes
            # most of those without hiding genuine repetition (a
            # plan that legitimately wants two consecutive ``keys``
            # of the SAME chord — rare — would also collapse, but
            # collapsing 2 → 1 of an idempotent step is harmless).
            return _dedup_adjacent_steps(merged)
        # Total miss — let the caller fall through to whole-intent
        # planning by returning an empty plan.
        return []

    async def _llm_plan_chunk(
        self,
        *,
        chunk: str,
        full_intent: str,
        rule_plan: list[PlanStep],
        no_focus: bool,
        platform: str,
        vault_name: str | None,
        memory: str,
    ) -> list[PlanStep]:
        """Plan a single sub-intent. Same validation as the full
        :meth:`_llm_plan`; the only difference is the prompt makes
        clear we want steps that EXTEND the existing plan."""
        if self.ctx.vision_client is None:
            return []
        agent_descriptions = "\n".join(
            f"  - {name}: {desc}" for name, (_, desc) in REGISTRY.items()
        )
        already = ", ".join(
            f"{s.name}({s.kwargs})" for s in rule_plan
        ) or "(empty)"
        memory_block = ""
        if memory:
            memory_block = (
                "Long-lived notes from the user (authoritative):\n"
                f"{memory}\n\n"
            )
        prompt = (
            "You are a JSON planner. Plan the additional agent "
            "steps needed for ONE sub-intent of a larger task.\n\n"
            f"{memory_block}"
            f"Full original intent:\n    {full_intent}\n\n"
            f"Steps already planned (from rule matching):\n    {already}\n\n"
            f"Sub-intent to plan:\n    {chunk}\n\n"
            "Available agents:\n"
            f"{agent_descriptions}\n\n"
            "Rules:\n"
            "  * Reply with EXACTLY a JSON object whose 'plan' is the "
            "ordered steps to ADD AFTER the already-planned ones.\n"
            "  * Use ONLY the listed agents.\n"
            "  * Plan must be 1–5 steps.\n"
            "  * If the sub-intent involves typing into a terminal, "
            "use the 'type' agent with submit=true.\n"
            "  * Prefer 'keys' over 'click' when a keyboard shortcut "
            "exists.\n"
            "  * NO preamble, NO markdown, NO commentary.\n\n"
            'Schema: {"plan": [{"name": "<agent>", "kwargs": {...}}, ...]}'
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Plan the steps to accomplish ONLY: {chunk!r}. "
                "Reply JSON only."
            )},
        ]
        return await self._llm_plan_call(messages)

    # ───────────────────── LLM-planner fallback ─────────────────────

    async def _llm_plan(
        self,
        intent: str,
        *,
        no_focus: bool,
        platform: str,
        vault_name: str | None,
        memory: str = "",
    ) -> list[PlanStep]:
        """Ask the multimodal model to produce a plan.

        Validates every step against :data:`REGISTRY`; rejects plans
        that reference unknown actions, exceed the step cap, or
        contain malformed kwargs.
        """
        if self.ctx.vision_client is None:
            return []
        agent_descriptions = "\n".join(
            f"  - {name}: {desc}" for name, (_, desc) in REGISTRY.items()
        )
        memory_block = ""
        if memory:
            memory_block = (
                "Long-lived notes from the user (treat these as "
                "authoritative — they reflect the target machine's "
                "real configuration and the user's preferences):\n"
                f"{memory}\n\n"
            )
        prompt = (
            "You are a JSON planner. The user wants to accomplish an "
            "intent on a remote computer that we control via mouse + "
            "keyboard. Decompose the intent into a sequence of agent "
            "calls.\n\n"
            f"{memory_block}"
            f"User intent:\n    {intent}\n\n"
            "Available agents:\n"
            f"{agent_descriptions}\n\n"
            "Hard rules:\n"
            "  * Use ONLY the agents listed above; never invent names.\n"
            "  * Plan length must be between 1 and "
            f"{MAX_STEPS} steps.\n"
            "  * Each step has a 'name' (one of the agents above) and "
            "'kwargs' (a JSON object of arguments).\n"
            "  * For 'click', 'navigate', 'login', kwargs are typed:\n"
            "      click  -> {\"target\": \"<text description>\"}\n"
            "      navigate -> {\"url\": \"<url>\", \"platform\": "
            f"\"{platform}\"}}\n"
            "      login  -> "
            f"{{\"vault_name\": \"{vault_name or '<entry>'}\"}}"
            " (omit if no vault entry available)\n"
            "      type   -> {\"text\": \"...\", \"submit\": true|false}"
            " — set submit=true to press Enter after typing; do NOT "
            "embed '\\n' in text.\n"
            "      focus  -> "
            f"{{\"platform\": \"{platform}\"}}\n"
            "      keys   -> {\"modifiers\": [<list>], \"key\": "
            "\"<key>\"} — single keyboard shortcut.\n"
            "  * Prefix UI-affecting steps with a 'focus' step "
            f"unless --no-focus was set ({not no_focus} here).\n"
            "  * STRONGLY PREFER 'keys' over 'click' for actions "
            "that have a known keyboard shortcut — clicking icons "
            "is unreliable. Examples:\n"
            "      close window   → {\"modifiers\":[\"alt\"],\"key\":\"F4\"}\n"
            "      close tab      → {\"modifiers\":[\"ctrl\"],\"key\":\"w\"}\n"
            "      new tab        → {\"modifiers\":[\"ctrl\"],\"key\":\"t\"}\n"
            "      save           → {\"modifiers\":[\"ctrl\"],\"key\":\"s\"}\n"
            "      copy / paste / cut → ctrl + c/v/x\n"
            "      undo / redo    → ctrl+z / ctrl+shift+z\n"
            "      quit app       → ctrl+q\n"
            "      switch window  → alt+Tab\n"
            "      maximise/min   → super+Up / super+h\n"
            "      press Enter    → {\"modifiers\":[],\"key\":\"Enter\"}\n\n"
            "Respond with ONLY a JSON object — no preamble, no "
            "markdown.\n\n"
            'Schema: {"plan": ['
            '{"name": "<agent>", "kwargs": {...}}, ...]}\n\n'
            + _PLANNER_FEW_SHOT
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Plan the steps for: {intent!r}. Reply JSON only."
            )},
        ]
        return await self._llm_plan_call(messages)

    async def _llm_plan_call(
        self, messages: list[dict],
    ) -> list[PlanStep]:
        """Shared call+validation path for the whole-intent and
        per-chunk LLM planners.

        Three-pass attempt sequence — each only runs if the prior
        produced an empty / unparsable response:

          1. JSON-mode response_format + ``vision_model``.
          2. Free-form ``vision_model``.
          3. Free-form on the ``ocr_model`` if available — different
             model, different failure modes; sometimes the small
             OCR model outperforms the larger general model on
             strict format compliance.
        """
        attempts: list[dict[str, Any]] = []
        attempts.append({
            "model": self.ctx.vision_model,
            "json_mode": True, "label": "vision/json",
        })
        attempts.append({
            "model": self.ctx.vision_model,
            "json_mode": False, "label": "vision/freeform",
        })
        if self.ctx.ocr_model and self.ctx.ocr_model != self.ctx.vision_model:
            attempts.append({
                "model": self.ctx.ocr_model,
                "json_mode": False, "label": "ocr-model/freeform",
            })

        raw_last = ""
        for cfg in attempts:
            try:
                call_kwargs: dict[str, Any] = dict(
                    model=cfg["model"],
                    max_tokens=1200,
                    temperature=0.0,
                    messages=messages,
                )
                if cfg["json_mode"]:
                    call_kwargs["response_format"] = {"type": "json_object"}
                resp = await self.ctx.vision_client.chat.completions.create(
                    **call_kwargs
                )
            except Exception as e:
                logger.debug(
                    "LLM-planner attempt %s failed: %s",
                    cfg["label"], e,
                )
                continue
            try:
                raw = resp.choices[0].message.content or ""
            except Exception:
                continue
            raw_last = raw
            plan_dict = self._extract_json(raw) or {}
            steps_raw = plan_dict.get("plan")
            if isinstance(steps_raw, list) and steps_raw:
                validated = self._validate_steps(steps_raw)
                if validated:
                    print(
                        f"LLM planner ({cfg['label']}) produced "
                        f"{len(validated)} step(s)"
                    )
                    return validated
            logger.debug(
                "LLM-planner attempt %s produced no usable plan "
                "(raw=%s)", cfg["label"], raw[:160],
            )
        if raw_last:
            logger.warning(
                "LLM planner returned no plan after all attempts "
                "(last raw=%s)", raw_last[:200],
            )
        else:
            logger.warning("LLM planner returned no plan (raw=)")
        return []

    def _validate_steps(self, steps_raw: list) -> list[PlanStep]:
        """Validate a list of step dicts from the LLM. Returns
        ``[]`` if any entry is invalid."""
        validated: list[PlanStep] = []
        for entry in steps_raw[:MAX_STEPS]:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip().lower()
            kwargs = entry.get("kwargs", {})
            if name not in REGISTRY:
                logger.warning(
                    "LLM planner referenced unknown agent %r — "
                    "rejecting plan", name,
                )
                return []
            if not isinstance(kwargs, dict):
                logger.warning(
                    "LLM planner kwargs for %s is not a dict — "
                    "rejecting plan", name,
                )
                return []
            agent_cls = REGISTRY[name][0]
            # ``focus`` and ``dismiss`` are precondition steps that
            # downstream agents already pre-flight on their own.
            # A failed precondition should NOT kill the rest of
            # the plan.
            best_effort = name in {"focus", "dismiss"}
            # Auto-correct ``{"text": "...\n", "submit": false}``
            # (LLMs frequently embed a newline instead of using
            # submit=true).
            if name == "type" and isinstance(kwargs, dict):
                txt = kwargs.get("text", "")
                if isinstance(txt, str) and txt.endswith("\n"):
                    kwargs = {**kwargs, "text": txt.rstrip("\n"),
                              "submit": True}
            # Drop kwargs the agent's run() doesn't accept. The
            # LLM frequently invents arguments (e.g. focus
            # {"app": "Terminal"} — FocusAgent has no `app` param)
            # which would raise TypeError at dispatch time. Filtering
            # is generic — works for any agent without per-name
            # special-casing.
            kwargs = _filter_kwargs(agent_cls, kwargs, name=name)
            validated.append(
                PlanStep(name, agent_cls, kwargs, best_effort=best_effort)
            )
        return validated

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        if not raw:
            return None
        import json
        # Try direct parse first (model in JSON-mode).
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Pull the first {...} substring.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
