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
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.click import ClickAgent
from terminaleyes.agents.cursor import CursorAgent
from terminaleyes.agents.focus import FocusAgent
from terminaleyes.agents.login import LoginAgent
from terminaleyes.agents.navigate import NavigateAgent
from terminaleyes.agents.scroll import ScrollAgent
from terminaleyes.agents.target import TargetAgent
from terminaleyes.agents.type_text import TypeAgent
from terminaleyes.agents.verify import VerifyAgent
from terminaleyes.agents.wake import WakeAgent

logger = logging.getLogger(__name__)


# Hard step cap so a runaway plan can't lock up the target.
MAX_STEPS = 12


@dataclass
class PlanStep:
    name: str
    agent_cls: type
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerOutcome(Outcome):
    pass


# ───────────────── agent registry ─────────────────

REGISTRY: dict[str, tuple[type, str]] = {
    "wake":     (WakeAgent,     "wake the remote screen / dismiss screensaver"),
    "verify":   (VerifyAgent,   "ask a yes/no visual question about the screen"),
    "focus":    (FocusAgent,    "centre and maximise the foreground app"),
    "login":    (LoginAgent,    "wake + verify-login + type password from vault"),
    "type":     (TypeAgent,     "type text (optional secret + Enter)"),
    "navigate": (NavigateAgent, "type a URL into a browser address bar (browser-aware)"),
    "click":    (ClickAgent,    "find a target by description; scroll-and-retry if not visible"),
    "scroll":   (ScrollAgent,   "scroll up/down via the mouse wheel"),
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
    ['login', 'open reddit.com']. Conservative split — only on
    ``" and "`` / ``" then "`` / ``"; "`` outside of quotes."""
    # Tokenize via shlex so quoted segments stay intact.
    try:
        tokens = shlex.split(intent, posix=True)
    except ValueError:
        # Unbalanced quotes — fall back to a naive split.
        tokens = intent.split()
    chunks: list[list[str]] = [[]]
    for tok in tokens:
        if tok.lower() in ("and", "then") or tok == ";":
            if chunks[-1]:
                chunks.append([])
        else:
            chunks[-1].append(tok)
    return [" ".join(c) for c in chunks if c]


def _plan_one(
    intent: str,
    *,
    no_focus: bool = False,
    vault_name: str | None = None,
    platform: str = "linux",
) -> list[PlanStep]:
    """Map a single (atomic) intent to a list of plan steps."""
    s = intent.strip()
    sl = s.lower()

    # login
    if sl == "login" or sl.startswith("log in"):
        return [PlanStep(
            "login", LoginAgent,
            {"vault_name": vault_name} if vault_name else {},
        )]

    # focus / center / centre
    if sl in ("focus", "center", "centre", "maximize", "maximise"):
        return [PlanStep("focus", FocusAgent, {"platform": platform})]

    # navigate / go to / open <url>
    nav_match = re.match(
        r"^(?:navigate to|go to|open(?:\s+the)?(?:\s+page\s+at)?)\s+(.+)$",
        sl, re.IGNORECASE,
    )
    if nav_match:
        # Use the original casing for the URL.
        url = re.match(
            r"^(?:navigate to|go to|open(?:\s+the)?(?:\s+page\s+at)?)\s+(.+)$",
            s, re.IGNORECASE,
        ).group(1).strip()
        steps: list[PlanStep] = []
        if not no_focus:
            steps.append(PlanStep("focus", FocusAgent, {"platform": platform}))
        steps.append(PlanStep(
            "navigate", NavigateAgent,
            {"url": url, "platform": platform},
        ))
        return steps

    # click <target>
    click_match = re.match(r"^click\s+(.+)$", s, re.IGNORECASE)
    if click_match:
        target = click_match.group(1).strip()
        steps = []
        if not no_focus:
            steps.append(PlanStep("focus", FocusAgent, {"platform": platform}))
        steps.append(PlanStep(
            "click", ClickAgent, {"target": target},
        ))
        return steps

    # type <text>
    type_match = re.match(r"^type\s+(.+)$", s, re.IGNORECASE)
    if type_match:
        text = type_match.group(1).strip()
        # Strip surrounding quotes if present.
        if (text.startswith("'") and text.endswith("'")) or (
            text.startswith('"') and text.endswith('"')
        ):
            text = text[1:-1]
        return [PlanStep(
            "type", TypeAgent, {"text": text, "submit": False},
        )]

    # wake
    if sl == "wake":
        return [PlanStep("wake", WakeAgent, {})]

    # scroll <direction> [N]
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

    # No rule matched.
    return []


def plan_intent(
    intent: str,
    *,
    no_focus: bool = False,
    vault_name: str | None = None,
    platform: str = "linux",
) -> list[PlanStep]:
    """Build a plan for ``intent``. Splits chained intents on
    ``" and "`` / ``" then "``."""
    parts = _split_chain(intent)
    plan: list[PlanStep] = []
    seen_names: list[str] = []
    for part in parts:
        steps = _plan_one(
            _strip_leading_prep(part),
            no_focus=no_focus,
            vault_name=vault_name,
            platform=platform,
        )
        if not steps:
            return []  # signal: no rule matched
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
    return plan


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
    ) -> ControllerOutcome:
        plan = plan_intent(
            intent,
            no_focus=no_focus,
            vault_name=vault_name,
            platform=platform,
        )
        plan_source = "rules"
        if not plan and allow_llm_fallback:
            print(
                f"No rule matched {intent!r}; asking LLM planner..."
            )
            plan = await self._llm_plan(
                intent,
                no_focus=no_focus,
                platform=platform,
                vault_name=vault_name,
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
            print(f"\n[{i}/{len(plan)}] {step.name} ...")
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
                return ControllerOutcome(
                    success=False,
                    reason=f"stopped at step {i} ({step.name})",
                    data={
                        "plan": [s.name for s in plan],
                        "results": [
                            (name, o.success, o.reason)
                            for name, o in results
                        ],
                    },
                )
        return ControllerOutcome(
            success=True,
            reason=f"completed all {len(plan)} steps",
            data={
                "plan": [s.name for s in plan],
                "results": [
                    (name, o.success, o.reason) for name, o in results
                ],
            },
        )

    # ───────────────────── LLM-planner fallback ─────────────────────

    async def _llm_plan(
        self,
        intent: str,
        *,
        no_focus: bool,
        platform: str,
        vault_name: str | None,
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
        prompt = (
            "You are a JSON planner. The user wants to accomplish an "
            "intent on a remote computer that we control via mouse + "
            "keyboard. Decompose the intent into a sequence of agent "
            "calls.\n\n"
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
            "      type   -> {\"text\": \"...\", \"submit\": false}\n"
            "      focus  -> "
            f"{{\"platform\": \"{platform}\"}}\n"
            "  * Prefix UI-affecting steps with a 'focus' step "
            f"unless --no-focus was set ({not no_focus} here).\n\n"
            "Respond with ONLY a JSON object — no preamble, no "
            "markdown.\n\n"
            'Schema: {"plan": ['
            '{"name": "<agent>", "kwargs": {...}}, ...]}'
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Plan the steps for: {intent!r}. Reply JSON only."
            )},
        ]
        for attempt in range(2):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.ctx.vision_model,
                    max_tokens=900,
                    temperature=0.0,
                    messages=messages,
                )
                if attempt == 0:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await self.ctx.vision_client.chat.completions.create(
                    **kwargs
                )
                break
            except Exception as e:
                if attempt == 0:
                    logger.debug(
                        "LLM-planner json_object format failed (%s); "
                        "retrying free-form", e,
                    )
                    continue
                logger.warning("LLM-planner call failed: %s", e)
                return []
        # Extract JSON.
        raw = ""
        try:
            raw = resp.choices[0].message.content or ""
        except Exception:
            return []
        plan_dict = self._extract_json(raw) or {}
        steps_raw = plan_dict.get("plan")
        if not isinstance(steps_raw, list) or not steps_raw:
            logger.warning(
                "LLM planner returned no plan (raw=%s)", raw[:200],
            )
            return []
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
            validated.append(PlanStep(name, agent_cls, kwargs))
        if not validated:
            return []
        print(f"LLM planner produced {len(validated)} step(s)")
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
