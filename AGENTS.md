# AGENTS.md

terminaleyes is built around a tiered agent architecture. Each agent
is a small, testable unit with a typed `Outcome` return; higher-tier
agents compose lower-tier ones. The deep technical reference lives in
[CLAUDE.md](./CLAUDE.md); this file is the index.

## At a glance

```
agents/
├── base.py / context.py   — Agent ABC, Outcome, AgentContext
├── vault.py               — AES-256-GCM encrypted credential store
│
├── verify.py              — tier-1: visual yes/no oracle
├── cursor.py              — tier-1: locate cursor (HSV / variance / diff)
├── target.py              — tier-1: locate target by description
│
├── wake.py                — tier-2: wake screen sequence
├── type_text.py           — tier-2: text input (with secret mode)
│
├── focus.py               — tier-3: centre + maximise the foreground app
├── login.py               — tier-3: wake + verify-login + type secret
├── navigate.py            — tier-3: URL-bar typing
├── click.py               — tier-3: find-and-click (was SearchAgent)
│
└── controller.py          — top-level: rules + LLM-planner fallback
```

Each user-facing CLI subcommand maps to one agent or to the controller.

## Tier 1 — atomic primitives

| Agent | Job | Notes |
|-------|-----|-------|
| `VerifyAgent` | "Does the screen look like X?" — yes/no | Visual-only steering; refuses to hallucinate on dark/asleep screens. JSON-mode + free-form retry. |
| `CursorAgent` | Locate the cursor in the current frame | HSV (motion-verified) → oscillation-variance → ROI-prior diff. No model calls. |
| `TargetAgent` | Locate a target by description | Cascade: OCR (quoted-token primary) → scene-map+ShowUI grounding → ShowUI on cropped sidebar/footer. |

## Tier 2 — actions

| Agent | Job |
|-------|-----|
| `WakeAgent` | Mouse jiggle + Down arrow + click (idempotent). |
| `TypeAgent` | Send text via the keyboard backend. `secret=True` redacts from local logs; `submit=True` follows with Enter. |

## Tier 3 — workflows

| Agent | Job | Notes |
|-------|-----|-------|
| `FocusAgent` | Verify the foreground app is centred/maximised; if not, drive `Super+Up` (GNOME) or `Cmd+Ctrl+F` (macOS). | Awake pre-check refuses to act on dark/asleep screens. |
| `LoginAgent` | `Wake → poll-VerifyAgent("login screen?") → TypeAgent(secret) → Enter`. | Password from explicit arg, vault, file, env var, or `getpass`. Visual-only verification — no reliance on the literal word "password" being on screen. |
| `NavigateAgent` | URL-bar navigation: `Ctrl+L → Ctrl+A → type → Enter`. | Cross-platform (`Cmd+L` for macOS). |
| `ClickAgent` (alias `SearchAgent`) | Find a target by description and click it. | Wraps the existing `VisualServoHomer` pipeline (cursor detection → target localisation → visual servo → click retry → post-click oracle). |

## Tier 4 — storage

| `Vault` | AES-256-GCM at `~/.config/terminaleyes/vault.enc`, mode `0600`. Scrypt KDF (`N=2^15`, `r=8`, `p=1`). Master passphrase via `getpass` or `TERMINALEYES_VAULT_PASSPHRASE` env var. CLI: `vault add/get/list/remove/status`. |

## Top level — `ControllerAgent`

The orchestrator. Takes a free-form intent, plans a sequence of agent
calls, runs them with audit trail, fails safe.

**Two-phase planning:**

1. **Rule-based router** (default, fast, no LLM):
   - `login` → `[LoginAgent]`
   - `focus` / `center` → `[FocusAgent]`
   - `go to URL` / `open URL` → `[FocusAgent, NavigateAgent]`
   - `click X` → `[FocusAgent, ClickAgent(target=X)]`
   - `type X` → `[TypeAgent(text=X)]`
   - `A and B` / `A then B` → chain of plans
2. **LLM-planner fallback**: when no rule matches, prompt the
   multimodal model with the registry, expect a JSON plan, validate
   every step's agent name against the registry, reject malformed or
   unknown actions. Disable with `--no-llm-fallback`.

**Defaults:**
- Click-like steps are prefixed with `FocusAgent` unless `--no-focus`.
- Hard cap on total steps (`MAX_STEPS = 12`).
- Each step's `Outcome` is collected; final outcome surfaces the full
  audit trail.

**Usage:**

```bash
terminaleyes do "click the Run button"
terminaleyes do "go to reddit.com/r/LocalLLaMA"
terminaleyes do "login and open reddit.com" --vault myhost
terminaleyes do --dry-run "wake the screen and centre the browser"
terminaleyes do --no-focus "click X"        # skip the auto-focus prefix
terminaleyes do --no-llm-fallback "..."     # rules-only, refuses unknown intents
```

## Where to look first

- **Click behaviour** → `commander/visual_servo_homer.py` (the loop), surfaced via `agents/click.py`
- **Cursor detection knobs** → `agents/cursor.py` + `commander/cursor_finder.py`
- **OCR target locator** → `agents/target.py` + `commander/ocr_finder.py`
- **Login wake / verify polling** → `agents/login.py`
- **Vault crypto** → `agents/vault.py`
- **Pi BT/USB HID** → `raspi/server.py`, `raspi/bt_hid.py`, plus
  CLAUDE.md "Pi Zero 2 W — Critical lessons learned"
- **CLI subcommand wiring** → `cli.py`
- **Defaults (model name, base URLs)** → `config/settings.py`

## CLI surface

```bash
# Controller (high-level)
terminaleyes do "<intent>"                  # rules → LLM fallback

# Direct agent invocations (low-level)
terminaleyes login [--vault NAME] [--password-file F] [--password-env V]
terminaleyes focus [--platform linux|macos]
terminaleyes interact                       # legacy REPL, routes through homer

# Vault
terminaleyes vault add NAME
terminaleyes vault get NAME
terminaleyes vault list
terminaleyes vault remove NAME
terminaleyes vault status

# Other (Pi-side / agent loop / capture / etc.)
terminaleyes-pi                             # Pi REST endpoint
terminaleyes run --goal "..."               # older goal-driven loop
terminaleyes watch                          # passive screen observer
terminaleyes endpoint                       # local dev terminal display
terminaleyes capture-test                   # save one webcam frame
```

## Adding a new agent

1. Create `agents/<name>.py` subclassing `Agent` with an `async def run(...)`.
2. If it should be reachable via the controller, add it to `REGISTRY`
   in `agents/controller.py` with a one-line capability description.
3. Add a rule to `_plan_one` if there's a natural English shape; the
   LLM-planner fallback handles long-tail intents automatically.
4. Direct CLI subcommand wiring in `cli.py` if it's standalone-useful.
