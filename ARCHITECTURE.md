# terminaleyes Architecture

## System Overview

terminaleyes is a vision-based agentic terminal controller. It controls a terminal purely through visual feedback (webcam capture) and keystroke output. The control path is intentionally indirect -- vision in, keystrokes out -- to support future deployment on a Raspberry Pi acting as a USB HID keyboard plugged into a physical machine.

```
                    +---------------------+
                    |    Physical World    |
                    +---------------------+
                           |        ^
                    webcam |        | USB HID / HTTP keystrokes
                           v        |
+--------------------------------------------------------------------------+
|                        terminaleyes Agent                                 |
|                                                                          |
|  +------------------+    +------------------+    +-------------------+   |
|  | Vision Capture   |--->| MLLM Interpreter |--->| Agent / Decision  |   |
|  | (webcam/OpenCV)  |    | (Claude/GPT-4V)  |    | Engine            |   |
|  +------------------+    +------------------+    +-------------------+   |
|                                                         |                |
|                                                         v                |
|                                                  +-------------------+   |
|                                                  | Keyboard Output   |   |
|                                                  | (HTTP / USB HID)  |   |
|                                                  +-------------------+   |
+--------------------------------------------------------------------------+
                                                         |
                                                         v
                                              +--------------------+
                                              | Target Machine     |
                                              | (HTTP Endpoint or  |
                                              |  Physical Machine) |
                                              +--------------------+
```

## Component Architecture

### 1. Vision Capture (`capture/`)

**Responsibility:** Capture frames from a webcam showing the terminal display.

**Key Classes:**
- `CaptureSource` (ABC) -- Abstract interface for any visual source
- `WebcamCapture` -- Concrete implementation using OpenCV

**Design Decisions:**
- Async interface with `__aenter__`/`__aexit__` for resource management
- Blocking OpenCV calls run in thread pool executor via `asyncio.run_in_executor`
- Configurable crop region to focus on the terminal area of the webcam view
- Frame counter and timestamps for correlation with interpreter output

**Data Flow:** Camera device --> numpy array (BGR) --> `CapturedFrame` model

### 2. MLLM Interpreter (`interpreter/`)

**Responsibility:** Send terminal screenshots to a multimodal LLM and receive structured interpretations of the terminal state.

**Key Classes:**
- `MLLMProvider` (ABC) -- Provider-agnostic interface
- `AnthropicProvider` -- Claude API implementation
- `OpenAIProvider` -- GPT-4V API implementation

**Design Decisions:**
- Strategy pattern allows swapping providers without changing the agent
- Shared system prompt ensures consistent interpretation format across providers
- Shared `_parse_response` and `_encode_frame_to_base64` utilities in the base class
- Structured output via JSON schema in the system prompt
- `MLLMError` exception carries provider name and raw response for debugging

**Data Flow:** `CapturedFrame` --> base64 PNG --> MLLM API --> JSON --> `TerminalState`

### 3. Agent / Decision Engine (`agent/`)

**Responsibility:** Orchestrate the capture-interpret-decide-act loop and maintain context.

**Key Classes:**
- `AgentStrategy` (ABC) -- Pluggable decision-making logic
- `AgentLoop` -- Central orchestrator

**Design Decisions:**
- The loop is the only component that knows about all other components
- Strategies are pure decision functions (no side effects)
- Context accumulates full history for strategy decision-making
- Configurable error tolerance (max consecutive errors)
- Step limits prevent infinite loops
- Clean separation between orchestration (AgentLoop) and decision logic (AgentStrategy)

**Control Flow:**
```
AgentLoop.run(goal):
    while not done:
        frame = capture.capture_frame()
        state = interpreter.interpret(frame)
        context.add_observation(state)
        action, reasoning = strategy.decide_action(context, state)
        if action:
            keyboard.send(action)
            context.add_action(action, reasoning)
        status = strategy.evaluate_completion(context, state)
        if status in (COMPLETED, FAILED):
            break
```

### 4. Keyboard Output (`keyboard/`)

**Responsibility:** Translate logical keyboard actions into the appropriate protocol for the target.

**Key Classes:**
- `KeyboardOutput` (ABC) -- Abstract pluggable interface
- `HttpKeyboardOutput` -- HTTP backend for the local endpoint
- `UsbHidKeyboardOutput` -- Placeholder for future Raspberry Pi hardware

**Design Decisions:**
- Abstract interface is the core architectural boundary for hardware swapping
- Three action types: `Keystroke`, `KeyCombo`, `TextInput` (discriminated union)
- HTTP backend uses httpx async client
- `send_line()` convenience method in the base class
- Async context manager for clean connection lifecycle

**API Contract (HTTP Backend):**
```
POST /keystroke    {"key": "Enter"}
POST /key-combo    {"modifiers": ["ctrl"], "key": "c"}
POST /text         {"text": "ls -la"}
GET  /health       -> {"status": "ok", ...}
```

### 5. Local HTTP Endpoint (`endpoint/`)

**Responsibility:** Temporary software stand-in for a physical machine. Receives keystrokes, runs them in a persistent shell, and renders output in a visible terminal window.

**Key Classes:**
- `create_app()` -- FastAPI application factory
- `PersistentShell` -- Long-running shell subprocess manager
- `TerminalDisplay` -- pygame-based terminal window renderer

**Design Decisions:**
- FastAPI for async HTTP handling
- Application factory pattern for testability (inject mock shell/display)
- PersistentShell maintains a real shell process (not one-off commands)
- TerminalDisplay runs in its own thread (pygame requires main thread on some OSes, but we use a dedicated thread to keep the async server responsive)
- The display renders with monospace font on dark background to be easily readable by the webcam + MLLM
- Screen buffer maintains scrollback history

**Architecture Note:** This entire module is temporary. When deployed on physical hardware, the endpoint is replaced by a real machine, and `HttpKeyboardOutput` is replaced by `UsbHidKeyboardOutput`.

### 6. Configuration (`config/`)

**Responsibility:** Load and validate application settings.

**Key Classes:**
- `Settings` (pydantic-settings BaseSettings) -- Root configuration
- Section models: `CaptureConfig`, `MLLMConfig`, `EndpointConfig`, `KeyboardConfig`, `AgentConfig`, `LoggingConfig`

**Design Decisions:**
- YAML file for human-readable configuration
- Pydantic v2 for validation with descriptive error messages
- Environment variable overrides for secrets (API keys) -- never in YAML
- `TERMINALEYES_` prefix for all env vars, double underscore for nesting
- `load_settings()` function handles file loading with fallback to defaults

### 7. Domain Models (`domain/`)

**Responsibility:** Define all shared data structures.

**Key Models:**
- `CapturedFrame` -- Webcam frame with metadata
- `TerminalState` / `TerminalContent` -- Interpreted terminal state
- `Keystroke` / `KeyCombo` / `TextInput` -- Keyboard actions (discriminated union)
- `AgentGoal` / `AgentAction` / `AgentContext` -- Agent state

**Design Decisions:**
- Pydantic v2 BaseModel for validation and serialization
- Frozen models where immutability is appropriate
- Discriminated union for keyboard actions (`action_type` field)
- numpy arrays allowed in CapturedFrame via `arbitrary_types_allowed`

## Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Image capture | OpenCV (`opencv-python`) | Most portable, well-documented webcam support |
| Image processing | Pillow, numpy | Standard Python imaging stack |
| MLLM clients | `anthropic`, `openai` SDKs | Official, well-maintained async clients |
| HTTP server | FastAPI + uvicorn | Async-native, automatic validation, good DX |
| HTTP client | httpx | Async support, modern API, good error handling |
| Terminal display | pygame | Fine-grained rendering control, monospace fonts |
| Data models | Pydantic v2 | Validation, serialization, settings management |
| Configuration | YAML + pydantic-settings | Human-readable config with env var overrides |
| Async runtime | asyncio | Standard library, widely supported |
| Testing | pytest + pytest-asyncio | De facto standard, good async support |
| Type checking | mypy (strict) | Catches bugs early, documents interfaces |
| Linting | ruff | Fast, comprehensive, replaces multiple tools |

## Security Considerations

- API keys are loaded exclusively from environment variables, never stored in config files
- The HTTP endpoint binds to `0.0.0.0` by default -- consider restricting to `127.0.0.1` for local-only use
- The persistent shell runs with the same privileges as the endpoint process -- consider sandboxing
- No authentication on the HTTP endpoint currently -- suitable only for local development
- Frame images sent to MLLM APIs may contain sensitive terminal content

## Directory Structure

```
terminaleyes/
├── src/
│   └── terminaleyes/
│       ├── __init__.py              # Package root, version
│       ├── cli.py                   # CLI entry point
│       ├── domain/
│       │   ├── __init__.py          # Public exports
│       │   └── models.py           # All domain models
│       ├── capture/
│       │   ├── __init__.py
│       │   ├── base.py             # CaptureSource ABC
│       │   └── webcam.py           # WebcamCapture implementation
│       ├── interpreter/
│       │   ├── __init__.py
│       │   ├── base.py             # MLLMProvider ABC
│       │   ├── anthropic.py        # AnthropicProvider
│       │   └── openai.py           # OpenAIProvider
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── base.py             # AgentStrategy ABC
│       │   └── loop.py             # AgentLoop orchestrator
│       ├── keyboard/
│       │   ├── __init__.py
│       │   ├── base.py             # KeyboardOutput ABC
│       │   ├── http_backend.py     # HttpKeyboardOutput
│       │   └── usb_hid_backend.py  # UsbHidKeyboardOutput (placeholder)
│       ├── endpoint/
│       │   ├── __init__.py
│       │   ├── server.py           # FastAPI app and routes
│       │   ├── shell.py            # PersistentShell
│       │   └── display.py          # TerminalDisplay (pygame)
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py         # Settings models and loader
│       └── utils/
│           ├── __init__.py
│           ├── imaging.py          # Image encoding utilities
│           └── logging.py          # Logging setup
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── unit/
│   │   ├── test_capture/
│   │   ├── test_interpreter/
│   │   ├── test_agent/
│   │   ├── test_keyboard/
│   │   ├── test_endpoint/
│   │   └── test_config/
│   └── integration/
├── config/
│   └── terminaleyes.yaml.example   # Example configuration
├── docs/
├── pyproject.toml
├── README.md
├── ARCHITECTURE.md
└── TICKETS.md                      # Implementation tracking
```
