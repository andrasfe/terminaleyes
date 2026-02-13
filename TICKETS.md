# terminaleyes Implementation Tickets

This file tracks the implementation status of all components. Each ticket
represents a concrete unit of work with clear acceptance criteria.

Status legend:
- `SCAFFOLDED` -- Interface and structure created, needs implementation
- `NOT_STARTED` -- Not yet begun
- `IN_PROGRESS` -- Actively being worked on
- `COMPLETE` -- Implemented and tested
- `BLOCKED` -- Cannot proceed until dependencies are resolved

---

## Phase 1: Core Infrastructure

### TICK-001: Domain Models
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/domain/models.py`
- **Description:** All domain models are defined with Pydantic. Models are complete and usable.
- **Acceptance Criteria:**
  - [x] CapturedFrame model with numpy array support
  - [x] TerminalState and TerminalContent models
  - [x] Keyboard action discriminated union (Keystroke, KeyCombo, TextInput)
  - [x] AgentGoal, AgentAction, AgentContext models
  - [x] All models have type hints and docstrings
  - [ ] Unit tests for model validation edge cases

### TICK-002: Configuration Management
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/config/settings.py`
- **Dependencies:** None
- **Description:** YAML config loading with Pydantic validation and env var overrides. The `load_settings()` function is implemented. Settings models are complete.
- **Acceptance Criteria:**
  - [x] All config section models defined
  - [x] YAML loading implemented in load_settings()
  - [x] Environment variable override for API keys
  - [x] Example config file created
  - [ ] Full test coverage for config validation

### TICK-003: Logging Setup
- **Status:** COMPLETE
- **Priority:** MEDIUM
- **File:** `src/terminaleyes/utils/logging.py`
- **Description:** `setup_logging()` is fully implemented.
- **Acceptance Criteria:**
  - [x] setup_logging() configures root logger
  - [x] Console handler (stderr) always added
  - [x] File handler added when configured
  - [x] Log level configurable

---

## Phase 2: Vision Pipeline

### TICK-010: CaptureSource Abstract Interface
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/capture/base.py`
- **Description:** ABC is defined with all method signatures. `stream()` is implemented.
- **Acceptance Criteria:**
  - [x] Abstract methods: open, close, capture_frame
  - [x] stream() async iterator implemented
  - [x] Async context manager implemented
  - [ ] CaptureError exception tested

### TICK-011: WebcamCapture Implementation
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/capture/webcam.py`
- **Dependencies:** TICK-010
- **Description:** OpenCV-based webcam capture. All methods raise NotImplementedError.
- **Acceptance Criteria:**
  - [ ] open() initializes cv2.VideoCapture
  - [ ] close() releases VideoCapture
  - [ ] capture_frame() captures and returns CapturedFrame
  - [ ] _capture_sync() runs in thread pool executor
  - [ ] _apply_crop() correctly slices numpy arrays
  - [ ] Resolution override works
  - [ ] Integration test with real webcam (manual)

### TICK-012: Image Encoding Utilities
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/utils/imaging.py`
- **Dependencies:** None
- **Description:** Image format conversion utilities. All functions raise NotImplementedError.
- **Acceptance Criteria:**
  - [ ] numpy_to_base64_png produces valid base64 PNG
  - [ ] numpy_to_pil converts BGR to RGB correctly
  - [ ] pil_to_numpy converts RGB to BGR correctly
  - [ ] resize_for_mllm preserves aspect ratio
  - [ ] Unit tests with sample images

---

## Phase 3: MLLM Integration

### TICK-020: MLLMProvider Abstract Interface
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/interpreter/base.py`
- **Description:** ABC with shared utilities. `_encode_frame_to_base64` and `_parse_response` raise NotImplementedError.
- **Acceptance Criteria:**
  - [x] Abstract methods: interpret, health_check
  - [x] DEFAULT_SYSTEM_PROMPT defined
  - [ ] _encode_frame_to_base64 implemented
  - [ ] _parse_response implemented with JSON extraction
  - [ ] MLLMError exception tested

### TICK-021: Anthropic Provider Implementation
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/interpreter/anthropic.py`
- **Dependencies:** TICK-020, TICK-012
- **Description:** Claude vision API integration. All methods raise NotImplementedError.
- **Acceptance Criteria:**
  - [ ] _ensure_client creates AsyncAnthropic client
  - [ ] interpret() sends image and receives structured response
  - [ ] health_check() verifies API connectivity
  - [ ] Rate limit handling
  - [ ] Timeout handling
  - [ ] Integration test with real API (manual, requires key)

### TICK-022: OpenAI Provider Implementation
- **Status:** SCAFFOLDED
- **Priority:** MEDIUM
- **File:** `src/terminaleyes/interpreter/openai.py`
- **Dependencies:** TICK-020, TICK-012
- **Description:** GPT-4V vision API integration. All methods raise NotImplementedError.
- **Acceptance Criteria:**
  - [ ] _ensure_client creates AsyncOpenAI client
  - [ ] interpret() sends image and receives structured response
  - [ ] health_check() verifies API connectivity
  - [ ] Rate limit handling
  - [ ] Timeout handling
  - [ ] Integration test with real API (manual, requires key)

---

## Phase 4: Keyboard Output

### TICK-030: KeyboardOutput Abstract Interface
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/keyboard/base.py`
- **Description:** ABC is defined. `send_line()` convenience method implemented.
- **Acceptance Criteria:**
  - [x] Abstract methods: connect, disconnect, send_keystroke, send_key_combo, send_text
  - [x] send_line() convenience method
  - [x] Async context manager
  - [ ] KeyboardOutputError tested

### TICK-031: HTTP Keyboard Backend
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/keyboard/http_backend.py`
- **Dependencies:** TICK-030
- **Description:** httpx-based HTTP client for the local endpoint. All methods raise NotImplementedError.
- **Acceptance Criteria:**
  - [ ] connect() creates httpx client and verifies /health
  - [ ] disconnect() closes the client
  - [ ] send_keystroke() POSTs to /keystroke
  - [ ] send_key_combo() POSTs to /key-combo
  - [ ] send_text() POSTs to /text
  - [ ] Error wrapping in KeyboardOutputError
  - [ ] Integration test with running endpoint

### TICK-032: USB HID Keyboard Backend
- **Status:** COMPLETE
- **Priority:** HIGH
- **File:** `src/terminaleyes/keyboard/usb_hid_backend.py`
- **Dependencies:** TICK-030, TICK-033
- **Description:** Wraps HidWriter to implement KeyboardOutput ABC for direct Pi usage.
- **Acceptance Criteria:**
  - [x] USB HID scan code mapping (raspi/hid_codes.py)
  - [x] Report descriptor building (scripts/setup_usb_gadget.sh)
  - [x] Pi communication protocol: REST API (raspi/server.py) or direct HID writes
  - [x] Key press/release timing with configurable delays
  - [x] Unit tests (test_usb_hid_backend.py)
  - [ ] Integration test with hardware (manual, pending Pi arrival)

### TICK-033: Raspberry Pi HID Scan Code Mapping
- **Status:** COMPLETE
- **Priority:** HIGH
- **File:** `src/terminaleyes/raspi/hid_codes.py`
- **Dependencies:** None
- **Description:** Full USB HID Usage Table mapping: key names to scan codes, character-to-HID conversion, modifier bitmasks, shifted character handling (US layout).
- **Acceptance Criteria:**
  - [x] KEY_CODES dict: a-z, 0-9, F1-F12, arrows, nav, punctuation
  - [x] MODIFIER_MAP: ctrl/shift/alt/meta with left/right variants
  - [x] SHIFT_CHARS: uppercase letters + shifted symbols
  - [x] char_to_hid() returns (modifier, scan_code) for any printable char
  - [x] key_name_to_hid() resolves named keys
  - [x] modifiers_to_bitmask() combines modifier names
  - [x] 24 unit tests passing

### TICK-034: HID Report Writer
- **Status:** COMPLETE
- **Priority:** HIGH
- **File:** `src/terminaleyes/raspi/hid_writer.py`
- **Dependencies:** TICK-033
- **Description:** Async writer that opens /dev/hidg0 and sends 8-byte USB HID keyboard reports. Handles press/release timing, text typing, key combos.
- **Acceptance Criteria:**
  - [x] open()/close() manage file descriptor via asyncio executor
  - [x] press_key() writes 8-byte report with modifier + scan code
  - [x] release_keys() writes all-zeros report
  - [x] tap_key() does press + delay + release
  - [x] send_keystroke/send_key_combo/send_text high-level methods
  - [x] Async context manager support
  - [x] Configurable keypress_delay and inter_char_delay
  - [x] 16 unit tests passing (mocked /dev/hidg0)
  - [ ] Integration test with hardware (manual, pending Pi arrival)

### TICK-035: Pi REST API Server
- **Status:** COMPLETE
- **Priority:** HIGH
- **File:** `src/terminaleyes/raspi/server.py`
- **Dependencies:** TICK-034
- **Description:** FastAPI server running on the Pi that accepts keyboard commands over HTTP and routes them to HidWriter. Same API contract as endpoint/server.py so HttpKeyboardOutput works unchanged.
- **Acceptance Criteria:**
  - [x] GET /health returns HID device status
  - [x] POST /keystroke sends key via HID
  - [x] POST /key-combo sends modifier+key via HID
  - [x] POST /text types text character by character via HID
  - [x] Error handling returns 400 for bad keys/modifiers
  - [x] Application factory with injectable writer for testing
  - [x] Lifespan manages HidWriter open/close
  - [x] `terminaleyes-pi` entry point in pyproject.toml
  - [x] 8 unit tests passing
  - [ ] Integration test with hardware (manual, pending Pi arrival)

### TICK-036: USB Gadget Setup Script
- **Status:** COMPLETE
- **Priority:** HIGH
- **File:** `scripts/setup_usb_gadget.sh`
- **Dependencies:** None (run on Pi hardware)
- **Description:** Shell script that configures Pi Zero USB OTG as HID keyboard gadget via Linux ConfigFS. Creates /dev/hidg0 with standard boot keyboard report descriptor.
- **Acceptance Criteria:**
  - [x] Loads libcomposite kernel module
  - [x] Creates gadget under /sys/kernel/config/usb_gadget/
  - [x] Sets USB device descriptor (vendor, product, serial)
  - [x] Writes HID report descriptor (8-byte boot keyboard)
  - [x] Binds to UDC (USB Device Controller)
  - [x] Teardown mode to cleanly remove gadget
  - [x] Error handling for missing dwc2/libcomposite
  - [ ] Test on actual Pi hardware (pending arrival)

---

## Phase 5: Local Endpoint

### TICK-040: PersistentShell
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/endpoint/shell.py`
- **Dependencies:** None
- **Description:** Long-running shell subprocess manager. All methods raise NotImplementedError.
- **Acceptance Criteria:**
  - [ ] start() creates subprocess with stdin/stdout pipes
  - [ ] stop() gracefully terminates the process
  - [ ] send_input() writes to stdin
  - [ ] send_signal() sends SIGINT/SIGTSTP/EOF
  - [ ] get_screen_content() returns visible terminal lines
  - [ ] _read_output_loop() continuously updates screen buffer
  - [ ] Consider pty for realistic terminal emulation
  - [ ] Shell survives across multiple commands

### TICK-041: TerminalDisplay
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/endpoint/display.py`
- **Dependencies:** None
- **Description:** pygame-based terminal window renderer. All methods raise NotImplementedError.
- **Acceptance Criteria:**
  - [ ] start() launches pygame window in background thread
  - [ ] stop() closes the window cleanly
  - [ ] update_content() is thread-safe
  - [ ] Monospace font rendering
  - [ ] Dark background, light text
  - [ ] Correct row/column sizing
  - [ ] Optional cursor rendering
  - [ ] Window is readable by webcam (manual test)

### TICK-042: FastAPI Endpoint Server
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/endpoint/server.py`
- **Dependencies:** TICK-040, TICK-041
- **Description:** HTTP server with routes for receiving keyboard actions. All handlers raise NotImplementedError.
- **Acceptance Criteria:**
  - [ ] GET /health returns status
  - [ ] POST /keystroke maps keys to shell input
  - [ ] POST /key-combo maps combos to signals/control chars
  - [ ] POST /text feeds text to shell stdin
  - [ ] GET /screen returns screen buffer (debug)
  - [ ] Lifespan manages shell and display lifecycle
  - [ ] Integration test: keystroke -> shell -> display update

---

## Phase 6: Agent Loop

### TICK-050: AgentStrategy Abstract Interface
- **Status:** SCAFFOLDED
- **Priority:** HIGH
- **File:** `src/terminaleyes/agent/base.py`
- **Description:** ABC for pluggable decision-making strategies.
- **Acceptance Criteria:**
  - [x] Abstract methods: decide_action, evaluate_completion, name
  - [ ] At least one concrete strategy implemented

### TICK-051: AgentLoop Implementation
- **Status:** SCAFFOLDED
- **Priority:** HIGH (critical path)
- **File:** `src/terminaleyes/agent/loop.py`
- **Dependencies:** TICK-010, TICK-020, TICK-030, TICK-050
- **Description:** Central orchestrator. run() method raises NotImplementedError.
- **Acceptance Criteria:**
  - [ ] run() implements the full capture-interpret-decide-act loop
  - [ ] stop() gracefully stops the loop
  - [ ] Error counting and abort on max consecutive errors
  - [ ] Step limit enforcement
  - [ ] Context accumulation (observations and actions)
  - [ ] Clean resource cleanup on exit
  - [ ] Integration test with mocked components

### TICK-052: Shell Command Strategy
- **Status:** NOT_STARTED
- **Priority:** HIGH
- **File:** To be created at `src/terminaleyes/agent/strategies/shell_command.py`
- **Dependencies:** TICK-050
- **Description:** A concrete strategy for executing shell commands.
- **Acceptance Criteria:**
  - [ ] decide_action() types commands when terminal is ready
  - [ ] decide_action() waits when terminal is busy
  - [ ] evaluate_completion() checks for expected output
  - [ ] Handles error states
  - [ ] Unit tests with mock observations

---

## Phase 7: CLI and Integration

### TICK-060: CLI Entry Point
- **Status:** SCAFFOLDED
- **Priority:** MEDIUM
- **File:** `src/terminaleyes/cli.py`
- **Dependencies:** TICK-051, TICK-042
- **Description:** argparse CLI with run, endpoint, and capture-test commands. Commands log errors and exit.
- **Acceptance Criteria:**
  - [ ] 'run' command initializes all components and starts agent loop
  - [ ] 'endpoint' command starts the HTTP server
  - [ ] 'capture-test' command captures and saves a frame
  - [ ] Config file path override works
  - [ ] Verbose flag enables debug logging

### TICK-061: End-to-End Integration Test
- **Status:** NOT_STARTED
- **Priority:** MEDIUM
- **File:** `tests/integration/test_full_loop.py`
- **Dependencies:** All Phase 1-6 tickets
- **Description:** Full integration test with the endpoint running locally.
- **Acceptance Criteria:**
  - [ ] Start endpoint server in background
  - [ ] Agent captures terminal display
  - [ ] Agent interprets screen (with mock MLLM)
  - [ ] Agent sends keystroke to endpoint
  - [ ] Verify shell received and executed command
  - [ ] Verify display updated

---

## Suggested Implementation Order

1. **TICK-012** (Image Encoding) -- No dependencies, foundational
2. **TICK-020** (MLLM Base _parse_response/_encode) -- Needed by providers
3. **TICK-011** (WebcamCapture) -- Needed for the pipeline
4. **TICK-040** (PersistentShell) -- Core of the endpoint
5. **TICK-041** (TerminalDisplay) -- Parallel with shell
6. **TICK-042** (FastAPI Server) -- Combines shell + display
7. **TICK-031** (HTTP Keyboard Backend) -- Talks to the server
8. **TICK-021** (Anthropic Provider) -- Primary MLLM provider
9. **TICK-052** (Shell Command Strategy) -- First concrete strategy
10. **TICK-051** (AgentLoop) -- Ties everything together
11. **TICK-060** (CLI) -- Final user-facing integration
12. **TICK-061** (E2E Test) -- Validates the full system
