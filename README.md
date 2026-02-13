# terminaleyes

A vision-based agentic terminal controller. The agent controls a terminal purely through visual feedback (webcam) and keystroke output — no screen scraping, no API access to the shell.

## How It Works

```
Webcam --> MLLM Interpreter --> Agent Strategy --> Keyboard Output --> Terminal
   ^                                                                      |
   +----------------------------------------------------------------------+
                        (visual feedback loop)
```

1. A **pygame fullscreen display** renders a persistent shell session (white background, black text, large monospace font)
2. A **webcam** captures what the screen looks like at 1920x1080
3. A **multimodal LLM** (via OpenRouter) interprets the captured image — reading visible text, detecting prompts, errors, etc.
4. An **agent strategy** decides the next keyboard action based on the goal and terminal state
5. The action is sent via **HTTP** to the endpoint, which feeds it to the shell
6. The display updates, and the loop repeats

The HTTP endpoint is a temporary stand-in — the architecture is designed so the keyboard output can be swapped for a Raspberry Pi USB HID device to control physical machines.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

### 1. Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
VISION_MODEL=google/gemini-2.0-flash-lite-001
```

### 2. Edit `config/terminaleyes.yaml`:

```yaml
capture:
  device_index: 0
  capture_interval: 2.0
  crop_enabled: false
  resolution_width: 1920
  resolution_height: 1080
mllm:
  max_tokens: 1024
endpoint:
  host: 0.0.0.0
  port: 8080
  shell_command: /bin/bash
  terminal_rows: 10
  terminal_cols: 25
  font_size: 24
  fg_color: [0, 0, 0]
  bg_color: [255, 255, 255]
  fullscreen: true
keyboard:
  backend: http
  http_base_url: http://localhost:8080
agent:
  action_delay: 2.5
  max_consecutive_errors: 5
  default_max_steps: 100
logging:
  level: INFO
```

Key settings for reliable MLLM reading:
- **White background + black text** — dramatically better OCR accuracy than dark terminals
- **1920x1080 webcam** — 9x more pixels than the default 640x480
- **Few rows/cols** (10x25) — auto-scales font to ~126px, easily readable through camera
- **Fullscreen** — maximizes text size on display

## Usage

### Start the endpoint (terminal + display)

```bash
terminaleyes endpoint
```

This opens a fullscreen pygame window rendering a persistent bash shell, and starts an HTTP server on port 8080 for receiving keyboard commands.

### Run the agent

```bash
terminaleyes run --goal "List files in the current directory" \
  --success-criteria "ls output is visible" \
  --max-steps 20
```

### Validate MLLM reading accuracy

Captures a webcam frame, sends it to the MLLM, and compares the interpretation against the actual screen content:

```bash
terminaleyes validate
```

### Calibrate camera position

Auto-detects where the terminal display appears in the webcam by flashing the screen white and black:

```bash
terminaleyes calibrate
```

### Test webcam capture

```bash
terminaleyes capture-test
```

## API Endpoints

When the endpoint server is running:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status |
| GET | `/screen` | Current terminal text content |
| POST | `/text` | Send text input `{"text": "ls -la\n"}` |
| POST | `/keystroke` | Send a key `{"key": "Enter"}` |
| POST | `/key-combo` | Send combo `{"modifiers": ["ctrl"], "key": "c"}` |

## Architecture

- **`src/terminaleyes/capture/`** — Webcam capture via OpenCV
- **`src/terminaleyes/interpreter/`** — MLLM providers (OpenRouter/OpenAI-compatible)
- **`src/terminaleyes/agent/`** — Agent loop and strategies
- **`src/terminaleyes/endpoint/`** — HTTP server, PTY shell, pygame display
- **`src/terminaleyes/config/`** — Settings from YAML + `.env`
- **`src/terminaleyes/calibration.py`** — Camera-to-terminal calibration

## License

MIT
