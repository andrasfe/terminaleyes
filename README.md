# terminaleyes

A vision-based agentic terminal controller. The agent controls a terminal purely through visual feedback (webcam) and keystroke output.

## Overview

terminaleyes implements an indirect control loop: it captures what a terminal looks like via webcam, interprets the screen content using a multimodal LLM (Claude, GPT-4V), decides what to do next, and sends keystrokes to the terminal. This architecture allows it to work with physical machines via a Raspberry Pi USB HID keyboard.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

```
Webcam --> MLLM Interpreter --> Agent Decision --> Keyboard Output --> Terminal
   ^                                                                      |
   +----------------------------------------------------------------------+
                        (visual feedback loop)
```

## Installation

```bash
# From the project root
pip install -e ".[dev]"
```

## Configuration

1. Copy the example configuration:
   ```bash
   cp config/terminaleyes.yaml.example config/terminaleyes.yaml
   ```

2. Set your API keys via environment variables:
   ```bash
   export TERMINALEYES_ANTHROPIC_API_KEY=sk-ant-...
   export TERMINALEYES_OPENAI_API_KEY=sk-...
   ```

3. Adjust settings in `config/terminaleyes.yaml` as needed.

## Usage

### Start the HTTP endpoint (terminal simulator)

```bash
terminaleyes endpoint
```

This starts the local HTTP server with a persistent shell and terminal display window.

### Run the agent

```bash
terminaleyes run --goal "List all files in /tmp" --success-criteria "ls output is visible"
```

### Test webcam capture

```bash
terminaleyes capture-test
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=terminaleyes

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## Implementation Status

See [TICKETS.md](TICKETS.md) for detailed implementation tracking.

## License

MIT
