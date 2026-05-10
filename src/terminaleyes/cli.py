"""Command-line interface for the terminaleyes agent.

Provides the main entry point for running the agent loop, starting
the endpoint server, or running individual components for testing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="terminaleyes",
        description="Vision-based agentic terminal controller",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file (default: config/terminaleyes.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save all session screenshots and "
             "artefacts. If unset, falls back to "
             "$TERMINALEYES_OUTPUT_DIR or "
             "~/.local/share/terminaleyes/runs/<timestamp>/",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run the agent loop")
    run_parser.add_argument(
        "--goal", type=str, required=True,
        help="Goal description for the agent",
    )
    run_parser.add_argument(
        "--success-criteria", type=str, default="",
        help="How to determine the goal is achieved",
    )
    run_parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Maximum steps before the agent gives up",
    )

    subparsers.add_parser("endpoint", help="Start the HTTP command endpoint server")
    subparsers.add_parser("capture-test", help="Test webcam capture (saves a frame)")
    subparsers.add_parser("validate", help="Capture frame, interpret via MLLM, compare with actual screen")
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Auto-detect terminal position in webcam and save crop config",
    )
    calibrate_parser.add_argument(
        "--no-save", action="store_true",
        help="Show results without saving to config",
    )

    interact_parser = subparsers.add_parser(
        "interact",
        help="Interactive visual control — ask questions and give commands via REPL",
    )
    interact_parser.add_argument(
        "--screen-check", action="store_true",
        help="Run initial screen visibility check (skipped by default)",
    )
    interact_parser.add_argument(
        "--calibrate", action="store_true",
        help="Deprecated no-op (homing is now closed-loop, no calibration needed)",
    )
    interact_parser.add_argument(
        "-m", "--message", type=str, default=None,
        help='Execute a single command then exit (e.g. -m "click the Run button")',
    )

    command_parser = subparsers.add_parser(
        "command",
        help="Watch screen via webcam and act on visual conditions",
    )
    command_parser.add_argument(
        "instruction", type=str,
        help='Natural language instruction, e.g. "when you see a blue Run button, click it"',
    )
    command_parser.add_argument(
        "--interval", type=float, default=None,
        help="Seconds between captures (default: config value, 180)",
    )
    command_parser.add_argument(
        "--one-shot", action="store_true",
        help="Stop after the first trigger (default: keep watching continuously)",
    )
    command_parser.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate conditions but don't execute actions",
    )

    do_parser = subparsers.add_parser(
        "do",
        help="Run the ControllerAgent on a high-level intent (chains agents)",
    )
    do_parser.add_argument(
        "intent", type=str,
        help="Free-form intent, e.g. 'login and open reddit.com'",
    )
    do_parser.add_argument(
        "--no-focus", action="store_true",
        help="Don't prepend FocusAgent before clicks/navigation",
    )
    do_parser.add_argument(
        "--vault", type=str, default=None,
        help="Vault entry to use when the plan includes a login step",
    )
    do_parser.add_argument(
        "--platform", choices=["linux", "macos"], default="linux",
        help="Target platform (selects key combos). Default linux.",
    )
    do_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan without executing",
    )
    do_parser.add_argument(
        "--no-llm-fallback", action="store_true",
        help="Disable the LLM-planner fallback when no rule matches",
    )
    do_parser.add_argument(
        "--cc-url", type=str, default=None,
        help="Send the intent to a running Command Center instead of "
             "running locally (default: auto-detect "
             "http://127.0.0.1:8765, fall back to local).",
    )
    do_parser.add_argument(
        "--local", action="store_true",
        help="Force local execution even if a Command Center is "
             "running. Useful for debugging without UI overhead.",
    )

    focus_parser = subparsers.add_parser(
        "focus",
        help="Verify the foreground app is centred/maximised; fix if not",
    )
    focus_parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="How many corrective actions to try before giving up (default 3).",
    )
    focus_parser.add_argument(
        "--platform", choices=["linux", "macos"], default="linux",
        help="Which key-combo set to use (default linux/GNOME).",
    )

    vault_parser = subparsers.add_parser(
        "vault",
        help="Manage the local encrypted vault for secrets",
    )
    vault_sub = vault_parser.add_subparsers(
        dest="vault_command", help="Vault operations",
    )
    v_add = vault_sub.add_parser(
        "add", help="Store/overwrite a named secret (value via getpass)",
    )
    v_add.add_argument("name", type=str, help="Entry name")
    v_get = vault_sub.add_parser(
        "get", help="Print a stored secret to stdout (warns if TTY)",
    )
    v_get.add_argument("name", type=str)
    v_get.add_argument(
        "--no-confirm", action="store_true",
        help="Skip the TTY confirmation prompt",
    )
    v_list = vault_sub.add_parser(
        "list", help="List entry names (never values)",
    )
    v_remove = vault_sub.add_parser(
        "remove", help="Delete an entry",
    )
    v_remove.add_argument("name", type=str)
    vault_sub.add_parser(
        "status", help="Show backend and vault file path",
    )

    login_parser = subparsers.add_parser(
        "login",
        help="Wake the remote screen and type the login password",
    )
    login_parser.add_argument(
        "--password-file", type=str, default=None,
        help="Path to a single-line file holding the password. "
             "The path appears in `ps`; the password content does not.",
    )
    login_parser.add_argument(
        "--password-env", type=str, default=None,
        help="Read the password from this environment variable.",
    )
    login_parser.add_argument(
        "--vault", type=str, default=None,
        help="Read the password from the local vault under this name "
             "(see `terminaleyes vault add`).",
    )
    login_parser.add_argument(
        "--no-wake", action="store_true",
        help="Skip the wake sequence (mouse jiggle + Down + click).",
    )
    login_parser.add_argument(
        "--click-input", action="store_true",
        help="Use the visual homer to click the password field "
             "before typing (default: rely on auto-focus).",
    )
    login_parser.add_argument(
        "--no-submit", action="store_true",
        help="Type the password but do not press Enter.",
    )
    login_parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the visual login-screen check (default: a vision "
             "model decides whether the current screen LOOKS like a "
             "login/password prompt before typing).",
    )
    login_parser.add_argument(
        "--verify-attempts", type=int, default=6,
        help="Number of times to re-check the screen if the first "
             "look does not show a login prompt (default 6). Each "
             "attempt nudges the mouse / arrow keys between checks.",
    )
    login_parser.add_argument(
        "--verify-interval", type=float, default=1.0,
        help="Seconds to wait between verification polls (default 1.0).",
    )

    watch_parser = subparsers.add_parser(
        "watch",
        help="Passively watch a screen via webcam and build a session summary",
    )
    watch_parser.add_argument(
        "--interval", type=float, default=None,
        help="Minutes between captures (default: config value, 3)",
    )
    watch_parser.add_argument(
        "--duration", type=float, default=None,
        help="Hours to run (default: config value, 1)",
    )
    watch_parser.add_argument(
        "--output", type=str, default=None,
        help="Save session JSON to this file",
    )

    cc_parser = subparsers.add_parser(
        "commandcenter",
        aliases=["cc"],
        help="Start the Command Center web UI (FastAPI on LAN)",
    )
    cc_parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind address (default: 0.0.0.0 — LAN reachable)",
    )
    cc_parser.add_argument(
        "--port", type=int, default=8765,
        help="Port (default: 8765)",
    )
    cc_parser.add_argument(
        "--frames-dir", type=str, default=None,
        help="Watch directory for PNG frames "
             "(default: $TERMINALEYES_OUTPUT_DIR or "
             "~/.local/share/terminaleyes/runs/)",
    )
    cc_parser.add_argument(
        "--max-frames", type=int, default=500,
        help="Max frames to keep in the index (default: 500)",
    )
    cc_parser.add_argument(
        "--device-index", type=int, default=None,
        help="Capture device index to use for the boot frame and "
             "every run (overrides settings.capture.device_index "
             "from config / TERMINALEYES_CAPTURE__DEVICE_INDEX env). "
             "Typical: 0 = built-in webcam, 1 = USB capture card.",
    )
    cc_parser.add_argument(
        "--no-boot-frame", action="store_true",
        help="Skip the one-shot capture taken at startup so the UI "
             "has something to show before the first run.",
    )

    memory_parser = subparsers.add_parser(
        "memory",
        help="Show / edit / clear the controller's long-lived "
             "memory (markdown file injected into the LLM-planner "
             "prompt at run start).",
    )
    memory_parser.add_argument(
        "memory_action", nargs="?", default="show",
        choices=["show", "path", "edit", "clear"],
        help="show (default): print contents; "
             "path: print the file path; "
             "edit: open in $EDITOR; "
             "clear: delete the file.",
    )

    return parser.parse_args(argv)


async def _run_agent(settings, args) -> None:
    """Initialize all components and run the agent loop."""
    from terminaleyes.capture.webcam import WebcamCapture
    from terminaleyes.domain.models import AgentGoal, CropRegion
    from terminaleyes.interpreter.openai import OpenAIProvider
    from terminaleyes.keyboard.http_backend import HttpKeyboardOutput
    from terminaleyes.agent.loop import AgentLoop
    from terminaleyes.agent.strategies.mllm_driven import MLLMDrivenStrategy

    # Build capture source
    crop = None
    if settings.capture.crop_enabled:
        crop = CropRegion(
            x=settings.capture.crop_x,
            y=settings.capture.crop_y,
            width=settings.capture.crop_width,
            height=settings.capture.crop_height,
        )
    resolution = None
    if settings.capture.resolution_width and settings.capture.resolution_height:
        resolution = (settings.capture.resolution_width, settings.capture.resolution_height)

    capture = WebcamCapture(
        device_index=settings.capture.device_index,
        crop_region=crop,
        resolution=resolution,
    )

    # Build MLLM provider
    api_key = ""
    base_url = settings.mllm.base_url
    if settings.mllm.provider == "openai":
        api_key = settings.openai_api_key.get_secret_value()
    # If OpenRouter key is set, use it
    or_key = settings.openrouter_api_key.get_secret_value()
    if or_key:
        api_key = or_key
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"

    interpreter = OpenAIProvider(
        api_key=api_key,
        model=settings.mllm.model,
        base_url=base_url,
        max_tokens=settings.mllm.max_tokens,
    )

    # Build keyboard output
    keyboard = HttpKeyboardOutput(
        base_url=settings.keyboard.http_base_url,
        timeout=settings.keyboard.http_timeout,
    )

    # Build strategy
    strategy = MLLMDrivenStrategy(mllm=interpreter)

    # Build agent loop
    loop = AgentLoop(
        capture=capture,
        interpreter=interpreter,
        keyboard=keyboard,
        strategy=strategy,
        capture_interval=settings.capture.capture_interval,
        action_delay=settings.agent.action_delay,
        max_consecutive_errors=settings.agent.max_consecutive_errors,
    )

    goal = AgentGoal(
        goal_id="cli-goal",
        description=args.goal,
        success_criteria=args.success_criteria or args.goal,
        max_steps=args.max_steps,
    )

    result = await loop.run(goal)
    print(f"\nResult: {result.current_goal.status.value}")
    print(f"Steps taken: {result.step_count}")
    if result.action_history:
        print("\nAction history:")
        for action in result.action_history:
            print(f"  [{action.step_number}] {action.action.action_type}: {action.reasoning[:80]}")


async def _capture_test(settings) -> None:
    """Capture a single frame and save to file."""
    from terminaleyes.capture.webcam import WebcamCapture
    from terminaleyes.domain.models import CropRegion
    import cv2

    crop = None
    if settings.capture.crop_enabled:
        crop = CropRegion(
            x=settings.capture.crop_x,
            y=settings.capture.crop_y,
            width=settings.capture.crop_width,
            height=settings.capture.crop_height,
        )

    capture = WebcamCapture(
        device_index=settings.capture.device_index,
        crop_region=crop,
    )

    async with capture:
        frame = await capture.capture_frame()
        outfile = "capture_test.png"
        cv2.imwrite(outfile, frame.image)
        print(f"Saved frame to {outfile} ({frame.image.shape[1]}x{frame.image.shape[0]})")


async def _validate(settings) -> None:
    """Capture a frame, interpret it via MLLM, and compare with actual screen."""
    import httpx
    import cv2
    from terminaleyes.capture.webcam import WebcamCapture
    from terminaleyes.domain.models import CropRegion
    from terminaleyes.interpreter.openai import OpenAIProvider

    # Build capture
    crop = None
    if settings.capture.crop_enabled:
        crop = CropRegion(
            x=settings.capture.crop_x, y=settings.capture.crop_y,
            width=settings.capture.crop_width, height=settings.capture.crop_height,
        )
    resolution = None
    if settings.capture.resolution_width and settings.capture.resolution_height:
        resolution = (settings.capture.resolution_width, settings.capture.resolution_height)
    capture = WebcamCapture(
        device_index=settings.capture.device_index, crop_region=crop,
        resolution=resolution,
    )

    # Build MLLM provider
    api_key = settings.openrouter_api_key.get_secret_value() or settings.openai_api_key.get_secret_value()
    base_url = settings.mllm.base_url
    if not base_url and settings.openrouter_api_key.get_secret_value():
        base_url = "https://openrouter.ai/api/v1"

    interpreter = OpenAIProvider(
        api_key=api_key, model=settings.mllm.model,
        base_url=base_url, max_tokens=settings.mllm.max_tokens,
    )

    # Capture and interpret
    async with capture:
        frame = await capture.capture_frame()
        cv2.imwrite("validate_capture.png", frame.image)
        print(f"Saved webcam capture to validate_capture.png ({frame.image.shape[1]}x{frame.image.shape[0]})")

    print("\nInterpreting via MLLM...")
    state = await interpreter.interpret(frame)

    # Get actual screen content from endpoint
    actual = "(endpoint not available)"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{settings.keyboard.http_base_url}/screen")
            actual = r.json().get("content", "")
    except Exception as e:
        actual = f"(could not fetch: {e})"

    print("\n" + "=" * 60)
    print("MLLM INTERPRETATION")
    print("=" * 60)
    print(f"Readiness:  {state.readiness.value}")
    print(f"Confidence: {state.confidence}")
    print(f"Prompt:     {state.content.prompt_text}")
    print(f"Last cmd:   {state.content.last_command}")
    print(f"Last out:   {state.content.last_output}")
    print(f"Work dir:   {state.content.working_directory}")
    print(f"Errors:     {state.content.error_messages}")
    print(f"\nVisible text ({len(state.content.visible_text)} chars):")
    print("-" * 40)
    print(state.content.visible_text[:500])
    print("-" * 40)

    print("\n" + "=" * 60)
    print("ACTUAL SCREEN CONTENT (from /screen endpoint)")
    print("=" * 60)
    print(actual[:500])
    print("=" * 60)

    # Similarity check (strip empty lines)
    actual_stripped = " ".join(actual.split())
    mllm_stripped = " ".join(state.content.visible_text.split())
    actual_words = set(actual_stripped.lower().split())
    mllm_words = set(mllm_stripped.lower().split())
    if actual_words and mllm_words:
        overlap = len(actual_words & mllm_words)
        total = len(actual_words | mllm_words)
        similarity = overlap / total * 100
        print(f"\nWord overlap: {overlap}/{total} ({similarity:.0f}%)")
        if similarity >= 30:
            print("OK -- MLLM is reading the terminal correctly.")
        else:
            # Check if the MLLM text is a substring of actual or vice versa
            if mllm_stripped.lower() in actual_stripped.lower():
                print("OK -- MLLM text is a subset of the actual screen content.")
            else:
                print("WARNING: Low similarity -- camera may not be focused on the terminal.")
    print()


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the terminaleyes CLI."""
    args = parse_args(argv)

    if args.command is None:
        parse_args(["--help"])
        return

    from terminaleyes.config.settings import load_settings
    from terminaleyes.utils.logging import setup_logging

    settings = load_settings(args.config)

    if args.verbose:
        settings.logging.level = "DEBUG"

    setup_logging(settings.logging)

    if args.command == "run":
        logger.info("Starting agent loop with goal: %s", args.goal)
        asyncio.run(_run_agent(settings, args))

    elif args.command == "endpoint":
        logger.info("Starting endpoint server")
        from terminaleyes.endpoint.server import create_app
        import uvicorn
        ep = settings.endpoint
        window_x = ep.window_x
        window_y = ep.window_y

        # Auto-position text area from calibration data if no explicit position set
        if window_x is None and window_y is None:
            cap = settings.capture
            if cap.crop_width > 0 and cap.crop_height > 0:
                from terminaleyes.calibration import compute_window_position
                import pygame
                pygame.init()
                info = pygame.display.Info()
                screen_w, screen_h = info.current_w, info.current_h
                pygame.quit()

                # Estimate window size from font metrics
                char_w_est = int(ep.font_size * 0.6)
                line_h_est = int(ep.font_size * 1.2)
                pad = 30
                win_w = ep.terminal_cols * char_w_est + pad * 2
                win_h = ep.terminal_rows * line_h_est + pad * 2

                calibration = {
                    "crop_x": cap.crop_x,
                    "crop_y": cap.crop_y,
                    "crop_width": cap.crop_width,
                    "crop_height": cap.crop_height,
                    "frame_width": cap.resolution_width or 1920,
                    "frame_height": cap.resolution_height or 1080,
                }
                window_x, window_y = compute_window_position(
                    calibration, screen_w, screen_h, win_w, win_h,
                )
                logger.info("Auto-positioned window at (%d, %d) size %dx%d", window_x, window_y, win_w, win_h)
            else:
                # No calibration: default to bottom-center of screen
                import pygame
                pygame.init()
                info = pygame.display.Info()
                screen_w, screen_h = info.current_w, info.current_h
                pygame.quit()

                char_w_est = int(ep.font_size * 0.6)
                line_h_est = int(ep.font_size * 1.2)
                pad = 30
                win_w = ep.terminal_cols * char_w_est + pad * 2
                win_h = ep.terminal_rows * line_h_est + pad * 2

                window_x = (screen_w - win_w) // 2
                window_y = screen_h - win_h - 50
                logger.info("No calibration data, defaulting to bottom-center (%d, %d)", window_x, window_y)

        app = create_app(
            shell_command=ep.shell_command,
            rows=ep.terminal_rows,
            cols=ep.terminal_cols,
            font_size=ep.font_size,
            bg_color=ep.bg_color,
            fg_color=ep.fg_color,
            fullscreen=ep.fullscreen,
            window_x=window_x,
            window_y=window_y,
        )
        uvicorn.run(
            app,
            host=ep.host,
            port=ep.port,
        )

    elif args.command == "capture-test":
        logger.info("Running capture test")
        asyncio.run(_capture_test(settings))

    elif args.command == "validate":
        logger.info("Running MLLM validation")
        asyncio.run(_validate(settings))

    elif args.command == "interact":
        logger.info("Starting interactive visual commander")
        asyncio.run(_run_interact(settings, args))

    elif args.command == "command":
        logger.info("Starting visual command agent")
        asyncio.run(_run_command(settings, args))

    elif args.command == "watch":
        logger.info("Starting screen watcher")
        asyncio.run(_watch(settings, args))

    elif args.command == "calibrate":
        logger.info("Running camera calibration")
        asyncio.run(_calibrate(settings, save=not args.no_save))

    elif args.command == "login":
        logger.info("Starting remote login flow")
        asyncio.run(_run_login(settings, args))

    elif args.command == "vault":
        _run_vault(args)

    elif args.command == "focus":
        logger.info("Running focus agent")
        asyncio.run(_run_focus(settings, args))

    elif args.command == "do":
        logger.info("Running controller agent")
        asyncio.run(_run_controller(settings, args))

    elif args.command in ("commandcenter", "cc"):
        logger.info("Starting Command Center")
        asyncio.run(_run_commandcenter(settings, args))

    elif args.command == "memory":
        _run_memory(args)


def _run_memory(args) -> None:
    """show / path / edit / clear the controller's memory file."""
    import os
    import subprocess
    from terminaleyes.agents.controller import (
        _memory_path, load_memory,
    )

    path = _memory_path()
    action = getattr(args, "memory_action", "show") or "show"

    if action == "path":
        print(path)
        return
    if action == "show":
        text = load_memory()
        if not text:
            print(f"(empty — {path} does not exist or is empty)")
            return
        print(f"# {path}")
        print(text)
        return
    if action == "edit":
        editor = os.environ.get("EDITOR") or "nano"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                "# terminaleyes — controller memory\n"
                "\n"
                "Long-lived notes the controller injects into the\n"
                "LLM-planner prompt on every run. Plain markdown.\n"
                "Add anything here that the planner should know\n"
                "across sessions — preferred apps, target machine\n"
                "quirks, naming conventions.\n",
                encoding="utf-8",
            )
        subprocess.call([editor, str(path)])
        return
    if action == "clear":
        if path.exists():
            path.unlink()
            print(f"removed {path}")
        else:
            print(f"(nothing to clear — {path} does not exist)")
        return


async def _run_interact(settings, args=None) -> None:
    """Run the interactive visual commander REPL."""
    from terminaleyes.capture.webcam import WebcamCapture
    from terminaleyes.commander.evaluator import ConditionEvaluator
    from terminaleyes.commander.executor import ActionExecutor
    from terminaleyes.commander.interactive import InteractiveSession
    from terminaleyes.keyboard.http_backend import HttpKeyboardOutput
    from terminaleyes.mouse.http_backend import HttpMouseOutput

    cfg = settings.commander

    print("=" * 60)
    print("INTERACTIVE VISUAL COMMANDER")
    print("=" * 60)
    print(f"LM Studio: {cfg.lmstudio_base_url} ({cfg.lmstudio_model})")
    print(f"Pi: {cfg.pi_base_url} (transport={cfg.transport})")
    print(f"Screen: {cfg.screen_width}x{cfg.screen_height}")
    print()

    # Build webcam capture
    resolution = None
    if settings.capture.resolution_width and settings.capture.resolution_height:
        resolution = (settings.capture.resolution_width, settings.capture.resolution_height)

    capture = WebcamCapture(
        device_index=settings.capture.device_index,
        resolution=resolution,
    )

    # Build evaluator
    evaluator = ConditionEvaluator(
        model=cfg.lmstudio_model,
        base_url=cfg.lmstudio_base_url,
        max_tokens=cfg.lmstudio_max_tokens,
    )

    # Build keyboard + mouse
    keyboard = HttpKeyboardOutput(
        base_url=cfg.pi_base_url,
        timeout=10.0,
        transport=cfg.transport,
    )
    mouse = HttpMouseOutput(
        base_url=cfg.pi_base_url,
        timeout=10.0,
        transport=cfg.transport,
    )

    # Build executor with capture + evaluator for visual homing
    executor = ActionExecutor(
        keyboard=keyboard,
        mouse=mouse,
        screen_width=cfg.screen_width,
        screen_height=cfg.screen_height,
        capture=capture,
        evaluator=evaluator,
    )

    # Connect to Pi
    await keyboard.connect()
    await mouse.connect()

    # Build and run interactive session
    session = InteractiveSession(
        capture=capture,
        evaluator=evaluator,
        executor=executor,
        model=cfg.lmstudio_model,
        base_url=cfg.lmstudio_base_url,
        max_tokens=cfg.lmstudio_max_tokens,
        vision_model=cfg.lmstudio_vision_model,
        vision_base_url=cfg.vision_base_url,
        skip_screen_check=not getattr(args, 'screen_check', False),
        force_calibration=getattr(args, 'calibrate', False),
        single_message=getattr(args, 'message', None),
    )

    try:
        await session.start()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        await keyboard.disconnect()
        await mouse.disconnect()


async def _run_command(settings, args) -> None:
    """Run the visual command agent."""
    import signal
    from terminaleyes.capture.webcam import WebcamCapture
    from terminaleyes.commander.parser import CommandParser
    from terminaleyes.commander.evaluator import ConditionEvaluator
    from terminaleyes.commander.executor import ActionExecutor
    from terminaleyes.commander.loop import CommandLoop
    from terminaleyes.keyboard.http_backend import HttpKeyboardOutput
    from terminaleyes.mouse.http_backend import HttpMouseOutput

    cfg = settings.commander

    print("=" * 60)
    print("VISUAL COMMAND AGENT")
    print("=" * 60)
    print(f"LM Studio: {cfg.lmstudio_base_url} ({cfg.lmstudio_model})")
    print(f"Pi: {cfg.pi_base_url} (transport={cfg.transport})")
    print(f"Screen: {cfg.screen_width}x{cfg.screen_height}")
    print()

    # 1. Parse instruction
    print("Parsing instruction...")
    parser = CommandParser(
        model=cfg.lmstudio_model,
        base_url=cfg.lmstudio_base_url,
        max_tokens=cfg.lmstudio_max_tokens,
    )
    command = await parser.parse(args.instruction)

    # Default is continuous — --one-shot overrides to single trigger
    command = command.model_copy(update={"one_shot": False})
    if args.one_shot:
        command = command.model_copy(update={"one_shot": True})
    if args.interval is not None:
        command = command.model_copy(update={"interval_seconds": args.interval})

    print(f"\nParsed command:")
    print(f"  Condition: {command.condition.description}")
    if command.condition.element_text:
        print(f"  Element text: {command.condition.element_text}")
    if command.condition.visual_cues:
        print(f"  Visual cues: {', '.join(command.condition.visual_cues)}")
    print(f"  Action: {command.action.action_type} ({command.action.button or command.action.key or command.action.text or ''})")
    print(f"  Interval: {command.interval_seconds}s")
    print(f"  Mode: {'one-shot' if command.one_shot else 'continuous'}")
    if args.dry_run:
        print(f"  DRY RUN: actions will NOT be executed")
    print("=" * 60)

    # 2. Build webcam capture (no crop — watching arbitrary screen)
    resolution = None
    if settings.capture.resolution_width and settings.capture.resolution_height:
        resolution = (settings.capture.resolution_width, settings.capture.resolution_height)

    capture = WebcamCapture(
        device_index=settings.capture.device_index,
        resolution=resolution,
    )

    # 3. Build condition evaluator
    evaluator = ConditionEvaluator(
        model=cfg.lmstudio_model,
        base_url=cfg.lmstudio_base_url,
        max_tokens=cfg.lmstudio_max_tokens,
    )

    # 4. Build keyboard + mouse outputs
    keyboard = HttpKeyboardOutput(
        base_url=cfg.pi_base_url,
        timeout=10.0,
        transport=cfg.transport,
    )
    mouse = HttpMouseOutput(
        base_url=cfg.pi_base_url,
        timeout=10.0,
        transport=cfg.transport,
    )

    # 5. Build executor with capture + evaluator for visual cursor homing
    executor = ActionExecutor(
        keyboard=keyboard,
        mouse=mouse,
        screen_width=cfg.screen_width,
        screen_height=cfg.screen_height,
        capture=capture,
        evaluator=evaluator,
    )

    if not args.dry_run:
        await keyboard.connect()
        await mouse.connect()

    # 6. Build and run loop
    loop = CommandLoop(
        capture=capture,
        evaluator=evaluator,
        executor=executor,
        confidence_threshold=cfg.confidence_threshold,
        change_threshold=cfg.change_threshold,
        max_consecutive_errors=cfg.max_consecutive_errors,
    )

    # Ctrl-C: first press sets stop flag, second press force-exits
    _sigint_count = [0]

    def _sigint_handler(sig, frame):
        _sigint_count[0] += 1
        if _sigint_count[0] >= 2:
            print("\nForce exit.")
            import sys
            sys.exit(1)
        print("\nStopping command loop (press Ctrl-C again to force exit)...")
        loop.stop()

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        session = await loop.run(command)
        print(f"\nFinal status: {session.status}")
    finally:
        if not args.dry_run:
            await keyboard.disconnect()
            await mouse.disconnect()


def _resolve_session_dir(args) -> Path:
    """Pick the per-invocation session directory.

    Priority: ``--output-dir`` CLI flag > ``TERMINALEYES_OUTPUT_DIR``
    env var > ``~/.local/share/terminaleyes/runs/<UTC-timestamp>/``.
    The directory is created lazily by :meth:`AgentContext.record_frame`.
    """
    import os
    from datetime import datetime as _dt

    base: Path | None = None
    arg_dir = getattr(args, "output_dir", None)
    if arg_dir is not None:
        base = Path(arg_dir).expanduser().resolve()
    elif os.environ.get("TERMINALEYES_OUTPUT_DIR"):
        base = Path(
            os.environ["TERMINALEYES_OUTPUT_DIR"]
        ).expanduser().resolve()

    ts = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    if base is None:
        base = Path.home() / ".local" / "share" / "terminaleyes" / "runs"
        return base / ts
    # If the user pointed at an existing dir, append a timestamp
    # subdir so each run is independently inspectable.
    if base.exists() and base.is_dir():
        return base / ts
    return base


async def _build_agent_context(
    settings, *, with_capture: bool = True, args=None,
):
    """Construct an AgentContext wired to the Pi + LM Studio.

    Used by tier-3 agents (focus, navigate, etc.). Not used by the
    pure-CLI vault subcommand (which doesn't need any Pi I/O).
    """
    from openai import AsyncOpenAI
    from terminaleyes.agents.context import AgentContext
    from terminaleyes.commander.evaluator import ConditionEvaluator
    from terminaleyes.keyboard.http_backend import HttpKeyboardOutput
    from terminaleyes.mouse.http_backend import HttpMouseOutput

    cfg = settings.commander

    keyboard = HttpKeyboardOutput(
        base_url=cfg.pi_base_url,
        timeout=10.0,
        transport=cfg.transport,
    )
    mouse = HttpMouseOutput(
        base_url=cfg.pi_base_url,
        timeout=10.0,
        transport=cfg.transport,
    )
    await keyboard.connect()
    await mouse.connect()

    capture = None
    if with_capture:
        from terminaleyes.capture.webcam import WebcamCapture
        resolution = None
        if (settings.capture.resolution_width
                and settings.capture.resolution_height):
            resolution = (
                settings.capture.resolution_width,
                settings.capture.resolution_height,
            )
        capture = WebcamCapture(
            device_index=settings.capture.device_index,
            resolution=resolution,
        )
        await capture.open()

    client = AsyncOpenAI(
        base_url=cfg.lmstudio_base_url, api_key="not-needed",
    )
    evaluator = ConditionEvaluator(
        model=cfg.lmstudio_model,
        base_url=cfg.lmstudio_base_url,
        max_tokens=cfg.lmstudio_max_tokens,
    )

    output_dir = _resolve_session_dir(args) if args is not None else None
    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Session output dir: {output_dir}")
        except OSError as e:
            print(f"WARNING: could not create output_dir {output_dir}: {e}")
            output_dir = None

    ctx = AgentContext(
        mouse=mouse,
        keyboard=keyboard,
        capture=capture,
        vision_client=client,
        vision_model=cfg.lmstudio_model,
        ocr_model=cfg.lmstudio_ocr_model,
        evaluator=evaluator,
        output_dir=output_dir,
    )
    return ctx, keyboard, mouse, capture


async def _run_controller(settings, args) -> None:
    """Run the ControllerAgent on a free-form intent.

    Auto-routes through a running Command Center when one is
    detected (so the UI sees every CLI invocation). Falls back to
    in-process local execution otherwise. Pass ``--local`` to skip
    the auto-route, or ``--cc-url URL`` to force routing to a
    specific cc instance.
    """
    from terminaleyes.agents.controller import ControllerAgent

    if not args.local:
        cc_url = args.cc_url or "http://127.0.0.1:8765"
        if await _cc_is_up(cc_url):
            print(f"Routing through Command Center at {cc_url}")
            ok = await _route_through_cc(cc_url, args)
            sys.exit(0 if ok else 1)

    ctx, keyboard, mouse, capture = await _build_agent_context(
        settings, with_capture=True, args=args,
    )
    # Wire the vault lazily — only if the plan needs it.
    try:
        agent = ControllerAgent(ctx)
        outcome = await agent.run(
            intent=args.intent,
            no_focus=args.no_focus,
            vault_name=args.vault,
            platform=args.platform,
            dry_run=args.dry_run,
            allow_llm_fallback=not args.no_llm_fallback,
        )
        if outcome:
            print(f"\n✓ Controller succeeded — {outcome.reason}")
            answer = (outcome.data or {}).get("answer", "") if outcome.data else ""
            if answer:
                print("\nAnswer:")
                for ln in str(answer).splitlines():
                    print(f"  {ln}")
        else:
            print(f"\n✗ Controller failed — {outcome.reason}")
    finally:
        if capture is not None:
            try:
                await capture.close()
            except Exception:
                pass
        await keyboard.disconnect()
        await mouse.disconnect()


async def _cc_is_up(base_url: str, *, timeout: float = 0.5) -> bool:
    """Lightweight probe: is a cc reachable + reporting state?"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{base_url}/api/state")
            return r.status_code == 200
    except Exception:
        return False


async def _route_through_cc(base_url: str, args) -> bool:
    """POST the intent to a running cc and stream its log SSE to
    stdout. Returns ``True`` if the run reported success.

    The cc owns the AgentContext for the run (so the webcam, Pi
    handles, and frame recording all happen in the cc process —
    the UI sees everything live). This CLI process just starts
    the run, prints the audit trail, and exits with the right
    status code.
    """
    import json as _json
    import time as _time

    import httpx

    payload = {
        "intent": args.intent,
        "no_focus": bool(args.no_focus),
        "vault": args.vault,
        "platform": args.platform,
        "dry_run": bool(args.dry_run),
        "allow_llm_fallback": not bool(args.no_llm_fallback),
    }

    async with httpx.AsyncClient(timeout=None) as c:
        # 1. Start the run.
        try:
            r = await c.post(
                f"{base_url}/api/run", json=payload, timeout=10.0,
            )
        except Exception as e:
            print(f"✗ Could not POST run to cc: {e}")
            return False
        if r.status_code == 409:
            print(
                "✗ Command Center is busy with another run; "
                "wait or use --local"
            )
            return False
        if r.status_code != 200:
            print(f"✗ cc rejected run: HTTP {r.status_code} {r.text}")
            return False
        record = r.json()
        run_id = record["run_id"]
        print(f"  cc run_id: {run_id}")

        # 2. Stream the SSE log lines to stdout.
        try:
            async with c.stream(
                "GET",
                f"{base_url}/api/runs/{run_id}/logs",
                timeout=None,
            ) as resp:
                async for raw in resp.aiter_lines():
                    if not raw:
                        continue
                    if raw.startswith("event: done"):
                        break
                    if not raw.startswith("data: "):
                        continue
                    body = raw[len("data: "):]
                    try:
                        ev = _json.loads(body)
                    except _json.JSONDecodeError:
                        continue
                    msg = ev.get("msg", "")
                    if msg:
                        print(msg)
        except Exception as e:
            # SSE may close on success; only complain if we never
            # got a final record below.
            print(f"  (SSE stream ended: {e})")

        # 3. Read the final record for the exit status.
        try:
            r2 = await c.get(
                f"{base_url}/api/runs/{run_id}", timeout=5.0,
            )
            rec = r2.json()
            status = rec.get("status", "unknown")
            reason = rec.get("reason") or ""
            mark = "✓" if status == "succeeded" else "✗"
            print(f"\n{mark} cc run {status}: {reason}")
            return status == "succeeded"
        except Exception as e:
            print(f"  (could not fetch final record: {e})")
            return False


def _check_port_free(host: str, port: int) -> None:
    """Refuse to start cc if ``host:port`` is already bound.

    macOS lets uvicorn workers bind the same port without raising,
    which silently splits browser requests across stale + fresh
    processes. Detect that pre-emptively and exit with a useful
    pointer.
    """
    import socket as _sk
    import sys as _sys

    probe = _sk.socket(_sk.AF_INET, _sk.SOCK_STREAM)
    probe.settimeout(0.5)
    test_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        probe.connect((test_host, port))
        # Connection succeeded → something is already serving here.
        probe.close()
        existing = []
        try:
            import subprocess as _sp
            out = _sp.check_output(
                ["lsof", "-i", f":{port}", "-P", "-sTCP:LISTEN"],
                text=True,
                stderr=_sp.DEVNULL,
            )
            existing = [
                line for line in out.splitlines()[1:]
                if line.strip()
            ]
        except Exception:
            pass
        print(
            f"\n✗ Port {port} is already in use on {test_host}.\n"
        )
        if existing:
            print("Listening processes:")
            for line in existing:
                print(f"  {line}")
            print(
                "\nTo stop existing cc instance(s):"
                "\n  pkill -f 'terminaleyes cc'"
                f"\nThen re-run, or pick a different port with --port.\n"
            )
        else:
            print(
                "Pick a different port with --port, or stop the "
                "existing process.\n"
            )
        _sys.exit(2)
    except (OSError, _sk.timeout):
        # Connection refused → port is free.
        pass
    finally:
        try:
            probe.close()
        except Exception:
            pass


async def _capture_boot_frame(settings, watch_dir) -> None:
    """One-shot capture at server boot so the UI has something to show.

    Saves a single PNG into ``<watch_dir>/_boot_<ts>/``. The FrameStore
    treats that directory as a "run", labels frames with run_id =
    "_boot_<ts>" so the UI can distinguish boot frames from real runs.
    Discards a few warm-up frames so auto-exposure / capture-card sync
    has time to settle.
    """
    import asyncio as _asyncio
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    import cv2 as _cv2

    from terminaleyes.capture.webcam import WebcamCapture

    resolution = None
    if (settings.capture.resolution_width
            and settings.capture.resolution_height):
        resolution = (
            settings.capture.resolution_width,
            settings.capture.resolution_height,
        )
    cap = WebcamCapture(
        device_index=settings.capture.device_index,
        resolution=resolution,
    )
    boot_dir = _Path(watch_dir) / (
        "_boot_" + _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    boot_dir.mkdir(parents=True, exist_ok=True)
    try:
        await cap.open()
        # Discard warm-up frames; the first 1-2 are often black or
        # half-dark while the device wakes up.
        for _ in range(3):
            try:
                await cap.capture_frame()
            except Exception:
                break
            await _asyncio.sleep(0.05)
        frame = await cap.capture_frame()
        ts = _dt.now().strftime("%H%M%S")
        out_path = boot_dir / f"0001_{ts}_boot.png"
        _cv2.imwrite(str(out_path), frame.image)
        print(f"  Boot frame: {out_path}")
    finally:
        try:
            await cap.close()
        except Exception:
            pass


async def _run_commandcenter(settings, args) -> None:
    """Boot the Command Center FastAPI server.

    The server itself does NOT open the webcam. A fresh AgentContext is
    built per run (matching `terminaleyes do`) and torn down when that
    run finishes. While idle, only the static frame watcher is active.
    """
    import socket
    from pathlib import Path

    import uvicorn

    from terminaleyes.commandcenter.factory import (
        make_default_context_factory,
    )
    from terminaleyes.commandcenter.frame_store import (
        DEFAULT_WATCH_DIR, FrameStore,
    )
    from terminaleyes.commandcenter.log_bus import LogBus
    from terminaleyes.commandcenter.server import create_app

    watch_dir = (
        Path(args.frames_dir).expanduser().resolve()
        if args.frames_dir else DEFAULT_WATCH_DIR
    )
    watch_dir.mkdir(parents=True, exist_ok=True)

    # CLI flag wins over settings / env for the capture device. Mutate
    # the settings model so the per-run factory picks up the same
    # value when it builds AgentContexts.
    if args.device_index is not None:
        settings.capture.device_index = args.device_index

    store = FrameStore(watch_dir=watch_dir, max_frames=args.max_frames)
    bus = LogBus()
    # The factory builds a fresh AgentContext per run with output_dir
    # = watch_dir / <run_id>, so frames the agents capture flow into
    # the FrameStore's watch dir and get streamed to the UI.
    context_factory = make_default_context_factory(
        settings, base_dir=watch_dir, bus=bus,
    )
    app = create_app(context_factory, frame_store=store, bus=bus)

    # Pre-flight: refuse to start if another process is already
    # listening on the requested port. macOS will sometimes let
    # multiple Python uvicorn workers bind the same port; that
    # silently splits browser requests across stale and fresh
    # processes which is brutal to debug. Fail loud instead.
    _check_port_free(args.host, args.port)

    # Boot frame: capture once before serving so the UI isn't empty.
    if not args.no_boot_frame:
        try:
            await _capture_boot_frame(settings, watch_dir)
        except Exception as e:
            print(f"  (boot frame skipped: {e})")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        lan_ip = sock.getsockname()[0]
        sock.close()
    except Exception:
        lan_ip = args.host
    print("=" * 60)
    print("TERMINALEYES COMMAND CENTER")
    print("=" * 60)
    print(f"  Local:   http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  LAN:     http://{lan_ip}:{args.port}")
    print(f"  Frames:  {watch_dir}")
    print(f"  Capture: device {settings.capture.device_index}")
    print("=" * 60)

    config = uvicorn.Config(
        app, host=args.host, port=args.port,
        log_level="info", access_log=False,
    )
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        try:
            await store.stop()
        except Exception:
            pass


async def _run_focus(settings, args) -> None:
    """Run the FocusAgent end-to-end."""
    from terminaleyes.agents.focus import FocusAgent

    ctx, keyboard, mouse, capture = await _build_agent_context(
        settings, with_capture=True, args=args,
    )
    try:
        agent = FocusAgent(ctx)
        outcome = await agent.run(
            max_attempts=args.max_attempts,
            platform=args.platform,
        )
        if outcome:
            print(f"✓ FocusAgent succeeded — {outcome.reason}")
        else:
            print(f"✗ FocusAgent failed — {outcome.reason}")
    finally:
        if capture is not None:
            try:
                await capture.close()
            except Exception:
                pass
        await keyboard.disconnect()
        await mouse.disconnect()


def _run_vault(args) -> None:
    """Dispatch ``terminaleyes vault <subcommand>``."""
    import sys
    from terminaleyes.agents.vault import (
        Vault, VaultError, get_passphrase,
    )

    sub = getattr(args, "vault_command", None)
    if sub is None:
        print(
            "Usage: terminaleyes vault {add|get|list|remove|status} ..."
        )
        return

    if sub == "status":
        # Don't decrypt for status — just file metadata.
        from terminaleyes.agents.vault import DEFAULT_PATH
        path = DEFAULT_PATH
        print(f"Backend : file (AES-256-GCM, scrypt KDF)")
        print(f"Path    : {path}")
        print(f"Exists  : {path.exists()}")
        if path.exists():
            import os as _os
            mode = oct(_os.stat(path).st_mode & 0o777)
            print(f"Mode    : {mode}")
        return

    try:
        passphrase = get_passphrase()
        vault = Vault(passphrase)

        if sub == "add":
            import getpass as _gp
            value = _gp.getpass(f"Value for {args.name!r}: ")
            if not value:
                print("Refusing to store empty value.")
                return
            vault.set(args.name, value)
            print(f"Stored {args.name!r}.")
        elif sub == "get":
            if sys.stdout.isatty() and not args.no_confirm:
                ans = input(
                    f"About to print secret {args.name!r} to terminal. "
                    "Continue? [y/N]: "
                )
                if ans.strip().lower() not in ("y", "yes"):
                    print("Cancelled.")
                    return
            print(vault.get(args.name))
        elif sub == "list":
            names = vault.names()
            if not names:
                print("(empty)")
            else:
                for n in names:
                    print(n)
        elif sub == "remove":
            if vault.remove(args.name):
                print(f"Removed {args.name!r}.")
            else:
                print(f"No entry named {args.name!r}.")
        else:
            print(f"Unknown vault subcommand: {sub!r}")
    except VaultError as e:
        print(f"Vault error: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Not found: {e}")
        sys.exit(1)
    finally:
        # Drop the passphrase reference promptly.
        passphrase = None


async def _run_login(settings, args) -> None:
    """Wake the remote screen and type the login password."""
    from terminaleyes.commander.login import LoginFlow, resolve_password
    from terminaleyes.keyboard.http_backend import HttpKeyboardOutput
    from terminaleyes.mouse.http_backend import HttpMouseOutput

    cfg = settings.commander

    # Resolve password BEFORE printing config — we want any error
    # (missing file / env var / vault entry) to surface before
    # bringing up the BT transport.
    password = resolve_password(
        file_path=args.password_file,
        env_var=args.password_env,
        vault_name=args.vault,
    )
    if not password:
        print("Refusing to send empty password.")
        return
    print(
        f"Login: Pi={cfg.pi_base_url} (transport={cfg.transport}), "
        f"password length={len(password)}"
    )

    keyboard = HttpKeyboardOutput(
        base_url=cfg.pi_base_url,
        timeout=10.0,
        transport=cfg.transport,
    )
    mouse = HttpMouseOutput(
        base_url=cfg.pi_base_url,
        timeout=10.0,
        transport=cfg.transport,
    )
    await keyboard.connect()
    await mouse.connect()

    session = None
    needs_session = args.click_input or not args.no_verify
    if needs_session:
        # Build a full InteractiveSession so we can run the visual
        # login-screen check and (optionally) the input-field click.
        from terminaleyes.capture.webcam import WebcamCapture
        from terminaleyes.commander.evaluator import ConditionEvaluator
        from terminaleyes.commander.executor import ActionExecutor
        from terminaleyes.commander.interactive import InteractiveSession

        resolution = None
        if (settings.capture.resolution_width
                and settings.capture.resolution_height):
            resolution = (
                settings.capture.resolution_width,
                settings.capture.resolution_height,
            )
        capture = WebcamCapture(
            device_index=settings.capture.device_index,
            resolution=resolution,
        )
        await capture.open()
        evaluator = ConditionEvaluator(
            model=cfg.lmstudio_model,
            base_url=cfg.lmstudio_base_url,
            max_tokens=cfg.lmstudio_max_tokens,
        )
        executor = ActionExecutor(
            keyboard=keyboard,
            mouse=mouse,
            screen_width=cfg.screen_width,
            screen_height=cfg.screen_height,
            capture=capture,
            evaluator=evaluator,
        )
        session = InteractiveSession(
            capture=capture,
            evaluator=evaluator,
            executor=executor,
            model=cfg.lmstudio_model,
            base_url=cfg.lmstudio_base_url,
            max_tokens=cfg.lmstudio_max_tokens,
            vision_model=cfg.lmstudio_vision_model,
            vision_base_url=cfg.vision_base_url,
            skip_screen_check=True,
        )
        await session._ensure_client()

    flow = LoginFlow(mouse=mouse, keyboard=keyboard, session=session)
    try:
        sent = await flow.login(
            password=password,
            wake=not args.no_wake,
            click_input=args.click_input,
            submit=not args.no_submit,
            verify=not args.no_verify,
            verify_attempts=args.verify_attempts,
            verify_interval=args.verify_interval,
        )
        if sent:
            print("Login command sent.")
        else:
            print("Login NOT sent (verification refused).")
    finally:
        # Drop the password reference promptly — Python won't zero
        # the string memory but it'll be gc-eligible.
        password = None
        await keyboard.disconnect()
        await mouse.disconnect()


async def _watch(settings, args) -> None:
    """Run the passive screen watcher."""
    import signal
    from terminaleyes.capture.webcam import WebcamCapture
    from terminaleyes.watcher.reader import ScreenReader
    from terminaleyes.watcher.memory import MemoryStore
    from terminaleyes.watcher.loop import WatchLoop

    interval = args.interval if args.interval is not None else settings.watch.capture_interval_minutes
    duration = args.duration if args.duration is not None else settings.watch.session_duration_hours

    print("=" * 60)
    print("SCREEN WATCHER")
    print("=" * 60)
    print(f"Capture interval: {interval} min")
    print(f"Session duration: {duration} hr")
    print(f"Change threshold: {settings.watch.change_threshold}")
    print()
    print("NOTE: This tool captures and reads your screen via webcam.")
    print("Press Ctrl+C to stop early and generate summary.")
    print("=" * 60)
    print()

    # Build webcam capture (no crop — watch arbitrary screens)
    resolution = None
    if settings.capture.resolution_width and settings.capture.resolution_height:
        resolution = (settings.capture.resolution_width, settings.capture.resolution_height)

    capture = WebcamCapture(
        device_index=settings.capture.device_index,
        resolution=resolution,
    )

    # Build screen reader
    api_key = settings.openrouter_api_key.get_secret_value() or settings.openai_api_key.get_secret_value()
    base_url = settings.mllm.base_url
    if not base_url and settings.openrouter_api_key.get_secret_value():
        base_url = "https://openrouter.ai/api/v1"

    model = settings.watch.model_override or settings.mllm.model

    reader = ScreenReader(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=settings.mllm.max_tokens,
    )

    memory = MemoryStore()
    loop = WatchLoop(
        capture=capture,
        reader=reader,
        memory=memory,
        capture_interval_minutes=interval,
        session_duration_hours=duration,
        change_threshold=settings.watch.change_threshold,
    )

    # Handle Ctrl+C gracefully
    def _sigint_handler(sig, frame):
        print("\nStopping watch (generating summary)...")
        loop.stop()

    signal.signal(signal.SIGINT, _sigint_handler)

    session = await loop.run()

    print()
    print("=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Duration:  {session.duration_minutes:.1f} min")
    print(f"Captures:  {session.total_captures}")
    print(f"Changes:   {session.changes_detected}")
    print()
    print(session.final_summary)
    print("=" * 60)

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(session.model_dump(mode="json"), f, indent=2, default=str)
        print(f"\nSession saved to {args.output}")


async def _calibrate(settings, save: bool = True) -> None:
    """Auto-detect terminal position in webcam feed."""
    from terminaleyes.calibration import calibrate, apply_calibration_to_config

    print("Starting calibration...")
    print("The screen will flash WHITE and BLACK. Keep the camera steady.\n")

    resolution = None
    if settings.capture.resolution_width and settings.capture.resolution_height:
        resolution = (settings.capture.resolution_width, settings.capture.resolution_height)

    result = await calibrate(
        device_index=settings.capture.device_index,
        fullscreen=settings.endpoint.fullscreen,
        resolution=resolution,
    )

    print(f"\nDetected terminal region:")
    print(f"  Position: ({result['crop_x']}, {result['crop_y']})")
    print(f"  Size:     {result['crop_width']}x{result['crop_height']}")
    print(f"  Frame:    {result['frame_width']}x{result['frame_height']}")
    print(f"  Coverage: {result['area_ratio'] * 100:.0f}% of camera frame")
    print(f"\nDebug images saved: calibration_*.png")

    if save:
        config_path = "config/terminaleyes.yaml"
        apply_calibration_to_config(config_path, result)
        print(f"\nCrop settings saved to {config_path}")
        print("The agent will now auto-crop to the terminal region.")
    else:
        print("\nTo apply manually, add to config/terminaleyes.yaml:")
        print(f"  capture:")
        print(f"    crop_enabled: true")
        print(f"    crop_x: {result['crop_x']}")
        print(f"    crop_y: {result['crop_y']}")
        print(f"    crop_width: {result['crop_width']}")
        print(f"    crop_height: {result['crop_height']}")


if __name__ == "__main__":
    main()
