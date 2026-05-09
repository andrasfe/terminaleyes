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


async def _build_agent_context(settings, *, with_capture: bool = True):
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

    ctx = AgentContext(
        mouse=mouse,
        keyboard=keyboard,
        capture=capture,
        vision_client=client,
        vision_model=cfg.lmstudio_model,
        evaluator=evaluator,
    )
    return ctx, keyboard, mouse, capture


async def _run_controller(settings, args) -> None:
    """Run the ControllerAgent on a free-form intent."""
    from terminaleyes.agents.controller import ControllerAgent

    ctx, keyboard, mouse, capture = await _build_agent_context(
        settings, with_capture=True,
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


async def _run_focus(settings, args) -> None:
    """Run the FocusAgent end-to-end."""
    from terminaleyes.agents.focus import FocusAgent

    ctx, keyboard, mouse, capture = await _build_agent_context(
        settings, with_capture=True,
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
