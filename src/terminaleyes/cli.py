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
    capture = WebcamCapture(
        device_index=settings.capture.device_index, crop_region=crop,
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
        app = create_app(
            shell_command=ep.shell_command,
            rows=ep.terminal_rows,
            cols=ep.terminal_cols,
            font_size=ep.font_size,
            bg_color=ep.bg_color,
            fg_color=ep.fg_color,
            fullscreen=ep.fullscreen,
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

    elif args.command == "calibrate":
        logger.info("Running camera calibration")
        asyncio.run(_calibrate(settings, save=not args.no_save))


async def _calibrate(settings, save: bool = True) -> None:
    """Auto-detect terminal position in webcam feed."""
    from terminaleyes.calibration import calibrate, apply_calibration_to_config

    print("Starting calibration...")
    print("The screen will flash WHITE and BLACK. Keep the camera steady.\n")

    result = await calibrate(
        device_index=settings.capture.device_index,
        fullscreen=settings.endpoint.fullscreen,
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
