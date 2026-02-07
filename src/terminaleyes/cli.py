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
        )
        uvicorn.run(
            app,
            host=ep.host,
            port=ep.port,
        )

    elif args.command == "capture-test":
        logger.info("Running capture test")
        asyncio.run(_capture_test(settings))


if __name__ == "__main__":
    main()
