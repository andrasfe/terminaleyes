"""Terminal display renderer for the command endpoint.

Renders the shell's screen buffer in a fullscreen window with a positioned
text rectangle that the webcam + MLLM can read reliably. Uses pygame for
precise control over rendering (monospace font, colors, cursor).

The display is fullscreen (guaranteed always-on-top on Wayland) with a black
background. Only the text area has the configured bg_color (white), making it
unobtrusive while keeping the terminal content visible to the camera.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class TerminalDisplay:
    """Renders terminal content in a fullscreen window with positioned text area.

    The display runs in its own thread to avoid blocking the async
    event loop of the HTTP server.
    """

    def __init__(
        self,
        rows: int = 24,
        cols: int = 80,
        font_size: int = 24,
        bg_color: tuple[int, int, int] = (255, 255, 255),
        fg_color: tuple[int, int, int] = (0, 0, 0),
        window_title: str = "terminaleyes - Terminal",
        fullscreen: bool = True,
        window_x: int | None = None,
        window_y: int | None = None,
    ) -> None:
        self._rows = rows
        self._cols = cols
        self._font_size = font_size
        self._bg_color = bg_color
        self._fg_color = fg_color
        self._window_title = window_title
        self._fullscreen = fullscreen
        self._window_x = window_x
        self._window_y = window_y
        self._content: str = ""
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the display window in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._render_loop, daemon=True, name="terminal-display"
        )
        self._thread.start()
        logger.info("Terminal display started")

    def stop(self) -> None:
        """Stop the display window."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Terminal display stopped")

    def update_content(self, content: str) -> None:
        """Update the displayed terminal content (thread-safe)."""
        with self._lock:
            self._content = content

    def _render_loop(self) -> None:
        """Main pygame rendering loop running in its own thread.

        Always uses fullscreen with black background. The text area
        (white rectangle) is positioned where the camera can see it,
        using window_x/window_y as the text area offset within the
        fullscreen surface.
        """
        import pygame

        pygame.init()

        info = pygame.display.Info()
        screen_w, screen_h = info.current_w, info.current_h
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)

        # Calculate font and text block size
        padding = 30
        font = self._find_mono_font(pygame, self._font_size)
        char_w, char_h = font.size("M")
        line_height = int(char_h * 1.2)
        text_block_w = self._cols * char_w
        text_block_h = self._rows * line_height
        rect_w = text_block_w + padding * 2
        rect_h = text_block_h + padding * 2

        # Position the text rectangle using window_x/window_y (from calibration)
        # or center on screen if not specified
        if self._window_x is not None and self._window_y is not None:
            rect_x = self._window_x
            rect_y = self._window_y
        else:
            rect_x = (screen_w - rect_w) // 2
            rect_y = (screen_h - rect_h) // 2

        # Text offset within the rectangle
        text_x = rect_x + padding
        text_y = rect_y + padding

        logger.info(
            "Fullscreen %dx%d, text area at (%d,%d) size %dx%d, font: %d",
            screen_w, screen_h, rect_x, rect_y, rect_w, rect_h, self._font_size,
        )

        pygame.display.set_caption(self._window_title)

        clock = pygame.time.Clock()
        cursor_visible = True
        cursor_timer = 0.0
        last_time = time.time()

        while self._running:
            now = time.time()
            dt = now - last_time
            last_time = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                    break

            # Black background (unobtrusive)
            screen.fill((0, 0, 0))

            # White text rectangle where camera can see it
            text_rect = pygame.Rect(rect_x, rect_y, rect_w, rect_h)
            pygame.draw.rect(screen, self._bg_color, text_rect)

            with self._lock:
                content = self._content

            lines = content.split("\n")

            for i, line in enumerate(lines[: self._rows]):
                truncated = line[: self._cols]
                if truncated:
                    surface = font.render(truncated, True, self._fg_color)
                    screen.blit(surface, (text_x, text_y + i * line_height))

            # Blinking cursor
            cursor_timer += dt
            if cursor_timer >= 0.5:
                cursor_visible = not cursor_visible
                cursor_timer = 0.0

            if cursor_visible:
                cursor_line = min(len(lines) - 1, self._rows - 1) if lines else 0
                cursor_col = len(lines[cursor_line]) if cursor_line < len(lines) else 0
                cursor_col = min(cursor_col, self._cols - 1)
                cursor_rect = pygame.Rect(
                    text_x + cursor_col * char_w,
                    text_y + cursor_line * line_height,
                    char_w,
                    line_height,
                )
                pygame.draw.rect(screen, self._fg_color, cursor_rect)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    @staticmethod
    def _find_mono_font(pygame, size: int):
        """Find a monospace font at the given size."""
        for name in ["dejavusansmono", "liberationmono", "couriernew", "monospace", "courier"]:
            path = pygame.font.match_font(name)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.SysFont("monospace", size)
