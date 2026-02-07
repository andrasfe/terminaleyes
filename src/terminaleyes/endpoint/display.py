"""Terminal display renderer for the command endpoint.

Renders the shell's screen buffer in a window that looks like a real
terminal, so the webcam + MLLM can read it reliably. Uses pygame for
precise control over rendering (monospace font, colors, cursor).
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class TerminalDisplay:
    """Renders terminal content in a pygame window.

    Creates a window that visually resembles a terminal emulator with
    dark background, monospace font, and blinking cursor.

    The display runs in its own thread to avoid blocking the async
    event loop of the HTTP server.
    """

    def __init__(
        self,
        rows: int = 24,
        cols: int = 80,
        font_size: int = 24,
        bg_color: tuple[int, int, int] = (30, 30, 30),
        fg_color: tuple[int, int, int] = (192, 192, 192),
        window_title: str = "terminaleyes - Terminal",
        fullscreen: bool = False,
    ) -> None:
        self._rows = rows
        self._cols = cols
        self._font_size = font_size
        self._bg_color = bg_color
        self._fg_color = fg_color
        self._window_title = window_title
        self._fullscreen = fullscreen
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
        mode = "fullscreen" if self._fullscreen else f"{self._cols}x{self._rows}"
        logger.info("Terminal display started (%s)", mode)

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
        """Main pygame rendering loop running in its own thread."""
        import pygame

        pygame.init()

        padding = 20

        if self._fullscreen:
            # Get display info before creating the window
            info = pygame.display.Info()
            screen_w, screen_h = info.current_w, info.current_h
            screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)

            # Calculate font size to fill the screen
            # Try to fit rows x cols with padding
            usable_w = screen_w - padding * 2
            usable_h = screen_h - padding * 2

            # Find the largest font size where rows*line_height fits vertically
            # and cols*char_w fits horizontally
            best_size = self._font_size
            for test_size in range(8, 80):
                test_font = self._find_mono_font(pygame, test_size)
                cw, ch = test_font.size("M")
                lh = int(ch * 1.2)
                if cw * self._cols <= usable_w and lh * self._rows <= usable_h:
                    best_size = test_size
                else:
                    break

            font = self._find_mono_font(pygame, best_size)
            logger.info(
                "Fullscreen %dx%d, auto font size: %d",
                screen_w, screen_h, best_size,
            )
        else:
            font = self._find_mono_font(pygame, self._font_size)
            char_w, char_h = font.size("M")
            line_height = int(char_h * 1.2)
            win_w = self._cols * char_w + padding * 2
            win_h = self._rows * line_height + padding * 2
            screen = pygame.display.set_mode((win_w, win_h))

        pygame.display.set_caption(self._window_title)

        char_w, char_h = font.size("M")
        line_height = int(char_h * 1.2)

        # Center the text area in the window
        win_w, win_h = screen.get_size()
        text_block_w = self._cols * char_w
        text_block_h = self._rows * line_height
        offset_x = (win_w - text_block_w) // 2
        offset_y = (win_h - text_block_h) // 2

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
                # Allow Escape to exit fullscreen
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                    break

            screen.fill(self._bg_color)

            with self._lock:
                content = self._content

            lines = content.split("\n")

            for i, line in enumerate(lines[: self._rows]):
                truncated = line[: self._cols]
                if truncated:
                    surface = font.render(truncated, True, self._fg_color)
                    screen.blit(surface, (offset_x, offset_y + i * line_height))

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
                    offset_x + cursor_col * char_w,
                    offset_y + cursor_line * line_height,
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
