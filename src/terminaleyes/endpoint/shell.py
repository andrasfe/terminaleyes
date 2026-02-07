"""Persistent shell session for the command endpoint.

Manages a long-running shell process using a pseudo-terminal (pty)
for realistic terminal emulation, maintaining state across commands.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import pty
import signal
import struct
import termios

logger = logging.getLogger(__name__)


class PersistentShell:
    """Manages a persistent interactive shell subprocess via pty.

    Uses a pseudo-terminal for realistic terminal behavior including
    proper line editing, signal handling, and ANSI escape support.
    """

    def __init__(
        self,
        shell_command: str = "/bin/bash",
        rows: int = 24,
        cols: int = 80,
        scrollback_lines: int = 1000,
    ) -> None:
        self._shell_command = shell_command
        self._rows = rows
        self._cols = cols
        self._scrollback_lines = scrollback_lines
        self._process: asyncio.subprocess.Process | None = None
        self._screen_buffer: list[str] = []
        self._is_alive = False
        self._read_task: asyncio.Task[None] | None = None
        self._master_fd: int | None = None
        self._slave_fd: int | None = None
        self._pid: int | None = None

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    async def start(self) -> None:
        """Start the shell subprocess using a pty."""
        master_fd, slave_fd = pty.openpty()
        self._master_fd = master_fd
        self._slave_fd = slave_fd

        # Set terminal size
        winsize = struct.pack("HHHH", self._rows, self._cols, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["COLUMNS"] = str(self._cols)
        env["LINES"] = str(self._rows)

        pid = os.fork()
        if pid == 0:
            # Child process
            os.close(master_fd)
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
            os.dup2(slave_fd, 0)
            os.dup2(slave_fd, 1)
            os.dup2(slave_fd, 2)
            if slave_fd > 2:
                os.close(slave_fd)
            os.execvpe(self._shell_command, [self._shell_command], env)
        else:
            # Parent process
            os.close(slave_fd)
            self._slave_fd = None
            self._pid = pid

            # Make master_fd non-blocking
            flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
            fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            self._is_alive = True
            self._read_task = asyncio.create_task(self._read_output_loop())
            logger.info(
                "Started shell %s (pid=%d, %dx%d)",
                self._shell_command, pid, self._cols, self._rows,
            )

    async def stop(self) -> None:
        """Stop the shell subprocess gracefully."""
        if self._read_task is not None:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self._pid is not None:
            try:
                os.kill(self._pid, signal.SIGTERM)
                await asyncio.sleep(0.5)
                try:
                    os.waitpid(self._pid, os.WNOHANG)
                except ChildProcessError:
                    pass
                # Check if still alive
                try:
                    os.kill(self._pid, 0)
                    os.kill(self._pid, signal.SIGKILL)
                    os.waitpid(self._pid, 0)
                except (ProcessLookupError, ChildProcessError):
                    pass
            except ProcessLookupError:
                pass
            self._pid = None

        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

        self._is_alive = False
        logger.info("Shell stopped")

    async def send_input(self, data: str) -> None:
        """Send input data to the shell's stdin via pty."""
        if not self._is_alive or self._master_fd is None:
            raise ShellError("Shell is not alive")
        try:
            os.write(self._master_fd, data.encode())
        except OSError as e:
            raise ShellError(f"Failed to write to shell: {e}") from e

    async def send_signal(self, signal_name: str) -> None:
        """Send a signal or control character to the shell."""
        if not self._is_alive or self._master_fd is None:
            raise ShellError("Shell is not alive")

        control_chars = {
            "SIGINT": b"\x03",     # Ctrl+C
            "SIGTSTP": b"\x1a",    # Ctrl+Z
            "EOF": b"\x04",        # Ctrl+D
        }
        char = control_chars.get(signal_name.upper())
        if char:
            os.write(self._master_fd, char)
            logger.debug("Sent %s to shell", signal_name)
        else:
            raise ShellError(f"Unknown signal: {signal_name}")

    def get_screen_content(self) -> str:
        """Get the current terminal screen content."""
        visible = self._screen_buffer[-self._rows:]
        # Pad with empty lines if fewer than rows
        while len(visible) < self._rows:
            visible.insert(0, "")
        # Truncate lines to cols
        visible = [line[:self._cols] for line in visible]
        return "\n".join(visible)

    async def _read_output_loop(self) -> None:
        """Background task that reads shell output and updates the buffer."""
        loop = asyncio.get_event_loop()
        partial_line = ""
        while self._is_alive and self._master_fd is not None:
            try:
                data = await loop.run_in_executor(
                    None, self._read_master
                )
                if data is None:
                    await asyncio.sleep(0.05)
                    continue
                text = partial_line + data
                partial_line = ""
                # Strip ANSI escape sequences for cleaner display
                import re
                # CSI sequences (e.g., colors, cursor movement)
                text = re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', text)
                # OSC sequences (e.g., window title)
                text = re.sub(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)', '', text)
                # Other escape sequences
                text = re.sub(r'\x1b[()][AB012]', '', text)
                text = re.sub(r'\x1b[>=]', '', text)
                # Control characters except newline/tab
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
                text = text.replace('\r\n', '\n').replace('\r', '\n')
                lines = text.split("\n")
                if not text.endswith("\n"):
                    partial_line = lines.pop()
                for line in lines:
                    self._screen_buffer.append(line)
                # Keep partial line in buffer for display
                if partial_line:
                    self._screen_buffer.append(partial_line)
                # Trim scrollback
                if len(self._screen_buffer) > self._scrollback_lines:
                    self._screen_buffer = self._screen_buffer[-self._scrollback_lines:]
                logger.debug("Shell output: %d lines in buffer", len(self._screen_buffer))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Read loop error: %s", e)
                await asyncio.sleep(0.1)

    def _read_master(self) -> str | None:
        """Read from the master pty fd (blocking call, run in executor)."""
        import select
        try:
            r, _, _ = select.select([self._master_fd], [], [], 0.1)
            if r:
                data = os.read(self._master_fd, 4096)
                if data:
                    return data.decode("utf-8", errors="replace")
        except (OSError, ValueError):
            return None
        return None


class ShellError(Exception):
    """Raised when shell operations fail."""
