# examples/desktop_agent/desktop_tools.py

"""
Desktop Automation Tools
=========================
Custom BaseTool subclasses for screen interaction.
These register directly into SumoSpace's ToolRegistry.

Tools:
  take_screenshot — capture screen to PNG, return path
  read_screen     — OCR the screen (or a region) to text
  click_at        — click at x,y coordinates
  double_click    — double-click at x,y
  right_click     — right-click at x,y
  type_text       — type a string via keyboard
  hotkey          — press keyboard shortcuts (e.g. ctrl+s)
  move_mouse      — move mouse to x,y
  open_app        — open an application by name
  wait            — pause for N seconds
  scroll          — scroll up/down
  drag_to         — click-drag from one point to another
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# SumoSpace tool base
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sumospace.tools import BaseTool, ToolResult


# ─── Screenshot Tool ─────────────────────────────────────────────────────────

class ScreenshotTool(BaseTool):
    """Capture the full screen or a region to a PNG file."""
    name = "take_screenshot"
    description = "Take a screenshot of the entire screen. Returns the file path."

    def __init__(self, output_dir: str = "/tmp/sumo_screenshots"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    async def run(self, region: str = "", filename: str = "", **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            self._counter += 1
            fname = filename or f"step_{self._counter:03d}.png"
            path = self._output_dir / fname

            if region:
                # region = "x,y,w,h"
                parts = [int(x.strip()) for x in region.split(",")]
                img = pyautogui.screenshot(region=tuple(parts))
            else:
                img = pyautogui.screenshot()

            img.save(str(path))
            w, h = img.size
            return ToolResult(
                tool=self.name, success=True,
                output=f"Screenshot saved: {path} ({w}x{h})",
                metadata={"path": str(path), "width": w, "height": h,
                          "step": self._counter},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


# ─── OCR / Screen Reader Tool ────────────────────────────────────────────────

class ReadScreenTool(BaseTool):
    """Read text from the screen using OCR (pytesseract)."""
    name = "read_screen"
    description = (
        "Read all visible text on screen using OCR. "
        "Optionally specify a region as 'x,y,w,h'."
    )

    async def run(self, region: str = "", image_path: str = "", **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            from PIL import Image

            if image_path and Path(image_path).exists():
                img = Image.open(image_path)
            elif region:
                parts = [int(x.strip()) for x in region.split(",")]
                img = pyautogui.screenshot(region=tuple(parts))
            else:
                img = pyautogui.screenshot()

            try:
                import pytesseract
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(None, pytesseract.image_to_string, img)
            except ImportError:
                # Fallback: describe image dimensions
                text = (
                    f"[OCR unavailable — install pytesseract] "
                    f"Screen captured: {img.size[0]}x{img.size[1]} pixels"
                )

            return ToolResult(
                tool=self.name, success=True,
                output=text.strip() or "[No text detected on screen]",
                metadata={"chars": len(text)},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


# ─── Click Tools ─────────────────────────────────────────────────────────────

class ClickTool(BaseTool):
    """Click at screen coordinates."""
    name = "click_at"
    description = "Left-click at the given x,y screen coordinates."

    async def run(self, x: int = 0, y: int = 0, **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            pyautogui.click(int(x), int(y))
            await asyncio.sleep(0.3)
            return ToolResult(
                tool=self.name, success=True,
                output=f"Clicked at ({x}, {y})",
                metadata={"x": x, "y": y},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class DoubleClickTool(BaseTool):
    """Double-click at screen coordinates."""
    name = "double_click"
    description = "Double-click at the given x,y screen coordinates."

    async def run(self, x: int = 0, y: int = 0, **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            pyautogui.doubleClick(int(x), int(y))
            await asyncio.sleep(0.3)
            return ToolResult(
                tool=self.name, success=True,
                output=f"Double-clicked at ({x}, {y})",
                metadata={"x": x, "y": y},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class RightClickTool(BaseTool):
    """Right-click at screen coordinates."""
    name = "right_click"
    description = "Right-click at the given x,y screen coordinates."

    async def run(self, x: int = 0, y: int = 0, **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            pyautogui.rightClick(int(x), int(y))
            await asyncio.sleep(0.3)
            return ToolResult(
                tool=self.name, success=True,
                output=f"Right-clicked at ({x}, {y})",
                metadata={"x": x, "y": y},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


# ─── Keyboard Tools ──────────────────────────────────────────────────────────

class TypeTextTool(BaseTool):
    """Type text using the keyboard."""
    name = "type_text"
    description = "Type the given text string using keyboard input."

    async def run(self, text: str = "", interval: float = 0.02, **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            pyautogui.write(text, interval=float(interval))
            await asyncio.sleep(0.2)
            return ToolResult(
                tool=self.name, success=True,
                output=f"Typed: '{text[:80]}{'...' if len(text) > 80 else ''}'",
                metadata={"length": len(text)},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class HotkeyTool(BaseTool):
    """Press a keyboard shortcut."""
    name = "hotkey"
    description = (
        "Press a keyboard shortcut. Pass keys as comma-separated string, "
        "e.g. 'ctrl,s' or 'alt,f4' or 'super' (Windows/Super key)."
    )

    async def run(self, keys: str = "", **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            key_list = [k.strip() for k in keys.split(",")]
            pyautogui.hotkey(*key_list)
            await asyncio.sleep(0.5)
            return ToolResult(
                tool=self.name, success=True,
                output=f"Pressed hotkey: {'+'.join(key_list)}",
                metadata={"keys": key_list},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class PressKeyTool(BaseTool):
    """Press a single key."""
    name = "press_key"
    description = (
        "Press a single key. Examples: 'enter', 'tab', 'escape', 'space', "
        "'backspace', 'delete', 'up', 'down', 'left', 'right', 'f1'...'f12'."
    )

    async def run(self, key: str = "", **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            pyautogui.press(key.strip())
            await asyncio.sleep(0.3)
            return ToolResult(
                tool=self.name, success=True,
                output=f"Pressed key: {key}",
                metadata={"key": key},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


# ─── Mouse Movement ──────────────────────────────────────────────────────────

class MoveMouseTool(BaseTool):
    """Move the mouse cursor to coordinates."""
    name = "move_mouse"
    description = "Move the mouse cursor to x,y coordinates."

    async def run(self, x: int = 0, y: int = 0, duration: float = 0.3, **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            pyautogui.moveTo(int(x), int(y), duration=float(duration))
            return ToolResult(
                tool=self.name, success=True,
                output=f"Mouse moved to ({x}, {y})",
                metadata={"x": x, "y": y},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class ScrollTool(BaseTool):
    """Scroll the mouse wheel."""
    name = "scroll"
    description = "Scroll up (positive) or down (negative). E.g. clicks=3 scrolls up, clicks=-3 scrolls down."

    async def run(self, clicks: int = 3, x: int = 0, y: int = 0, **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            if x and y:
                pyautogui.scroll(int(clicks), int(x), int(y))
            else:
                pyautogui.scroll(int(clicks))
            await asyncio.sleep(0.3)
            direction = "up" if int(clicks) > 0 else "down"
            return ToolResult(
                tool=self.name, success=True,
                output=f"Scrolled {direction} by {abs(int(clicks))} clicks",
                metadata={"clicks": clicks},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


class DragToTool(BaseTool):
    """Click and drag from current position to target."""
    name = "drag_to"
    description = "Click-drag from current mouse position to x,y coordinates."

    async def run(self, x: int = 0, y: int = 0, duration: float = 0.5, **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            pyautogui.dragTo(int(x), int(y), duration=float(duration))
            await asyncio.sleep(0.3)
            return ToolResult(
                tool=self.name, success=True,
                output=f"Dragged to ({x}, {y})",
                metadata={"x": x, "y": y},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


# ─── Application Launcher ────────────────────────────────────────────────────

class OpenAppTool(BaseTool):
    """Open an application by name or command."""
    name = "open_app"
    description = (
        "Open a desktop application. Provide the app name (e.g. 'code', "
        "'firefox', 'nautilus', 'terminal', 'gedit') or a full command."
    )

    # Common app name → launch command mapping
    APP_MAP = {
        "vscode": "code",
        "vs code": "code",
        "code": "code",
        "firefox": "firefox",
        "chrome": "google-chrome",
        "chromium": "chromium-browser",
        "terminal": "gnome-terminal",
        "files": "nautilus",
        "file manager": "nautilus",
        "nautilus": "nautilus",
        "gedit": "gedit",
        "text editor": "gedit",
        "calculator": "gnome-calculator",
        "settings": "gnome-control-center",
        "gimp": "gimp",
        "libreoffice": "libreoffice",
    }

    async def run(self, app: str = "", args: str = "", **_) -> ToolResult:
        start = time.monotonic()
        try:
            cmd = self.APP_MAP.get(app.lower().strip(), app.strip())
            if args:
                cmd = f"{cmd} {args}"

            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # Give the app a moment to launch
            await asyncio.sleep(1.5)
            return ToolResult(
                tool=self.name, success=True,
                output=f"Launched: {cmd} (pid: {proc.pid})",
                metadata={"command": cmd, "pid": proc.pid},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


# ─── Wait Tool ────────────────────────────────────────────────────────────────

class WaitTool(BaseTool):
    """Pause execution for a specified duration."""
    name = "wait"
    description = "Wait for N seconds before the next action."

    async def run(self, seconds: float = 1.0, **_) -> ToolResult:
        start = time.monotonic()
        seconds = min(float(seconds), 10.0)  # cap at 10s
        await asyncio.sleep(seconds)
        return ToolResult(
            tool=self.name, success=True,
            output=f"Waited {seconds}s",
            metadata={"seconds": seconds},
            duration_ms=(time.monotonic() - start) * 1000,
        )


# ─── Locate On Screen ────────────────────────────────────────────────────────

class LocateOnScreenTool(BaseTool):
    """Find a UI element by searching for a template image on screen."""
    name = "locate_on_screen"
    description = (
        "Find a UI element by matching a template image on screen. "
        "Returns the center coordinates if found."
    )

    async def run(self, image_path: str = "", confidence: float = 0.8, **_) -> ToolResult:
        start = time.monotonic()
        try:
            import pyautogui
            location = pyautogui.locateOnScreen(
                image_path, confidence=float(confidence)
            )
            if location:
                center = pyautogui.center(location)
                return ToolResult(
                    tool=self.name, success=True,
                    output=f"Found at center ({center.x}, {center.y})",
                    metadata={"x": center.x, "y": center.y,
                              "region": str(location)},
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            else:
                return ToolResult(
                    tool=self.name, success=False,
                    output="Element not found on screen",
                    error="No match found",
                )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, output="", error=str(e))


# ─── Registry Helper ─────────────────────────────────────────────────────────

ALL_DESKTOP_TOOLS: list[type[BaseTool]] = [
    ScreenshotTool,
    ReadScreenTool,
    ClickTool,
    DoubleClickTool,
    RightClickTool,
    TypeTextTool,
    HotkeyTool,
    PressKeyTool,
    MoveMouseTool,
    ScrollTool,
    DragToTool,
    OpenAppTool,
    WaitTool,
    LocateOnScreenTool,
]


def register_desktop_tools(registry) -> list[str]:
    """Register all desktop tools into a SumoSpace ToolRegistry. Returns tool names."""
    names = []
    for cls in ALL_DESKTOP_TOOLS:
        tool = cls()
        registry.register(tool)
        names.append(tool.name)
    return names
