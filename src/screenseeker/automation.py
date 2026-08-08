"""Deterministic desktop automation: screenshot, click, type, save, close.

Everything here is keyboard/clipboard-driven and screen-position-independent,
EXCEPT clicking a grounded coordinate. Grounding is used only for clicks; typing
and saving deliberately use global shortcuts (Ctrl+S, Enter, Alt+F4) so they work
regardless of window position and never depend on pixel-perfect vision.

Guarded imports: pyautogui/mss require a display, so this module is import-safe in
headless CI; the functions raise clearly if called without a backend.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

import sys
if sys.platform.startswith("win"):
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

try:                                    # optional at import time
    import pyautogui
    import pyperclip
    import numpy as np
    from PIL import Image
    _HAVE_GUI = True
except Exception:                       # headless / sandbox
    _HAVE_GUI = False


def _require_gui() -> None:
    if not _HAVE_GUI:
        raise RuntimeError(
            "GUI backend unavailable (pyautogui/mss). Run on the Windows desktop "
            "with a display; the sandbox/CI cannot drive the screen."
        )


def screenshot():
    """Return the full desktop as a numpy RGB array (physical pixels).

    mss is tried first (fast); if it yields a degenerate/tiny frame we fall back
    to PIL ImageGrab, which reliably captures the full primary screen on Windows.
    DPI awareness is set at import so this matches PyAutoGUI's coordinate space.
    """
    _require_gui()
    try:
        import mss
        with mss.mss() as sct:
            raw = sct.grab(sct.monitors[1])          # primary monitor
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        arr = np.array(img)
        if arr.shape[0] > 100 and arr.shape[1] > 100:
            return arr
        log.warning("mss returned tiny frame %s; using ImageGrab", arr.shape)
    except Exception as e:
        log.warning("mss screenshot failed (%s); using ImageGrab", e)
    from PIL import ImageGrab
    return np.array(ImageGrab.grab().convert("RGB"))


def foreground_title() -> str:
    """Title of the currently focused window (Windows); '' elsewhere.

    Used to guarantee we only send Alt+F4 to Notepad -- never to the desktop,
    where Alt+F4 would pop the 'Shut Down Windows' dialog.
    """
    if not sys.platform.startswith("win"):
        return ""
    try:
        import ctypes
        u = ctypes.windll.user32
        hwnd = u.GetForegroundWindow()
        n = u.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value
    except Exception:
        return ""


def notepad_is_focused() -> bool:
    return "notepad" in foreground_title().lower()


def double_click(x: float, y: float) -> None:
    _require_gui()
    pyautogui.doubleClick(int(round(x)), int(round(y)))


def left_click(x: float, y: float) -> None:
    _require_gui()
    pyautogui.click(int(round(x)), int(round(y)))


def type_text(text: str, settle: float = 0.3) -> None:
    """Paste rather than keystroke: fast and exact for long bodies."""
    _require_gui()
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(settle)


def press(*keys: str) -> None:
    _require_gui()
    if len(keys) == 1:
        pyautogui.press(keys[0])
    else:
        pyautogui.hotkey(*keys)


def save_as(full_path: Path, settle: float = 0.4) -> None:
    """Save the current Notepad buffer to `full_path`, driven entirely by the dialog
    window TITLE — no vision, no VLM pop-up handler.

    The Save As dialog and the overwrite confirmation ("Confirm Save As") are known
    Windows dialogs, so we handle them deterministically: wait for the Save As window,
    type the path, Enter; then if an overwrite/confirm dialog is still in front, press
    Enter (its default button is Yes). Treating these as "unknown pop-ups" and letting
    a model guess a button is what previously dismissed the dialog by mistake.
    """
    _require_gui()
    pyautogui.hotkey("ctrl", "s")

    # wait until the Save As dialog is actually the foreground window
    deadline = time.time() + 3.0
    while time.time() < deadline and "save as" not in foreground_title().lower():
        time.sleep(0.15)

    pyautogui.hotkey("ctrl", "a")          # select the default filename
    type_text(str(full_path))              # paste the full path over it
    pyautogui.press("enter")
    time.sleep(settle)

    # If a Save As / "Confirm Save As" (overwrite) dialog is still up, confirm it.
    for _ in range(3):
        if "save as" in foreground_title().lower():
            pyautogui.press("enter")       # default button = Save / Yes
            time.sleep(settle)
        else:
            break


def close_window(settle: float = 0.4, on_dialog: "callable | None" = None) -> None:
    """Close the focused window with Alt+F4 -- but ONLY if it is Notepad.

    If Notepad is not in front (e.g. grounding missed and it never launched),
    we refuse to send Alt+F4, because on the bare desktop that opens the
    'Shut Down Windows' dialog. Safety over completeness.
    """
    _require_gui()
    if sys.platform.startswith("win") and not notepad_is_focused():
        log.warning("close_window: Notepad not focused (front='%s'); skipping Alt+F4",
                    foreground_title())
        return
    pyautogui.hotkey("alt", "f4")
    time.sleep(settle)
    if on_dialog:
        on_dialog()
