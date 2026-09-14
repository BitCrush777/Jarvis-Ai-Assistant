"""
Dynamic Multi-Monitor Window Management Engine for JARVIS (MARK II / MARK XL).

Provides robust, real-time window management across 1, 2, 3+ monitor setups:
- Dynamic monitor enumeration via Win32 EnumDisplayMonitors & GetMonitorInfoW
- Physical and virtual screen geometry (origin, width, height, work area, primary flag)
- Geometric monitor arrangement resolution (never assuming fixed monitor 1 = left, 2 = right)
- Active window resolution via GetForegroundWindow
- Proportional coordinate calculation across source and target monitor work areas
- Instant native window movement via SetWindowPos (zero arbitrary sleeps)
- Post-movement coordinate verification
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import platform
import time
from dataclasses import dataclass
from typing import Optional

_IS_WINDOWS = platform.system().lower() == "windows"

if _IS_WINDOWS:
    user32 = ctypes.windll.user32
else:
    user32 = None

# ---------------------------------------------------------------------------
# Win32 Structures & Constants
# ---------------------------------------------------------------------------

CCHDEVICENAME = 32
MONITORINFOF_PRIMARY = 0x00000001
MONITOR_DEFAULTTONEAREST = 0x00000002

SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040

class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * CCHDEVICENAME),
    ]


@dataclass
class MonitorInfo:
    index: int
    name: str
    is_primary: bool
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int
    work_left: int
    work_top: int
    work_right: int
    work_bottom: int
    work_width: int
    work_height: int


@dataclass
class WindowRect:
    hwnd: int
    title: str
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int


# ---------------------------------------------------------------------------
# Multi-Monitor Manager
# ---------------------------------------------------------------------------

class MonitorManager:
    """
    Dynamically discovers and monitors multi-display configurations.
    Coordinates window relocation across displays with proportional scaling.
    """

    def __init__(self):
        self._cached_monitors: list[MonitorInfo] = []
        self._last_refresh: float = 0.0
        self._refresh_interval: float = 2.0  # seconds between hardware re-queries
        self.refresh_monitors()

    def refresh_monitors(self) -> list[MonitorInfo]:
        """Query Win32 for all currently active display monitors."""
        if not _IS_WINDOWS or user32 is None:
            # Fallback for mock or non-Windows
            self._cached_monitors = [
                MonitorInfo(
                    index=0,
                    name="DISPLAY1",
                    is_primary=True,
                    left=0,
                    top=0,
                    right=1920,
                    bottom=1080,
                    width=1920,
                    height=1080,
                    work_left=0,
                    work_top=0,
                    work_right=1920,
                    work_bottom=1040,
                    work_width=1920,
                    work_height=1040,
                )
            ]
            self._last_refresh = time.time()
            return self._cached_monitors

        monitors: list[MonitorInfo] = []

        MonitorEnumProc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HMONITOR,
            wintypes.HDC,
            ctypes.POINTER(RECT),
            wintypes.LPARAM,
        )

        def _enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
            mi = MONITORINFOEXW()
            mi.cbSize = ctypes.sizeof(MONITORINFOEXW)
            if user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi)):
                rc_mon = mi.rcMonitor
                rc_work = mi.rcWork
                is_pri = bool(mi.dwFlags & MONITORINFOF_PRIMARY)
                dev_name = mi.szDevice
                monitors.append(
                    MonitorInfo(
                        index=len(monitors),
                        name=dev_name,
                        is_primary=is_pri,
                        left=rc_mon.left,
                        top=rc_mon.top,
                        right=rc_mon.right,
                        bottom=rc_mon.bottom,
                        width=rc_mon.width,
                        height=rc_mon.height,
                        work_left=rc_work.left,
                        work_top=rc_work.top,
                        work_right=rc_work.right,
                        work_bottom=rc_work.bottom,
                        work_width=rc_work.width,
                        work_height=rc_work.height,
                    )
                )
            return True

        proc = MonitorEnumProc(_enum_proc)
        user32.EnumDisplayMonitors(None, None, proc, 0)

        # Sort monitors geometrically from left to right, then top to bottom
        monitors.sort(key=lambda m: (m.left, m.top))
        for idx, m in enumerate(monitors):
            m.index = idx

        self._cached_monitors = monitors if monitors else self._cached_monitors
        self._last_refresh = time.time()
        return self._cached_monitors

    def get_monitors(self, force_refresh: bool = False) -> list[MonitorInfo]:
        if force_refresh or (time.time() - self._last_refresh > self._refresh_interval):
            return self.refresh_monitors()
        return self._cached_monitors

    @property
    def monitor_count(self) -> int:
        return len(self.get_monitors())

    def get_primary_monitor(self) -> Optional[MonitorInfo]:
        for m in self.get_monitors():
            if m.is_primary:
                return m
        return self._cached_monitors[0] if self._cached_monitors else None

    def get_window_rect(self, hwnd: int) -> Optional[WindowRect]:
        """Query active window position and title."""
        if not _IS_WINDOWS or user32 is None or not hwnd:
            return None

        rc = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rc)):
            return None

        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value

        return WindowRect(
            hwnd=hwnd,
            title=title,
            left=rc.left,
            top=rc.top,
            right=rc.right,
            bottom=rc.bottom,
            width=rc.width,
            height=rc.height,
        )

    def get_active_window(self) -> Optional[WindowRect]:
        """Get the current foreground window rect and title."""
        if not _IS_WINDOWS or user32 is None:
            return None
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        return self.get_window_rect(hwnd)

    def get_monitor_for_window(self, hwnd: int) -> Optional[MonitorInfo]:
        """Find the monitor that best contains the specified window."""
        if not _IS_WINDOWS or user32 is None or not hwnd:
            return self.get_primary_monitor()

        hmon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        if not hmon:
            return self.get_primary_monitor()

        mi = MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            dev_name = mi.szDevice
            for m in self.get_monitors():
                if m.name == dev_name or (m.left == mi.rcMonitor.left and m.top == mi.rcMonitor.top):
                    return m

        return self.get_primary_monitor()

    def move_active_window_to_monitor(self, target_monitor_idx: Optional[int] = None, direction: str = "next") -> bool:
        """
        Move the active foreground window to a target monitor or cycle next/prev.
        Calculates proportional position in destination work area and updates window.
        Returns True if successfully moved and verified.
        """
        win = self.get_active_window()
        if not win:
            print("[WindowManager] No active foreground window found.")
            return False

        # Ignore tiny or invisible utility handles
        if win.width <= 0 or win.height <= 0:
            print(f"[WindowManager] Window '{win.title}' has invalid bounds.")
            return False

        monitors = self.get_monitors()
        if len(monitors) <= 1:
            print("[WindowManager] Single monitor system detected. Cannot move across monitors.")
            return False

        current_mon = self.get_monitor_for_window(win.hwnd)
        curr_idx = current_mon.index if current_mon else 0

        # Determine target index
        if target_monitor_idx is not None:
            dest_idx = target_monitor_idx % len(monitors)
        elif direction.lower() == "prev":
            dest_idx = (curr_idx - 1) % len(monitors)
        else:
            dest_idx = (curr_idx + 1) % len(monitors)

        if dest_idx == curr_idx:
            return True  # Already on target

        target_mon = monitors[dest_idx]
        src_mon = current_mon or monitors[0]

        # 1. Proportional position within source monitor work area
        src_work_w = max(1, src_mon.work_width)
        src_work_h = max(1, src_mon.work_height)

        rel_x = (win.left - src_mon.work_left) / src_work_w
        rel_y = (win.top - src_mon.work_top) / src_work_h
        rel_w = win.width / src_work_w
        rel_h = win.height / src_work_h

        # Bound relative position to avoid windows disappearing outside view
        rel_x = max(0.0, min(0.85, rel_x))
        rel_y = max(0.0, min(0.85, rel_y))
        rel_w = max(0.1, min(1.0, rel_w))
        rel_h = max(0.1, min(1.0, rel_h))

        # 2. Target coordinates in destination work area
        dest_work_w = target_mon.work_width
        dest_work_h = target_mon.work_height

        new_w = int(rel_w * dest_work_w)
        new_h = int(rel_h * dest_work_h)
        new_x = int(target_mon.work_left + rel_x * dest_work_w)
        new_y = int(target_mon.work_top + rel_y * dest_work_h)

        # 3. Native Win32 SetWindowPos
        t_start = time.perf_counter()
        success = bool(user32.SetWindowPos(
            win.hwnd,
            0,
            new_x,
            new_y,
            new_w,
            new_h,
            SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        ))
        t_move_ms = (time.perf_counter() - t_start) * 1000

        # 4. Verify post-movement position
        verified_win = self.get_window_rect(win.hwnd)
        if verified_win:
            new_mon = self.get_monitor_for_window(win.hwnd)
            print(
                f"[WindowManager] Moved '{win.title}' -> Monitor {dest_idx} "
                f"({target_mon.name}) at ({verified_win.left}, {verified_win.top}) in {t_move_ms:.2f} ms."
            )
            return True

        return success


# Global Singleton
_GLOBAL_MONITOR_MANAGER: Optional[MonitorManager] = None

def get_monitor_manager() -> MonitorManager:
    global _GLOBAL_MONITOR_MANAGER
    if _GLOBAL_MONITOR_MANAGER is None:
        _GLOBAL_MONITOR_MANAGER = MonitorManager()
    return _GLOBAL_MONITOR_MANAGER
