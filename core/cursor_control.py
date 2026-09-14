"""
Vision-Based Cursor Control Engine for JARVIS (MARK II / MARK XL).

Provides ultra-smooth, responsive, and stable mouse cursor control via index fingertip tracking:
- Strict mode gating (POINTING_MODE active only on explicit gesture activation)
- Multi-monitor virtual screen boundary projection (SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN)
- Camera horizontal mirroring and active interaction box scaling
- Dual-layer jitter suppression (One Euro Filter + sub-pixel dead zone)
- Anti-jump latching on hand entrance, exit, and occlusion
- Natural pinch-to-click vs hold-to-drag state machine with threshold hysteresis
- Guaranteed mouse button release on hand loss, mode exit, or error
- Ultra-low latency native OS dispatch via ctypes.windll.user32
"""
from __future__ import annotations

import ctypes
import json
import math
import platform
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

_IS_WINDOWS = platform.system().lower() == "windows"
if _IS_WINDOWS:
    _user32 = ctypes.windll.user32
else:
    _user32 = None

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004


@dataclass
class ScreenBounds:
    origin_x: int
    origin_y: int
    width: int
    height: int


class PinchState(str, Enum):
    RELEASED = "RELEASED"
    PINCHING = "PINCHING"
    DRAGGING = "DRAGGING"


def get_virtual_screen_bounds() -> ScreenBounds:
    """
    Query Windows virtual screen boundaries to support multi-monitor setups.
    Returns ScreenBounds spanning all connected active monitors.
    """
    if _IS_WINDOWS and _user32:
        try:
            vx = int(_user32.GetSystemMetrics(76))  # SM_XVIRTUALSCREEN
            vy = int(_user32.GetSystemMetrics(77))  # SM_YVIRTUALSCREEN
            vw = int(_user32.GetSystemMetrics(78))  # SM_CXVIRTUALSCREEN
            vh = int(_user32.GetSystemMetrics(79))  # SM_CYVIRTUALSCREEN
            if vw > 0 and vh > 0:
                return ScreenBounds(origin_x=vx, origin_y=vy, width=vw, height=vh)
        except Exception:
            pass

    # Fallback to primary screen
    try:
        import pyautogui
        w, h = pyautogui.size()
        return ScreenBounds(origin_x=0, origin_y=0, width=int(w), height=int(h))
    except Exception:
        return ScreenBounds(origin_x=0, origin_y=0, width=1920, height=1080)


def _set_os_cursor_pos(x: int, y: int) -> None:
    """Move OS cursor instantly with zero latency."""
    if _IS_WINDOWS and _user32:
        _user32.SetCursorPos(int(x), int(y))
    else:
        try:
            import pyautogui
            pyautogui.moveTo(x, y)
        except Exception:
            pass


def _send_mouse_down() -> None:
    if _IS_WINDOWS and _user32:
        _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    else:
        try:
            import pyautogui
            pyautogui.mouseDown()
        except Exception:
            pass


def _send_mouse_up() -> None:
    if _IS_WINDOWS and _user32:
        _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    else:
        try:
            import pyautogui
            pyautogui.mouseUp()
        except Exception:
            pass


def _send_mouse_click() -> None:
    if _IS_WINDOWS and _user32:
        _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    else:
        try:
            import pyautogui
            pyautogui.click()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# One Euro Filter for Cursor Smoothing
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """
    Adaptive low-pass filter specifically tuned for cursor motion.
    Provides jitter-free hover stability with instant response during rapid motion.
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.2,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def _exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1.0 - a) * x_prev

    def filter(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return self.x_prev

        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# ---------------------------------------------------------------------------
# Vision Cursor Controller Core
# ---------------------------------------------------------------------------

class VisionCursorController:
    """
    Coordinates camera tracking input into stabilized desktop cursor motion,
    pinch-clicking, and drag-and-drop actions.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        screen_bounds: Optional[ScreenBounds] = None,
    ):
        self._cfg = config or self._load_config()
        self.enabled = bool(self._cfg.get("enabled", True))
        self.mirror_camera = bool(self._cfg.get("mirror_camera", True))
        self.sensitivity_x = float(self._cfg.get("sensitivity_x", 1.4))
        self.sensitivity_y = float(self._cfg.get("sensitivity_y", 1.4))
        self.dead_zone_px = float(self._cfg.get("dead_zone_px", 2.5))
        self.min_cutoff = float(self._cfg.get("min_cutoff", 1.2))
        self.beta = float(self._cfg.get("beta", 0.05))

        box = self._cfg.get("active_box", [0.15, 0.85, 0.20, 0.80])
        self.active_box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

        self.drag_hold_threshold = float(self._cfg.get("drag_hold_threshold", 0.25))
        self.pinch_click_distance = float(self._cfg.get("pinch_click_distance", 0.040))
        self.pinch_release_distance = self.pinch_click_distance + 0.015

        self.screen_bounds = screen_bounds or get_virtual_screen_bounds()
        self._lock = threading.Lock()

        # Mode Gating (POINTING_MODE)
        self._mode_active = False

        # One Euro Filters for X and Y screen coordinates
        self._filter_x: Optional[OneEuroFilter] = None
        self._filter_y: Optional[OneEuroFilter] = None

        # Position tracking & Dead Zone
        self._last_screen_x: float = 0.0
        self._last_screen_y: float = 0.0
        self._has_previous_pos: bool = False

        # Pinch & Drag state machine
        self._pinch_state = PinchState.RELEASED
        self._pinch_start_time: float = 0.0
        self._pinch_start_pos: tuple[float, float] = (0.0, 0.0)

        # Telemetry & Status Callbacks
        self._status_listeners: list[Callable[[str], None]] = []
        self._latest_telemetry: str = "CURSOR: IDLE"

    @staticmethod
    def _load_config() -> dict:
        try:
            if CONFIG_PATH.exists():
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                return data.get("cursor_control", {})
        except Exception:
            pass
        return {}

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._mode_active

    @property
    def pinch_state(self) -> PinchState:
        with self._lock:
            return self._pinch_state

    @property
    def latest_telemetry(self) -> str:
        with self._lock:
            return self._latest_telemetry

    def add_status_listener(self, listener: Callable[[str], None]) -> None:
        with self._lock:
            if listener not in self._status_listeners:
                self._status_listeners.append(listener)

    def remove_status_listener(self, listener: Callable[[str], None]) -> None:
        with self._lock:
            if listener in self._status_listeners:
                self._status_listeners.remove(listener)

    def _notify_status(self, text: str) -> None:
        self._latest_telemetry = text
        listeners = list(self._status_listeners)
        for cb in listeners:
            try:
                cb(text)
            except Exception:
                pass

    def enable_pointing_mode(self) -> None:
        """Activate POINTING_MODE to allow fingertip cursor control."""
        with self._lock:
            if not self._mode_active:
                self._mode_active = True
                self._has_previous_pos = False
                self._filter_x = None
                self._filter_y = None
                self._pinch_state = PinchState.RELEASED
                self._notify_status("CURSOR: POINTING MODE ACTIVE")
                print("[CursorControl] POINTING_MODE activated.")

    def disable_pointing_mode(self) -> None:
        """Deactivate POINTING_MODE and release any held mouse buttons."""
        with self._lock:
            if self._mode_active:
                if self._pinch_state == PinchState.DRAGGING:
                    try:
                        _send_mouse_up()
                    except Exception:
                        pass
                self._mode_active = False
                self._has_previous_pos = False
                self._pinch_state = PinchState.RELEASED
                self._notify_status("CURSOR: IDLE")
                print("[CursorControl] POINTING_MODE deactivated.")

    def reset(self) -> None:
        """Reset internal filters and state."""
        self.disable_pointing_mode()

    # -----------------------------------------------------------------------
    # Coordinate Mapping & Calibration
    # -----------------------------------------------------------------------

    def map_camera_to_screen(self, cam_x: float, cam_y: float) -> tuple[float, float]:
        """
        Transform normalized camera landmark coordinates into virtual screen pixels.
        Includes horizontal mirroring, active interaction box scaling, and sensitivity.
        """
        # 1. Camera horizontal mirroring
        x_mirrored = (1.0 - cam_x) if self.mirror_camera else cam_x
        y_cam = cam_y

        # 2. Scale within active interaction box
        x_min, x_max, y_min, y_max = self.active_box
        box_w = max(1e-4, x_max - x_min)
        box_h = max(1e-4, y_max - y_min)

        u = (x_mirrored - x_min) / box_w
        v = (y_cam - y_min) / box_h

        # 3. Apply sensitivity centered around midpoint (0.5)
        u = 0.5 + (u - 0.5) * self.sensitivity_x
        v = 0.5 + (v - 0.5) * self.sensitivity_y

        # Clamp normalized screen space [0.0, 1.0]
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        # 4. Project into Virtual Screen boundaries
        sb = self.screen_bounds
        screen_x = sb.origin_x + u * sb.width
        screen_y = sb.origin_y + v * sb.height

        return screen_x, screen_y

    # -----------------------------------------------------------------------
    # Main Per-Frame Update Pipeline
    # -----------------------------------------------------------------------

    def update(self, hand, timestamp: Optional[float] = None) -> Optional[tuple[int, int]]:
        """
        Process HandInfo frame. If POINTING_MODE is active and hand is pointing,
        calculates stabilized position, evaluates pinch/drag, and updates OS cursor.
        Guarantees mouse button release under any loss of tracking.
        """
        t_now = timestamp or time.time()

        with self._lock:
            if not self.enabled or not self._mode_active:
                return None

            try:
                # Handle hand absence or tracking loss
                if hand is None or hand.confidence < 0.60:
                    if self._pinch_state == PinchState.DRAGGING:
                        _send_mouse_up()
                        self._pinch_state = PinchState.RELEASED
                        print("[CursorControl] Tracking lost: Drag safely released.")
                    self._has_previous_pos = False
                    self._filter_x = None
                    self._filter_y = None
                    self._notify_status("CURSOR: TRACKING LOST")
                    return None

                lms = hand.landmarks
                thumb_tip = lms[4]
                index_tip = lms[8]

                # 1. Evaluate Pinch & Drag State with Distance Hysteresis
                dx_pinch = thumb_tip.x - index_tip.x
                dy_pinch = thumb_tip.y - index_tip.y
                pinch_dist = math.hypot(dx_pinch, dy_pinch)

                if self._pinch_state == PinchState.RELEASED:
                    is_pinched = pinch_dist <= self.pinch_click_distance
                else:
                    # In PINCHING or DRAGGING state: use hysteresis release threshold
                    is_pinched = pinch_dist <= self.pinch_release_distance

                # If pinched, track midpoint between thumb and index; otherwise track index tip
                if is_pinched:
                    target_cam_x = (thumb_tip.x + index_tip.x) * 0.5
                    target_cam_y = (thumb_tip.y + index_tip.y) * 0.5
                else:
                    target_cam_x = index_tip.x
                    target_cam_y = index_tip.y

                # 2. Map coordinates to screen pixels
                raw_screen_x, raw_screen_y = self.map_camera_to_screen(target_cam_x, target_cam_y)

                # 3. Apply One Euro Filter Smoothing
                if not self._has_previous_pos or self._filter_x is None or self._filter_y is None:
                    self._filter_x = OneEuroFilter(t_now, raw_screen_x, min_cutoff=self.min_cutoff, beta=self.beta)
                    self._filter_y = OneEuroFilter(t_now, raw_screen_y, min_cutoff=self.min_cutoff, beta=self.beta)
                    filtered_x = raw_screen_x
                    filtered_y = raw_screen_y
                    self._has_previous_pos = True
                else:
                    filtered_x = self._filter_x.filter(raw_screen_x, t_now)
                    filtered_y = self._filter_y.filter(raw_screen_y, t_now)

                # 4. Dead Zone Filter to eliminate stationary micro-jitter
                move_dist = math.hypot(filtered_x - self._last_screen_x, filtered_y - self._last_screen_y)
                if move_dist < self.dead_zone_px and self._has_previous_pos:
                    target_px_x = int(self._last_screen_x)
                    target_px_y = int(self._last_screen_y)
                else:
                    target_px_x = int(round(filtered_x))
                    target_px_y = int(round(filtered_y))
                    self._last_screen_x = filtered_x
                    self._last_screen_y = filtered_y

                # 5. Pinch State Machine Transition
                if is_pinched:
                    if self._pinch_state == PinchState.RELEASED:
                        self._pinch_state = PinchState.PINCHING
                        self._pinch_start_time = t_now
                        self._pinch_start_pos = (target_px_x, target_px_y)
                        self._notify_status("CURSOR: PINCH START")

                    elif self._pinch_state == PinchState.PINCHING:
                        hold_duration = t_now - self._pinch_start_time
                        if hold_duration >= self.drag_hold_threshold:
                            self._pinch_state = PinchState.DRAGGING
                            _send_mouse_down()
                            self._notify_status("CURSOR: DRAGGING")
                            print("[CursorControl] [DRAG START] Mouse down engaged.")

                    elif self._pinch_state == PinchState.DRAGGING:
                        self._notify_status(f"CURSOR: DRAGGING ({target_px_x}, {target_px_y})")

                else:
                    # Pinch is released
                    if self._pinch_state == PinchState.DRAGGING:
                        _send_mouse_up()
                        self._pinch_state = PinchState.RELEASED
                        self._notify_status("CURSOR: DRAG RELEASED")
                        print("[CursorControl] [DRAG END] Mouse up released.")

                    elif self._pinch_state == PinchState.PINCHING:
                        # Quick pinch released -> execute clean single click!
                        _send_mouse_click()
                        self._pinch_state = PinchState.RELEASED
                        self._notify_status("CURSOR: CLICK")
                        print("[CursorControl] [CLICK] Single left-click executed.")
                    else:
                        self._notify_status(f"CURSOR: ({target_px_x}, {target_px_y})")

                # 6. Dispatch OS cursor position
                _set_os_cursor_pos(target_px_x, target_px_y)
                return target_px_x, target_px_y

            except Exception as e:
                # Guaranteed safety: ensure mouse down is not left stuck on error
                if self._pinch_state == PinchState.DRAGGING:
                    try:
                        _send_mouse_up()
                    except Exception:
                        pass
                    self._pinch_state = PinchState.RELEASED
                print(f"[CursorControl] Error in update: {e}")
                return None


# ---------------------------------------------------------------------------
# Global Singleton Accessor
# ---------------------------------------------------------------------------

_GLOBAL_CURSOR_CONTROLLER: Optional[VisionCursorController] = None
_CURSOR_LOCK = threading.Lock()

def get_cursor_controller() -> VisionCursorController:
    """Get or initialize the global shared VisionCursorController singleton."""
    global _GLOBAL_CURSOR_CONTROLLER
    with _CURSOR_LOCK:
        if _GLOBAL_CURSOR_CONTROLLER is None:
            _GLOBAL_CURSOR_CONTROLLER = VisionCursorController()
        return _GLOBAL_CURSOR_CONTROLLER
