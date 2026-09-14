"""
Centralized Camera Pipeline for JARVIS (MARK II / MARK XL).

Provides thread-safe, non-competing camera frame acquisition for:
  - Real-time Hand Tracking
  - PyQt6 Live HUD feed
  - Vision Snapshot capture (NVIDIA NIM analysis)
  - Exercise / pose tracking plugins

Prevents multiple cv2.VideoCapture instances from conflicting on Windows DirectShow.
"""
from __future__ import annotations

import io
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"


def ensure_windows_camera_permissions() -> None:
    """Ensure Windows Privacy settings permit Python to access camera devices."""
    if platform.system().lower() != "windows":
        return
    try:
        import winreg
        root = winreg.HKEY_CURRENT_USER
        base_sub = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
        try:
            with winreg.OpenKey(root, base_sub, 0, winreg.KEY_READ | winreg.KEY_WRITE) as key:
                i = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        if "python" in subkey_name.lower():
                            with winreg.OpenKey(root, f"{base_sub}\\{subkey_name}", 0, winreg.KEY_SET_VALUE) as subkey:
                                winreg.SetValueEx(subkey, "Value", 0, winreg.REG_SZ, "Allow")
                        i += 1
                    except OSError:
                        break
        except Exception:
            pass

        try:
            exe_path = sys.executable.replace("/", "\\").replace("\\", "#")
            np_key = f"{base_sub}\\NonPackaged\\{exe_path}"
            with winreg.CreateKey(root, np_key) as subkey:
                winreg.SetValueEx(subkey, "Value", 0, winreg.REG_SZ, "Allow")
        except Exception:
            pass
    except Exception:
        pass


def _get_cv2_candidate_backends() -> list[tuple[str, int]]:
    """Return an ordered list of candidate (name, backend_id) pairs."""
    system = platform.system().lower()
    if system == "windows":
        candidates = []
        if hasattr(cv2, "CAP_DSHOW"):
            candidates.append(("DirectShow", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            candidates.append(("MediaFoundation", cv2.CAP_MSMF))
        candidates.append(("Default", cv2.CAP_ANY))
        return candidates
    if system == "darwin":
        candidates = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            candidates.append(("AVFoundation", cv2.CAP_AVFOUNDATION))
        candidates.append(("Default", cv2.CAP_ANY))
        return candidates
    return [("Default", cv2.CAP_ANY)]


def _get_cv2_backend() -> int:
    """Return the primary OpenCV camera backend for the current OS."""
    candidates = _get_cv2_candidate_backends()
    return candidates[0][1] if candidates else cv2.CAP_ANY


def enumerate_camera_devices(max_devices: int = 4) -> list[dict]:
    """
    Enumerate and probe available camera devices on the host system.
    Returns a list of device descriptors with name, backend, resolution, and status.
    """
    ensure_windows_camera_permissions()
    devices: list[dict] = []
    names: list[str] = []

    # Attempt to retrieve FriendlyNames on Windows via DirectShow
    if platform.system().lower() == "windows":
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            names = graph.get_input_devices()
        except Exception:
            names = []

    backends = _get_cv2_candidate_backends()

    for idx in range(max_devices):
        dev_name = names[idx] if idx < len(names) else f"Camera {idx}"
        if "smart connect" in dev_name.lower():
            devices.append({
                "index": idx,
                "name": dev_name,
                "backend": "None",
                "resolution": (0, 0),
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "format": "None",
                "status": "UNAVAILABLE",
            })
            continue

        opened = False
        res = (0, 0)
        fps = 0.0
        used_backend = "None"
        fourcc_str = "None"

        for b_name, b_id in backends:
            try:
                cap = cv2.VideoCapture(idx, b_id)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        opened = True
                        h, w = frame.shape[:2]
                        res = (w, h)
                        fps_prop = cap.get(cv2.CAP_PROP_FPS)
                        fps = float(fps_prop) if fps_prop > 0 else 30.0
                        used_backend = b_name
                        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                        fourcc_str = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]) if fourcc_int > 0 else "DEFAULT"
                        cap.release()
                        break
                cap.release()
            except Exception:
                pass

        devices.append({
            "index": idx,
            "name": dev_name,
            "backend": used_backend,
            "resolution": res,
            "width": res[0],
            "height": res[1],
            "fps": fps,
            "format": fourcc_str,
            "status": "ONLINE" if opened else "UNAVAILABLE",
        })

    return devices


def _load_camera_config() -> dict:
    """Read camera settings from config/api_keys.json."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[Camera] Warning reading config: {e}")
    return {}


class CameraService:
    """
    Unified Camera Service running a single background capture thread.
    Distributes frames to subscribers without creating competing device handles.
    """

    def __init__(
        self,
        camera_index: Optional[int] = None,
        resolution: Optional[tuple[int, int]] = None,
        target_fps: Optional[int] = None,
        mirror: Optional[bool] = None,
    ):
        cfg = _load_camera_config()
        self.camera_index = camera_index if camera_index is not None else int(cfg.get("camera_index", 0))

        # Camera resolution negotiation ("auto" or explicit [w, h])
        res_cfg = cfg.get("camera_resolution", cfg.get("processing_resolution", "auto"))
        if resolution is not None:
            self.resolution = resolution
            self.auto_resolution = False
        elif isinstance(res_cfg, str) and res_cfg.lower() == "auto":
            self.resolution = (1280, 720)
            self.auto_resolution = True
        elif isinstance(res_cfg, (list, tuple)) and len(res_cfg) == 2:
            self.resolution = (int(res_cfg[0]), int(res_cfg[1]))
            self.auto_resolution = False
        else:
            self.resolution = (1280, 720)
            self.auto_resolution = True

        fps_cfg = cfg.get("camera_fps", cfg.get("target_fps", 30))
        if target_fps is not None:
            self.target_fps = target_fps
        elif str(fps_cfg).lower() == "auto":
            self.target_fps = 30
        else:
            self.target_fps = int(fps_cfg)

        self.camera_mirror = mirror if mirror is not None else bool(cfg.get("camera_mirror", True))

        self.device_name: str = ""
        self.actual_resolution: tuple[int, int] = (0, 0)
        self.actual_fps: float = 0.0
        self.actual_format: str = ""
        self.active_backend: str = ""

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_id: int = 0
        self._fps_measured: float = 0.0

        self._subscribers: dict[str, Callable[[np.ndarray, float], None]] = {}
        self._sub_counter: int = 0
        self._state: str = "IDLE"  # IDLE, RUNNING, UNAVAILABLE, ERROR
        self._error_msg: str = ""

    @property
    def is_running(self) -> bool:
        return self._state == "RUNNING" and self._thread is not None and self._thread.is_alive()

    @property
    def state(self) -> str:
        return self._state

    @property
    def fps(self) -> float:
        return self._fps_measured

    @property
    def error_message(self) -> str:
        return self._error_msg

    def get_camera_profile(self) -> dict:
        """Return full hardware and operational telemetry profile."""
        with self._lock:
            return {
                "device_name": self.device_name or f"Camera {self.camera_index}",
                "index": self.camera_index,
                "backend": self.active_backend or "Unknown",
                "requested_resolution": self.resolution,
                "actual_resolution": self.actual_resolution if self.actual_resolution != (0, 0) else self.resolution,
                "actual_fps": self.actual_fps if self.actual_fps > 0 else float(self.target_fps),
                "measured_fps": self._fps_measured,
                "pixel_format": self.actual_format or "YUY2",
                "state": self._state,
                "mirror": self.camera_mirror,
            }

    def subscribe(self, callback: Callable[[np.ndarray, float], None], sub_id: Optional[str] = None, auto_start: bool = True) -> str:
        """
        Register a consumer to receive (frame, timestamp) on each new capture.
        Starts the camera if not already running and auto_start is True.
        """
        with self._lock:
            if sub_id is None:
                self._sub_counter += 1
                sub_id = f"sub_{self._sub_counter}"
            self._subscribers[sub_id] = callback

        if auto_start and not self.is_running:
            self.start()
        return sub_id

    def unsubscribe(self, sub_id: str) -> None:
        """Unregister a consumer."""
        with self._lock:
            self._subscribers.pop(sub_id, None)

    def get_latest_frame(self) -> tuple[Optional[np.ndarray], float]:
        """Get a copy of the latest captured frame and its timestamp without blocking."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_timestamp

    def start(self) -> bool:
        """Start the camera capture loop in a dedicated background worker."""
        with self._lock:
            if self.is_running:
                return True

            self._stop_event.clear()
            self._state = "INITIALIZING"
            self._error_msg = ""

            self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-pipeline")
            self._thread.start()

        # Wait briefly for startup
        t_start = time.time()
        while time.time() - t_start < 5.0:
            if self._state in ("RUNNING", "UNAVAILABLE", "ERROR"):
                break
            time.sleep(0.05)

        return self._state == "RUNNING"

    def stop(self) -> None:
        """Stop the camera capture loop and release the hardware handle."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._state = "IDLE"
            self._latest_frame = None

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Safely negotiate and open the highest practical quality video capture handle."""
        ensure_windows_camera_permissions()

        # Build candidate device indices (configured index first, fallback 0)
        candidates = [self.camera_index]
        if 0 not in candidates:
            candidates.append(0)

        backends = _get_cv2_candidate_backends()

        # Build candidate resolutions (prioritize native HD 720p)
        if getattr(self, "auto_resolution", True):
            res_candidates = [(1280, 720), (1920, 1080), (960, 540), (640, 480)]
        else:
            res_candidates = [self.resolution]
            for fb in [(1280, 720), (640, 480)]:
                if fb not in res_candidates:
                    res_candidates.append(fb)

        # Retrieve FriendlyNames if on Windows
        dev_names = []
        if platform.system().lower() == "windows":
            try:
                from pygrabber.dshow_graph import FilterGraph
                graph = FilterGraph()
                dev_names = graph.get_input_devices()
            except Exception:
                dev_names = []

        for idx in candidates:
            dev_name = dev_names[idx] if idx < len(dev_names) else f"Camera {idx}"
            if "smart connect" in dev_name.lower():
                continue
            for b_name, b_id in backends:
                cap = None
                try:
                    cap = cv2.VideoCapture(idx, b_id)
                    if not cap.isOpened():
                        if cap is not None:
                            cap.release()
                        continue
                except Exception:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    continue

                # Camera device opened with backend b_name. Negotiate resolution & format.
                negotiated = False
                try:
                    for req_w, req_h in res_candidates:
                        for codec in [None, cv2.VideoWriter_fourcc(*'MJPG')]:
                            try:
                                if codec is not None:
                                    cap.set(cv2.CAP_PROP_FOURCC, codec)
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
                                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                                try:
                                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                except Exception:
                                    pass

                                ret, test_frame = cap.read()
                                if not ret or test_frame is None or test_frame.size == 0:
                                    continue

                                # Query actual negotiated properties
                                act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                if act_w <= 0 or act_h <= 0:
                                    act_h, act_w = test_frame.shape[:2]

                                act_fps = cap.get(cv2.CAP_PROP_FPS)
                                fps_val = float(act_fps) if act_fps > 0 else float(self.target_fps)

                                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                                fourcc_str = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]) if fourcc_int > 0 else "DEFAULT"

                                # Successfully verified working camera!
                                self.camera_index = idx
                                self.device_name = dev_name
                                self.resolution = (act_w, act_h)
                                self.actual_resolution = (act_w, act_h)
                                self.actual_fps = fps_val
                                self.actual_format = fourcc_str
                                self.active_backend = b_name

                                # Discard 2 warm-up frames
                                for _ in range(2):
                                    cap.read()

                                print(
                                    f"[Camera] [OK] Connected to '{dev_name}' (idx {idx}) via {b_name} "
                                    f"at {act_w}x{act_h} @ {fps_val:.1f} FPS (Format: {fourcc_str})."
                                )
                                negotiated = True
                                return cap
                            except Exception:
                                continue
                finally:
                    if not negotiated and cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass

        return None

    def _capture_loop(self) -> None:
        """Background worker loop for camera acquisition."""
        cap = self._open_capture()
        if cap is None or not cap.isOpened():
            with self._lock:
                self._state = "UNAVAILABLE"
                self._error_msg = f"Camera {self.camera_index} is unavailable or locked by another application."
            print(f"[Camera] [FAIL] {self._error_msg}")
            return

        with self._lock:
            self._cap = cap
            self._state = "RUNNING"
        print(f"[Camera] [OK] Camera running at {self.actual_resolution[0]}x{self.actual_resolution[1]} @ {self.actual_fps:.1f} FPS (Backend: {self.active_backend}).")

        frame_interval = 1.0 / max(1, self.target_fps)
        fps_timer = time.time()
        fps_counter = 0
        consecutive_failures = 0
        has_received_frame = False

        try:
            while not self._stop_event.is_set():
                t_loop_start = time.time()
                ret, frame = cap.read()

                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    fail_limit = 90 if has_received_frame else 150
                    if consecutive_failures > fail_limit:
                        with self._lock:
                            self._state = "RECONNECTING"
                            self._error_msg = "Camera disconnected or signal lost. Attempting recovery..."
                        print(f"[Camera] [WARN] {self._error_msg}")
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None

                        # Recovery loop: attempt to re-open camera
                        while not self._stop_event.is_set():
                            if self._stop_event.wait(1.5):
                                break
                            reconnected_cap = self._open_capture()
                            if reconnected_cap is not None and reconnected_cap.isOpened():
                                cap = reconnected_cap
                                consecutive_failures = 0
                                has_received_frame = False
                                with self._lock:
                                    self._cap = cap
                                    self._state = "RUNNING"
                                    self._error_msg = ""
                                print(f"[Camera] [OK] Camera successfully recovered and re-opened.")
                                break

                        if cap is None:
                            break
                    time.sleep(0.02)
                    continue

                has_received_frame = True
                consecutive_failures = 0
                ts = time.time()
                with self._lock:
                    self._latest_frame = frame
                    self._latest_timestamp = ts
                    self._frame_id += 1

                # Calculate measured FPS
                fps_counter += 1
                if ts - fps_timer >= 1.0:
                    self._fps_measured = fps_counter / (ts - fps_timer)
                    fps_counter = 0
                    fps_timer = ts

                # Broadcast to subscribers safely
                with self._lock:
                    subs = list(self._subscribers.values())

                for sub in subs:
                    try:
                        sub(frame, ts)
                    except Exception as sub_err:
                        print(f"[Camera] Subscriber callback error: {sub_err}")

                # Throttle to target FPS
                elapsed = time.time() - t_loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0.002:
                    if self._stop_event.wait(sleep_time):
                        break

        except Exception as e:
            with self._lock:
                self._state = "ERROR"
                self._error_msg = str(e)
            print(f"[Camera] Error in capture loop: {e}")
        finally:
            with self._lock:
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                self._state = "IDLE"
                self._latest_frame = None
            print("[Camera] Camera pipeline stopped.")

    def capture_snapshot(self, quality: int = 75) -> tuple[bytes, str]:
        """
        Capture an immediate JPEG snapshot without reopening hardware if running.
        If idle, opens camera momentarily, grabs 1 frame, and releases.
        """
        frame, _ = self.get_latest_frame()
        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes(), "image/jpeg"

        # Camera is not running -> grab single snapshot safely via candidate probing
        cap = self._open_capture()
        if cap is None:
            raise RuntimeError(f"Camera could not be opened for snapshot.")

        try:
            for _ in range(3):
                cap.read()
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Camera returned empty frame.")
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes(), "image/jpeg"
        finally:
            cap.release()


_GLOBAL_CAMERA: Optional[CameraService] = None
_GLOBAL_LOCK = threading.Lock()


def get_camera_service() -> CameraService:
    """Get or initialize the global shared CameraService singleton."""
    global _GLOBAL_CAMERA
    with _GLOBAL_LOCK:
        if _GLOBAL_CAMERA is None:
            _GLOBAL_CAMERA = CameraService()
        return _GLOBAL_CAMERA
