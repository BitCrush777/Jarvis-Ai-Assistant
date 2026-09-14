"""
Real-Time Hand Tracking Foundation for JARVIS (MARK II / MARK XL).

Detects and tracks one or two hands using MediaPipe Tasks HandLandmarker.
Optimized Features:
  - Decoupled asynchronous worker thread: camera capture never blocked by inference
  - Stale frame skipping: always processes the latest available frame
  - High-precision 21-landmark 3D coordinate extraction
  - Handedness classification (Left / Right)
  - Key landmark extraction: wrist, palm center, 5 fingertips
  - Rotation-invariant finger extension analysis (extended vs folded)
  - One Euro Filter adaptive landmark smoothing (zero jitter, low lag)
  - Graceful disappearance & occlusion handling
  - Full-spectrum JARVIS HUD telemetry: FPS, gesture, cursor, CPU, RAM, active monitor
  - Clean decoupled interface for Phase 2 computer control
"""
from __future__ import annotations

import math
import os
import platform
import psutil
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from core.camera import get_camera_service, _load_camera_config
from core.window_manager import get_monitor_manager

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class HandLandmark:
    id: int
    name: str
    x: float       # Normalized [0, 1]
    y: float       # Normalized [0, 1]
    z: float       # Depth estimate relative to wrist
    x_px: int      # Pixel coordinate X
    y_px: int      # Pixel coordinate Y


@dataclass
class HandInfo:
    handedness: str                        # "Left" or "Right"
    confidence: float                      # Score in [0, 1]
    wrist: HandLandmark
    palm_center: HandLandmark
    thumb_tip: HandLandmark
    index_tip: HandLandmark
    middle_tip: HandLandmark
    ring_tip: HandLandmark
    pinky_tip: HandLandmark
    landmarks: list[HandLandmark]          # All 21 landmarks
    fingers_extended: dict[str, bool]      # {"thumb": bool, "index": bool, ...}

    @property
    def is_extended(self) -> dict[str, bool]:
        return self.fingers_extended


@dataclass
class HandFrameResult:
    timestamp: float
    hands: list[HandInfo]
    hand_count: int
    fps: float
    frame_width: int
    frame_height: int
    annotated_frame: Optional[np.ndarray] = None


# Landmark Name Map
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

SKELETON_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Knuckle base
    (5, 9), (9, 13), (13, 17),
]


# ---------------------------------------------------------------------------
# One Euro Filter for Jitter Reduction
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """
    Adaptive low-pass filter providing high smoothing at low speeds
    and near-zero latency during rapid motion.
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.5,
        beta: float = 0.01,
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


class LandmarkSmoother:
    """Manages independent 3D OneEuroFilters for 21 hand landmarks per hand."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.01):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[str, list[list[OneEuroFilter]]] = {}

    def reset(self, handedness: Optional[str] = None) -> None:
        if handedness:
            self._filters.pop(handedness, None)
        else:
            self._filters.clear()

    def reset_hand(self, handedness: str) -> None:
        self.reset(handedness)

    def smooth(
        self,
        handedness: str,
        t: float,
        coords: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        if handedness not in self._filters:
            hand_filters = []
            for x, y, z in coords:
                fx = OneEuroFilter(t, x, min_cutoff=self.min_cutoff, beta=self.beta)
                fy = OneEuroFilter(t, y, min_cutoff=self.min_cutoff, beta=self.beta)
                fz = OneEuroFilter(t, z, min_cutoff=self.min_cutoff, beta=self.beta)
                hand_filters.append([fx, fy, fz])
            self._filters[handedness] = hand_filters
            return coords

        smoothed = []
        hand_filters = self._filters[handedness]
        for i, (x, y, z) in enumerate(coords):
            fx, fy, fz = hand_filters[i]
            sx = fx.filter(x, t)
            sy = fy.filter(y, t)
            sz = fz.filter(z, t)
            smoothed.append((sx, sy, sz))

        return smoothed


# ---------------------------------------------------------------------------
# Hand Tracker Core Engine
# ---------------------------------------------------------------------------

class HandTracker:
    """
    High-performance real-time hand tracker using MediaPipe Tasks.
    Processes frames asynchronously in a dedicated worker thread with latest-frame-only drops.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_hands: Optional[int] = None,
        detection_confidence: Optional[float] = None,
        tracking_confidence: Optional[float] = None,
        debug_mode: Optional[bool] = None,
        mirror_camera: Optional[bool] = None,
    ):
        cfg = _load_camera_config()
        self.max_hands = max_hands or int(cfg.get("max_hands", 2))
        self.detection_confidence = detection_confidence or float(cfg.get("hand_detection_confidence", 0.5))
        self.tracking_confidence = tracking_confidence or float(cfg.get("hand_tracking_confidence", 0.5))
        self.debug_mode = debug_mode if debug_mode is not None else bool(cfg.get("vision_debug_mode", False))
        self.camera_mirror = mirror_camera if mirror_camera is not None else bool(cfg.get("camera_mirror", True))

        # Configurable decoupled tracking resolution
        track_res_cfg = cfg.get("tracking_resolution", "auto")
        if track_res_cfg == "auto" or not isinstance(track_res_cfg, (list, tuple)) or len(track_res_cfg) != 2:
            self.tracking_resolution: Optional[tuple[int, int]] = None
        else:
            self.tracking_resolution = (int(track_res_cfg[0]), int(track_res_cfg[1]))

        self.model_path = model_path or str(MODEL_PATH)
        self._ensure_model_downloaded()

        self._landmarker: Optional[vision.HandLandmarker] = None
        self._smoother = LandmarkSmoother(min_cutoff=1.5, beta=0.01)
        self._lock = threading.Lock()

        # Decoupled worker queue & thread
        self._pending_frame: Optional[tuple[np.ndarray, float]] = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._camera_sub_id: Optional[str] = None
        self._is_active = False

        self._latest_result: Optional[HandFrameResult] = None
        self._callbacks: list[Callable[[HandFrameResult], None]] = []

        # Performance telemetry
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._current_fps = 0.0
        self._process = psutil.Process()

        self._init_detector()

    def _ensure_model_downloaded(self) -> None:
        """Download hand_landmarker.task if not present locally."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"[HandTracker] Downloading MediaPipe hand landmarker model to {self.model_path}...")
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
                print(f"[HandTracker] Model downloaded ({os.path.getsize(self.model_path):,} bytes).")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download hand tracking model from {MODEL_URL}: {e}"
                ) from e

    def _init_detector(self) -> None:
        """Initialize the MediaPipe HandLandmarker instance."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_confidence,
            min_hand_presence_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        print("[HandTracker] [OK] MediaPipe HandLandmarker ready.")

    @property
    def is_active(self) -> bool:
        return self._is_active

    def set_debug_mode(self, enabled: bool) -> None:
        """Toggle the visual debug overlay on or off."""
        with self._lock:
            self.debug_mode = enabled

    def register_callback(self, callback: Callable[[HandFrameResult], None]) -> None:
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[HandFrameResult], None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    add_callback = register_callback
    remove_callback = unregister_callback

    def get_latest_result(self) -> Optional[HandFrameResult]:
        with self._lock:
            return self._latest_result

    def start(self) -> bool:
        """Connect to the unified camera pipeline and begin real-time hand tracking."""
        with self._lock:
            if self._is_active:
                return True
            self._is_active = True
            self._stop_event.clear()
            self._frame_event.clear()
            self._pending_frame = None

            # Start decoupled worker thread
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="hand-tracker-worker",
            )
            self._worker_thread.start()

        camera_service = get_camera_service()
        self._camera_sub_id = camera_service.subscribe(self._on_camera_frame, sub_id="hand_tracker")
        print("[HandTracker] Hand tracking started (decoupled worker active).")
        return True

    def stop(self) -> None:
        """Disconnect from camera pipeline and stop tracking."""
        self._stop_event.set()
        self._frame_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        with self._lock:
            self._is_active = False
            self._latest_result = None

        if self._camera_sub_id:
            camera_service = get_camera_service()
            camera_service.unsubscribe(self._camera_sub_id)
            self._camera_sub_id = None
        print("[HandTracker] Hand tracking stopped.")

    def _on_camera_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Instant subscriber callback invoked by CameraService.
        Stores the newest frame and immediately signals the worker thread (< 1µs).
        Never blocks camera acquisition!
        """
        if not self._is_active:
            return

        if self._worker_thread is not None and self._worker_thread.is_alive():
            with self._frame_lock:
                self._pending_frame = (frame, timestamp)
            self._frame_event.set()
        else:
            # Synchronous fallback for unit tests or offline pipelines
            self._process_and_dispatch(frame, timestamp)

    def _process_and_dispatch(self, frame: np.ndarray, timestamp: float) -> tuple[HandFrameResult, Optional[np.ndarray]]:
        result, annotated = self.process_frame(frame, timestamp)

        # Forward to GestureController
        try:
            from core.gesture_control import get_gesture_controller
            gc = get_gesture_controller()
            if gc.enabled:
                dominant_hand = result.hands[0] if result.hands else None
                gc.process_hand(dominant_hand, result.timestamp)
        except Exception:
            pass

        with self._lock:
            self._latest_result = result
            callbacks = list(self._callbacks)

        for cb in callbacks:
            try:
                cb(result)
            except Exception as cb_err:
                print(f"[HandTracker] Callback error: {cb_err}")

        return result, annotated

    def _worker_loop(self) -> None:
        """Decoupled worker loop: processes only the freshest available frame."""
        while not self._stop_event.is_set():
            if not self._frame_event.wait(0.05):
                continue
            self._frame_event.clear()

            with self._frame_lock:
                item = self._pending_frame
                self._pending_frame = None

            if item is None:
                continue

            frame, timestamp = item
            try:
                self._process_and_dispatch(frame, timestamp)
            except Exception as loop_err:
                print(f"[HandTracker] Worker loop error: {loop_err}")

    # -----------------------------------------------------------------------
    # Core Landmark & Geometry Processing
    # -----------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> tuple[HandFrameResult, Optional[np.ndarray]]:
        """
        Process a single BGR image frame and return HandFrameResult + annotated debug frame.
        Can be called directly for offline testing or video pipeline integration.
        """
        if frame is None or frame.size == 0 or self._landmarker is None:
            empty = HandFrameResult(
                timestamp=timestamp or time.time(),
                hands=[],
                hand_count=0,
                fps=0.0,
                frame_width=0,
                frame_height=0,
            )
            return empty, frame

        t_now = timestamp or time.time()
        h, w = frame.shape[:2]

        # Calculate FPS
        self._fps_counter += 1
        if t_now - self._fps_timer >= 1.0:
            self._current_fps = self._fps_counter / (t_now - self._fps_timer)
            self._fps_counter = 0
            self._fps_timer = t_now

        try:
            # Prepare tracking image (decoupled tracking resolution if configured)
            if self.tracking_resolution is not None and (w, h) != self.tracking_resolution:
                tracking_frame = cv2.resize(frame, self.tracking_resolution, interpolation=cv2.INTER_AREA)
            else:
                tracking_frame = frame

            # Convert to RGB MediaPipe Image
            rgb_frame = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hands
            detection_result = self._landmarker.detect(mp_image)
        except Exception as det_err:
            print(f"[HandTracker] MediaPipe inference error: {det_err}. Reinitializing detector...")
            try:
                self._init_detector()
            except Exception:
                pass
            clean_preview = cv2.flip(frame, 1) if self.camera_mirror else frame.copy()
            empty = HandFrameResult(
                timestamp=t_now,
                hands=[],
                hand_count=0,
                fps=self._current_fps,
                frame_width=w,
                frame_height=h,
                annotated_frame=clean_preview,
            )
            return empty, clean_preview

        hands: list[HandInfo] = []
        raw_landmarks_list = detection_result.hand_landmarks or []
        handedness_list = detection_result.handedness or []

        detected_handedness = set()

        for idx, raw_landmarks in enumerate(raw_landmarks_list):
            handedness_label = "Right"
            confidence = 0.85
            if idx < len(handedness_list) and handedness_list[idx]:
                top_cat = handedness_list[idx][0]
                handedness_label = top_cat.category_name or "Right"
                confidence = float(top_cat.score)

            detected_handedness.add(handedness_label)

            coords = [(lm.x, lm.y, lm.z) for lm in raw_landmarks]
            smoothed_coords = self._smoother.smooth(handedness_label, t_now, coords)

            landmarks = []
            for lm_idx, (sx, sy, sz) in enumerate(smoothed_coords):
                landmarks.append(
                    HandLandmark(
                        id=lm_idx,
                        name=LANDMARK_NAMES[lm_idx],
                        x=sx,
                        y=sy,
                        z=sz,
                        x_px=int(np.clip(sx * w, 0, w - 1)),
                        y_px=int(np.clip(sy * h, 0, h - 1)),
                    )
                )

            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            # Compute Palm Center (wrist + 4 MCP knuckles average)
            mcp_indices = [0, 1, 5, 9, 13, 17]
            palm_x = sum(landmarks[i].x for i in mcp_indices) / len(mcp_indices)
            palm_y = sum(landmarks[i].y for i in mcp_indices) / len(mcp_indices)
            palm_z = sum(landmarks[i].z for i in mcp_indices) / len(mcp_indices)

            palm_center = HandLandmark(
                id=-1,
                name="PALM_CENTER",
                x=palm_x,
                y=palm_y,
                z=palm_z,
                x_px=int(np.clip(palm_x * w, 0, w - 1)),
                y_px=int(np.clip(palm_y * h, 0, h - 1)),
            )

            # Analyze finger extensions
            fingers_extended = self._analyze_fingers_extended(landmarks, handedness_label)

            hand_info = HandInfo(
                handedness=handedness_label,
                confidence=confidence,
                wrist=wrist,
                palm_center=palm_center,
                thumb_tip=thumb_tip,
                index_tip=index_tip,
                middle_tip=middle_tip,
                ring_tip=ring_tip,
                pinky_tip=pinky_tip,
                landmarks=landmarks,
                fingers_extended=fingers_extended,
            )
            hands.append(hand_info)

        # Reset smoother for hands that disappeared
        for prev_hand in ("Left", "Right"):
            if prev_hand not in detected_handedness:
                self._smoother.reset(prev_hand)

        annotated: Optional[np.ndarray] = None
        if self.debug_mode:
            annotated = self._draw_debug_overlay(frame.copy(), hands, self._current_fps)
        else:
            annotated = self.draw_preview_overlay(frame.copy(), hands, mirror=self.camera_mirror)

        result = HandFrameResult(
            timestamp=t_now,
            hands=hands,
            hand_count=len(hands),
            fps=self._current_fps,
            frame_width=w,
            frame_height=h,
            annotated_frame=annotated,
        )

        return result, annotated

    def draw_preview_overlay(
        self,
        frame: np.ndarray,
        hands: list[HandInfo],
        mirror: bool = True,
    ) -> np.ndarray:
        """
        Renders a clean live camera preview with precise hand skeleton landmarks overlay.
        When mirror=True, flips the image horizontally and adjusts coordinates accordingly.
        When no hands are detected, returns the clean camera frame.
        """
        img = cv2.flip(frame, 1) if mirror else frame
        h, w = img.shape[:2]

        if not hands:
            return img

        # Compute dynamic scale based on resolution so overlays remain crisp and proportional
        scale = max(0.65, h / 720.0)
        line_w = max(1, int(round(2 * scale)))
        r_tip = max(4, int(round(6 * scale)))
        r_halo = max(6, int(round(8 * scale)))
        r_reticle = max(9, int(round(14 * scale)))
        r_wrist = max(3, int(round(5 * scale)))
        r_knuckle = max(2, int(round(3 * scale)))
        r_palm = max(4, int(round(7 * scale)))

        CYAN = (255, 215, 0)       # BGR: Neon Cyan
        DIM_CYAN = (180, 130, 0)
        GOLD = (0, 215, 255)       # Gold / Amber
        GREEN = (50, 235, 50)      # Neon Green
        WHITE = (245, 245, 255)

        cursor_active = False
        try:
            from core.cursor_control import get_cursor_controller
            cursor_active = get_cursor_controller().is_active
        except Exception:
            cursor_active = False

        for hand in hands:
            lms = hand.landmarks
            pts = []
            for lm in lms:
                px = (w - 1 - lm.x_px) if mirror else lm.x_px
                py = lm.y_px
                pts.append((px, py))

            # 1. Draw Skeleton Bones
            for p1, p2 in SKELETON_CONNECTIONS:
                if p1 < len(pts) and p2 < len(pts):
                    cv2.line(img, pts[p1], pts[p2], DIM_CYAN, line_w, cv2.LINE_AA)

            # 2. Draw Landmark Joints & Glowing Fingertips
            for idx, (px, py) in enumerate(pts):
                if idx in (4, 8, 12, 16, 20):  # Fingertips
                    is_pointing_tip = (idx == 8 and cursor_active)
                    halo_col = GREEN if is_pointing_tip else GOLD
                    cv2.circle(img, (px, py), r_tip, halo_col, -1, cv2.LINE_AA)
                    cv2.circle(img, (px, py), r_halo, WHITE, max(1, int(round(1 * scale))), cv2.LINE_AA)
                    if is_pointing_tip:
                        # Targeting reticle
                        cv2.circle(img, (px, py), r_reticle, GREEN, max(1, int(round(1.5 * scale))), cv2.LINE_AA)
                elif idx == 0:  # Wrist
                    cv2.circle(img, (px, py), r_wrist, CYAN, -1, cv2.LINE_AA)
                else:  # Knuckles
                    cv2.circle(img, (px, py), r_knuckle, CYAN, -1, cv2.LINE_AA)

            # 3. Palm Center Marker
            if len(pts) > 9:
                palm_x = int((pts[0][0] + pts[9][0]) * 0.5)
                palm_y = int((pts[0][1] + pts[9][1]) * 0.5)
                cv2.circle(img, (palm_x, palm_y), r_palm, GOLD, max(1, int(round(1 * scale))), cv2.LINE_AA)

        return img

    # -----------------------------------------------------------------------
    # Finger Extension Analysis
    # -----------------------------------------------------------------------

    @staticmethod
    def _classify_finger_extension(
        lms: list[HandLandmark],
        handedness: str = "Right",
    ) -> dict[str, bool]:
        """
        Static classification of finger extension (extended vs folded).
        Rotation-invariant based on wrist and MCP knuckles geometry.
        """
        wrist = lms[0]

        def dist(p1: HandLandmark, p2: HandLandmark) -> float:
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        hand_scale = max(1e-4, dist(wrist, lms[9]))

        thumb_tip = lms[4]
        thumb_mcp = lms[2]
        pinky_mcp = lms[17]
        dist_tip_pinky = dist(thumb_tip, pinky_mcp)
        dist_mcp_pinky = dist(thumb_mcp, pinky_mcp)
        thumb_extended = (dist_tip_pinky > dist_mcp_pinky * 1.15) and (dist(thumb_tip, wrist) > hand_scale * 0.75)

        finger_configs = [
            ("index", 8, 6, 5),
            ("middle", 12, 10, 9),
            ("ring", 16, 14, 13),
            ("pinky", 20, 18, 17),
        ]

        extensions = {"thumb": thumb_extended}

        for name, tip_id, pip_id, mcp_id in finger_configs:
            tip = lms[tip_id]
            pip = lms[pip_id]
            mcp = lms[mcp_id]

            d_tip_wrist = dist(tip, wrist)
            d_pip_wrist = dist(pip, wrist)
            d_mcp_wrist = dist(mcp, wrist)

            extended = (d_tip_wrist > d_pip_wrist * 1.05) and (d_tip_wrist > d_mcp_wrist * 1.25)
            extensions[name] = extended

        return extensions

    def _analyze_fingers_extended(
        self,
        lms: list[HandLandmark],
        handedness: str,
    ) -> dict[str, bool]:
        return self._classify_finger_extension(lms, handedness)
        """
        Determines whether each finger is extended or folded using rotation-invariant
        distances from the wrist and knuckle MCP joints.
        """
        wrist = lms[0]

        def dist(p1: HandLandmark, p2: HandLandmark) -> float:
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        # Palm scale reference: distance from wrist (0) to middle MCP (9)
        hand_scale = max(1e-4, dist(wrist, lms[9]))

        # --- Thumb ---
        thumb_tip = lms[4]
        thumb_mcp = lms[2]
        pinky_mcp = lms[17]
        dist_tip_pinky = dist(thumb_tip, pinky_mcp)
        dist_mcp_pinky = dist(thumb_mcp, pinky_mcp)
        thumb_extended = (dist_tip_pinky > dist_mcp_pinky * 1.15) and (dist(thumb_tip, wrist) > hand_scale * 0.75)

        # --- Index, Middle, Ring, Pinky ---
        finger_configs = [
            ("index", 8, 6, 5),
            ("middle", 12, 10, 9),
            ("ring", 16, 14, 13),
            ("pinky", 20, 18, 17),
        ]

        extensions = {"thumb": thumb_extended}

        for name, tip_id, pip_id, mcp_id in finger_configs:
            tip = lms[tip_id]
            pip = lms[pip_id]
            mcp = lms[mcp_id]

            d_tip_wrist = dist(tip, wrist)
            d_pip_wrist = dist(pip, wrist)
            d_mcp_wrist = dist(mcp, wrist)

            extended = (d_tip_wrist > d_pip_wrist * 1.05) and (d_tip_wrist > d_mcp_wrist * 1.25)
            extensions[name] = extended

        return extensions

    # -----------------------------------------------------------------------
    # Full JARVIS HUD Telemetry Debug Overlay
    # -----------------------------------------------------------------------

    def _draw_debug_overlay(
        self,
        img: np.ndarray,
        hands: list[HandInfo],
        fps: float,
    ) -> np.ndarray:
        """
        Renders rich futuristic JARVIS-styled visual overlays with complete telemetry:
        CAM FPS, PROC FPS, HANDS, CONF, GESTURE, STATE, CURSOR, MONITOR, CPU, RAM.
        """
        CYAN = (255, 215, 0)       # BGR: Neon Cyan/Blue
        DIM_CYAN = (160, 120, 0)
        GOLD = (0, 215, 255)       # Gold / Amber
        GREEN = (50, 235, 50)      # Neon Green
        WHITE = (240, 240, 255)
        DARK_BG = (10, 6, 2)

        # 1. Draw Skeleton and Landmarks for each hand
        for hand in hands:
            lms = hand.landmarks

            # Draw bone connections
            for p1, p2 in SKELETON_CONNECTIONS:
                pt1 = (lms[p1].x_px, lms[p1].y_px)
                pt2 = (lms[p2].x_px, lms[p2].y_px)
                cv2.line(img, pt1, pt2, DIM_CYAN, 2, cv2.LINE_AA)

            # Draw joint landmarks
            for idx, lm in enumerate(lms):
                pos = (lm.x_px, lm.y_px)
                if idx in (4, 8, 12, 16, 20):
                    cv2.circle(img, pos, 6, GOLD, -1, cv2.LINE_AA)
                    cv2.circle(img, pos, 8, WHITE, 1, cv2.LINE_AA)
                elif idx == 0:
                    cv2.circle(img, pos, 5, CYAN, -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, pos, 3, CYAN, -1, cv2.LINE_AA)

            # Palm Center reticle
            px, py = hand.palm_center.x_px, hand.palm_center.y_px
            cv2.circle(img, (px, py), 9, GOLD, 1, cv2.LINE_AA)
            cv2.line(img, (px - 14, py), (px + 14, py), GOLD, 1, cv2.LINE_AA)
            cv2.line(img, (px, py - 14), (px, py + 14), GOLD, 1, cv2.LINE_AA)

            palm_txt = f"{hand.handedness} ({hand.palm_center.x:.2f}, {hand.palm_center.y:.2f})"
            cv2.putText(
                img,
                palm_txt,
                (px - 40, py - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                WHITE,
                1,
                cv2.LINE_AA,
            )

        # 2. Query System Resources & Telemetries
        try:
            ram_mb = self._process.memory_info().rss / (1024 * 1024)
            cpu_pct = self._process.cpu_percent(interval=None)
        except Exception:
            ram_mb = 0.0
            cpu_pct = 0.0

        mon_info = ""
        try:
            mm = get_monitor_manager()
            pri = mm.get_primary_monitor()
            mon_info = f"MONS: {mm.monitor_count} (Pri: {pri.width}x{pri.height})" if pri else f"MONS: {mm.monitor_count}"
        except Exception:
            pass

        cam_fps = 0.0
        try:
            cam_fps = get_camera_service().fps
        except Exception:
            pass

        # 3. Draw HUD Telemetry Header
        cursor_extra = 0
        cursor_active = False
        cursor_txt = ""
        try:
            from core.cursor_control import get_cursor_controller
            cur_c = get_cursor_controller()
            cursor_active = cur_c.is_active
            if cursor_active:
                cursor_extra = 24
                cursor_txt = cur_c.latest_telemetry
        except Exception:
            pass

        header_h = 100 + len(hands) * 26 + 26 + cursor_extra
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (410, header_h), DARK_BG, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (10, 10), (410, header_h), CYAN, 1)

        cv2.putText(
            img,
            f"JARVIS VISION // CAM: {cam_fps:.0f} FPS  PROC: {fps:.0f} FPS",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            CYAN,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            f"HANDS: {len(hands)}/2 | CPU: {cpu_pct:.0f}% | RAM: {ram_mb:.0f}MB | {mon_info}",
            (20, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            WHITE,
            1,
            cv2.LINE_AA,
        )

        # Hand status lines
        y_line = 76
        for h_idx, hand in enumerate(hands):
            y_line = 76 + h_idx * 24
            fe = hand.fingers_extended
            f_str = (
                f"T:{'Y' if fe['thumb'] else '-'} "
                f"I:{'Y' if fe['index'] else '-'} "
                f"M:{'Y' if fe['middle'] else '-'} "
                f"R:{'Y' if fe['ring'] else '-'} "
                f"P:{'Y' if fe['pinky'] else '-'}"
            )
            line_txt = f"[{hand.handedness[0]}] Conf: {hand.confidence*100:.0f}%  Fingers: {f_str}"
            cv2.putText(
                img,
                line_txt,
                (20, y_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                GOLD,
                1,
                cv2.LINE_AA,
            )

        # Gesture Control & Cursor Telemetry
        y_gesture = y_line + 22 if hands else 76
        gesture_txt = "GESTURE CONTROL: READY"
        g_color = CYAN
        try:
            from core.gesture_control import get_gesture_controller
            gc = get_gesture_controller()
            gesture_txt = gc.latest_status
            if "EXEC" in gesture_txt or "ACTIVE" in gesture_txt:
                g_color = GREEN
            elif "COOLDOWN" in gesture_txt:
                g_color = GOLD
            elif "RE-ARMED" in gesture_txt or "READY" in gesture_txt:
                g_color = CYAN
            else:
                g_color = WHITE
        except Exception:
            pass

        cv2.putText(
            img,
            gesture_txt,
            (20, y_gesture),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            g_color,
            1,
            cv2.LINE_AA,
        )

        if cursor_active and cursor_txt:
            cv2.putText(
                img,
                cursor_txt,
                (20, y_gesture + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                GREEN,
                1,
                cv2.LINE_AA,
            )

        # 4. Draw Index Fingertip Targeting Reticle when pointing mode is active
        if cursor_active and len(hands) > 0:
            for hand in hands:
                tip_x, tip_y = hand.index_tip.x_px, hand.index_tip.y_px
                cv2.circle(img, (tip_x, tip_y), 14, GREEN, 1, cv2.LINE_AA)
                cv2.circle(img, (tip_x, tip_y), 4, GREEN, -1, cv2.LINE_AA)
                cv2.line(img, (tip_x - 20, tip_y), (tip_x + 20, tip_y), GREEN, 1, cv2.LINE_AA)
                cv2.line(img, (tip_x, tip_y - 20), (tip_x, tip_y + 20), GREEN, 1, cv2.LINE_AA)

        return img


# ---------------------------------------------------------------------------
# Global Singleton Accessor
# ---------------------------------------------------------------------------

_GLOBAL_HAND_TRACKER: Optional[HandTracker] = None
_TRACKER_LOCK = threading.Lock()

def get_hand_tracker() -> HandTracker:
    """Get or initialize the global shared HandTracker singleton."""
    global _GLOBAL_HAND_TRACKER
    with _TRACKER_LOCK:
        if _GLOBAL_HAND_TRACKER is None:
            _GLOBAL_HAND_TRACKER = HandTracker()
        return _GLOBAL_HAND_TRACKER
