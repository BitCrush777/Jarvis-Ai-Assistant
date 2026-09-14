from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import mss
    import mss.tools
    _MSS = True
except ImportError:
    _MSS = False

try:
    import PIL.Image
    _PIL = True
except ImportError:
    _PIL = False



def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_BASE        = _base_dir()
_CONFIG_PATH = _BASE / "config" / "api_keys.json"


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config_key(key: str, value) -> None:
    try:
        cfg = _load_config()
        cfg[key] = value
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    except Exception as e:
        print(f"[Vision] [WARN] Could not save config key '{key}': {e}")


def _get_api_key() -> str:
    cfg = _load_config()
    return cfg.get("nvidia_api_key", "")


def _get_os() -> str:
    return _load_config().get("os_system", "windows").lower()

_IMG_MAX_W = 1280
_IMG_MAX_H = 720
_JPEG_Q    = 82

_SYSTEM_PROMPT = (
    "You are JARVIS, Tony Stark's AI assistant. "
    "You are given an image from either the user's screen or their webcam. "
    "Analyze what you see with detail and intelligence. "
    "Describe objects, text, people, components, and their context clearly. "
    "For technical questions (circuits, code, hardware) give specific, expert answers. "
    "Be concise — 2-4 sentences — unless the question demands more detail. "
    "Speak directly to the user ('I can see...', 'You have...'). "
    "Address the user as 'sir' depending on the language they used."
)


def _compress(img_bytes: bytes, source_format: str = "PNG") -> tuple[bytes, str]:
    if not _PIL:
        return img_bytes, f"image/{source_format.lower()}"

    try:
        img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_Q, optimize=False)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        print(f"[Vision] ⚠️  Image compress failed: {e}")
        return img_bytes, f"image/{source_format.lower()}"

def _capture_screen() -> tuple[bytes, str]:
    if _MSS:
        try:
            with mss.mss() as sct:
                monitors = sct.monitors          # [0] = all combined, [1..n] = real screens
                target   = monitors[1] if len(monitors) > 1 else monitors[0]
                shot     = sct.grab(target)
                png      = mss.tools.to_png(shot.rgb, shot.size)
                return _compress(png, "PNG")
        except Exception as e:
            print(f"[Vision] [WARN] mss capture failed ({e}), falling back to PIL.ImageGrab...")

    if _PIL:
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab().convert("RGB")
            buf = io.BytesIO()
            img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.BILINEAR)
            img.save(buf, format="JPEG", quality=_JPEG_Q)
            return buf.getvalue(), "image/jpeg"
        except Exception as e:
            print(f"[Vision] [ERR] PIL ImageGrab capture failed: {e}")

    raise RuntimeError("No screen capture backend available (mss and PIL failed).")


def _cv2_backend() -> int:
    """Return the best OpenCV camera backend for the current OS."""
    if not _CV2:
        return 0
    os_name = _get_os()
    if os_name == "windows":
        return cv2.CAP_DSHOW    
    if os_name == "mac":
        return cv2.CAP_AVFOUNDATION  
    return cv2.CAP_ANY


def _probe_camera(index: int, backend: int, warmup: int = 5) -> bool:

    if not _CV2:
        return False
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return False
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    return bool(np.mean(frame) > 8)


def _detect_camera_index() -> int:

    backend = _cv2_backend()
    print("[Vision] Auto-detecting camera...")
    for idx in range(6):
        if _probe_camera(idx, backend):
            print(f"[Vision] [OK] Camera found at index {idx}")
            _save_config_key("camera_index", idx)
            return idx
        print(f"[Vision] [WARN] Camera index {idx}: no usable frame")

    print("[Vision] [WARN] No camera found -- defaulting to index 0")
    _save_config_key("camera_index", 0)
    return 0


def _get_camera_index() -> int:
    cfg = _load_config()
    if "camera_index" in cfg:
        return int(cfg["camera_index"])
    return _detect_camera_index()


def _capture_camera() -> tuple[bytes, str]:
    if not _CV2:
        raise RuntimeError("OpenCV (cv2) is not installed. Run: pip install opencv-python")

    try:
        from core.camera import get_camera_service
        cam = get_camera_service()
        frame, _ = cam.get_latest_frame()
        if frame is not None and frame.size > 0:
            if _PIL:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(rgb)
                img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.BILINEAR)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=_JPEG_Q)
                return buf.getvalue(), "image/jpeg"
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])
            return buf.tobytes(), "image/jpeg"
    except Exception as e:
        print(f"[Vision] CameraService snapshot fallback: {e}")

    index   = _get_camera_index()
    backend = _cv2_backend()
    cap     = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        raise RuntimeError(f"Camera index {index} could not be opened.")

    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Camera returned no frame.")

    if _PIL:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_Q)
        return buf.getvalue(), "image/jpeg"

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])
    return buf.tobytes(), "image/jpeg"

from core.llm_client import call_llm_vision


class _VisionSession:
    def __init__(self):
        self._player = None
        self._lock   = threading.Lock()
        self._active = False

    def start(self, player=None, timeout: float = 5.0) -> None:
        with self._lock:
            if player is not None:
                self._player = player
            self._active = True
        print("[Vision] [OK] NVIDIA Vision Session ready")

    def is_ready(self) -> bool:
        return self._active

    def analyze(self, image_bytes: bytes, mime_type: str, user_text: str) -> str:
        prompt = f"{_SYSTEM_PROMPT}\n\nUser Question: {user_text}"
        try:
            print(f"[Vision] [SEND] Calling NVIDIA Vision NIM ({len(image_bytes):,} bytes)...")
            t0 = time.time()
            result = call_llm_vision(image_bytes, prompt, mime_type=mime_type)
            print(f"[Vision] [RECV] Response in {time.time()-t0:.2f}s: {result[:80]}")

            if self._player:
                if hasattr(self._player, "write_log"):
                    self._player.write_log(f"Jarvis: {result}")
                if hasattr(self._player, "speak"):
                    self._player.speak(result)
                if hasattr(self._player, "stop_camera_stream"):
                    def _deferred_close():
                        time.sleep(2.5)
                        try:
                            self._player.stop_camera_stream()
                        except Exception:
                            pass
                    threading.Thread(target=_deferred_close, daemon=True).start()

            return result
        except Exception as e:
            err_msg = f"Vision analysis error: {e}"
            print(f"[Vision] [WARN] {err_msg}")
            if self._player and hasattr(self._player, "write_log"):
                self._player.write_log(f"ERR: {err_msg}")
            return err_msg


_session      = _VisionSession()
_session_lock = threading.Lock()
_session_up   = False


def _ensure_session(player=None) -> None:
    global _session_up
    with _session_lock:
        if not _session_up:
            _session.start(player=player)
            _session_up = True
        elif player is not None:
            _session._player = player


def screen_process(
    parameters:     dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:

    params    = parameters or {}
    user_text = (params.get("text") or params.get("user_text") or "").strip()
    angle     = params.get("angle", "screen").lower().strip()

    if not user_text:
        user_text = "Describe what you see with technical precision."

    print(f"[Vision] -> angle={angle!r}  question='{user_text[:80]}'")

    try:
        _ensure_session(player=player)
    except Exception as e:
        print(f"[Vision] [ERR] Could not start session: {e}")
        return f"Could not initialize vision session: {e}"

    try:
        if angle == "camera":
            image_bytes, mime_type = _capture_camera()
            print(f"[Vision] Camera captured: {len(image_bytes):,} bytes")
            if player and hasattr(player, "start_camera_stream"):
                try:
                    player.start_camera_stream()
                except Exception as _e:
                    print(f"[Vision] [WARN] Camera stream failed: {_e}")
            elif player and hasattr(player, "show_camera_frame"):
                try:
                    player.show_camera_frame(image_bytes)
                except Exception as _e:
                    print(f"[Vision] [WARN] Camera preview failed: {_e}")
        else:
            image_bytes, mime_type = _capture_screen()
            print(f"[Vision] Screen captured: {len(image_bytes):,} bytes")
    except Exception as e:
        print(f"[Vision] [ERR] Capture error: {e}")
        return f"Capture error: {e}"

    return _session.analyze(image_bytes, mime_type, user_text)


def warmup_session(player=None) -> None:
    try:
        _ensure_session(player=player)
    except Exception as e:
        print(f"[Vision] [WARN] Warmup failed: {e}")


if __name__ == "__main__":
    print("[TEST] screen_processor.py (NVIDIA NIM Vision)")
    print("=" * 52)
    mode = input("angle — screen / camera (default: screen): ").strip().lower() or "screen"
    q    = input("Question (Enter = default): ").strip() or "What do you see? Be brief."

    t0 = time.perf_counter()
    warmup_session()
    print(f"Session ready in {time.perf_counter()-t0:.2f}s\n")

    t1 = time.perf_counter()
    res = screen_process({"angle": mode, "text": q})
    print(f"Result in {time.perf_counter()-t1:.3f}s:\n{res}")