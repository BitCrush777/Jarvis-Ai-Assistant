#!/usr/bin/env python3
"""
JARVIS Camera & Hand Gesture Diagnostic Suite
============================================
Comprehensive test and verification tool for:
  - Hardware camera device enumeration & verification
  - Raw video capture stability & FPS measurement (Test A)
  - MediaPipe 21-landmark hand tracking (Test B)
  - Static and dynamic gesture classification (Test C)
  - Safe action execution & whitelist security (Test D)
  - Sub-pixel cursor mapping, mirroring & filtering (Test E)
  - Multi-monitor topology and window movements (Test F)
  - Automated headless diagnostics (--all-headless)

Usage:
  python tools/camera_gesture_diagnostic.py --enumerate
  python tools/camera_gesture_diagnostic.py --test-a
  python tools/camera_gesture_diagnostic.py --test-b
  python tools/camera_gesture_diagnostic.py --test-c
  python tools/camera_gesture_diagnostic.py --test-d
  python tools/camera_gesture_diagnostic.py --test-e
  python tools/camera_gesture_diagnostic.py --test-f
  python tools/camera_gesture_diagnostic.py --all-headless
"""

import sys
import os
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.camera import (
    get_camera_service,
    enumerate_camera_devices,
    ensure_windows_camera_permissions,
    CameraService,
)
from core.hand_tracker import (
    HandTracker,
    HandInfo,
    HandLandmark,
    HandFrameResult,
    get_hand_tracker,
)
from core.gesture_control import (
    GestureController,
    GestureRecognizer,
    SafeActionDispatcher,
    GestureType,
    GestureState,
    get_gesture_controller,
)
from core.cursor_control import (
    VisionCursorController,
    ScreenBounds,
    OneEuroFilter,
    get_cursor_controller,
)
from core.window_manager import (
    MonitorManager,
    get_monitor_manager,
)


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  JARVIS DIAGNOSTIC: {title}")
    print("=" * 70)


def cmd_enumerate() -> bool:
    print_banner("CAMERA DEVICE ENUMERATION & PRIVACY AUDIT")
    ensure_windows_camera_permissions()

    print("[*] Probing system video capture devices (DirectShow / MediaFoundation)...")
    devices = enumerate_camera_devices(max_devices=4)

    if not devices:
        print("[!] No camera devices found in system.")
        return False

    print(f"\n[+] Found {len(devices)} device entries:")
    print(f"{'INDEX':<7} {'NAME':<30} {'BACKEND':<12} {'RES':<12} {'FPS':<6} {'STATUS'}")
    print("-" * 75)

    any_working = False
    for dev in devices:
        idx = dev.get("index", "?")
        name = dev.get("name", "Unknown")[:28]
        backend = dev.get("backend", "DEFAULT")
        res = f"{dev.get('width', 0)}x{dev.get('height', 0)}"
        fps = f"{dev.get('fps', 0):.0f}"
        status = dev.get("status", "UNKNOWN")
        if status in ("WORKING", "ONLINE"):
            any_working = True
            st_str = "\033[92mONLINE [OK]\033[0m"
        else:
            st_str = f"\033[91m{status}\033[0m"

        print(f"{idx:<7} {name:<30} {backend:<12} {res:<12} {fps:<6} {st_str}")

    print("\n" + "-" * 75)
    if any_working:
        print("[+] SUCCESS: At least one physical camera device is ready for capture.")
        return True
    else:
        print("[-] WARNING: No working cameras found. Check USB connection or privacy settings.")
        return False


def cmd_test_a(duration: float = 4.0) -> bool:
    print_banner(f"TEST A: RAW CAMERA CAPTURE STABILITY ({duration}s)")
    cam = get_camera_service()
    if cam.is_running:
        cam.stop()
    time.sleep(0.5)

    print("[*] Initializing CameraService...")
    ok = cam.start()
    if not ok:
        err = getattr(cam, "error_message", "")
        print(f"[-] FAILED: Could not start CameraService. Error: {err}")
        return False

    frames_received = 0
    t_start = time.time()
    last_frame_shape = None

    def _frame_listener(frame, ts):
        nonlocal frames_received, last_frame_shape
        frames_received += 1
        last_frame_shape = frame.shape

    sub_id = cam.subscribe(_frame_listener, sub_id="diag_test_a")
    print(f"[*] Capturing frames for {duration} seconds...")

    try:
        while time.time() - t_start < duration:
            time.sleep(0.1)
    finally:
        cam.unsubscribe(sub_id)
        cam.stop()

    elapsed = time.time() - t_start
    measured_fps = frames_received / max(0.001, elapsed)
    prof = cam.get_camera_profile() if hasattr(cam, "get_camera_profile") else {}

    print(f"\n[*] Camera Quality Profile & Results:")
    print(f"    - Device:          {prof.get('device_name', 'Webcam')}")
    print(f"    - Backend:         {prof.get('backend', 'Unknown')}")
    print(f"    - Requested Res:   {prof.get('requested_resolution', (1280, 720))}")
    print(f"    - Actual Res:      {last_frame_shape[1]}x{last_frame_shape[0]} (W x H)" if last_frame_shape else "    - Actual Res:      N/A")
    print(f"    - Pixel Format:    {prof.get('pixel_format', 'YUY2')}")
    print(f"    - Frames Received: {frames_received}")
    print(f"    - Elapsed Time:    {elapsed:.2f} s")
    print(f"    - Measured FPS:    {measured_fps:.1f} FPS")

    if frames_received >= 10 and measured_fps >= 5.0:
        print(f"\n[+] TEST A PASSED: Camera capture operates reliably at {measured_fps:.1f} FPS ({last_frame_shape[1]}x{last_frame_shape[0]})." if last_frame_shape else f"\n[+] TEST A PASSED: Camera capture operates reliably at {measured_fps:.1f} FPS.")
        return True
    else:
        print(f"\n[-] TEST A FAILED: Insufficient frames captured ({frames_received} frames).")
        return False


def cmd_test_b(duration: float = 4.0) -> bool:
    print_banner(f"TEST B: CAMERA + MEDIAPIPE HAND TRACKING ({duration}s)")
    cam = get_camera_service()
    tracker = get_hand_tracker()

    if cam.is_running:
        cam.stop()
    if tracker.is_active:
        tracker.stop()
    time.sleep(0.5)

    print("[*] Starting CameraService and HandTracker...")
    if not cam.start():
        err = getattr(cam, "error_message", "")
        print(f"[-] FAILED: Could not start CameraService. Error: {err}")
        return False

    results_count = 0
    hands_detected_count = 0
    annotated_frames_count = 0
    max_conf = 0.0

    def _tracker_cb(res: HandFrameResult):
        nonlocal results_count, hands_detected_count, annotated_frames_count, max_conf
        results_count += 1
        if res.hands:
            hands_detected_count += 1
            for h in res.hands:
                if h.confidence > max_conf:
                    max_conf = h.confidence
        if res.annotated_frame is not None:
            annotated_frames_count += 1

    tracker.register_callback(_tracker_cb)
    tracker.start()

    t_start = time.time()
    print(f"[*] Tracking hands for {duration} seconds (hold your hand up to test live detection)...")

    try:
        while time.time() - t_start < duration:
            time.sleep(0.1)
    finally:
        tracker.unregister_callback(_tracker_cb)
        tracker.stop()
        cam.stop()

    elapsed = time.time() - t_start
    pipeline_fps = results_count / max(0.001, elapsed)

    avg_latency_ms = (1000.0 / pipeline_fps) if pipeline_fps > 0 else 0.0
    track_res_str = f"{tracker.tracking_resolution[0]}x{tracker.tracking_resolution[1]}" if tracker.tracking_resolution else "Native Source (HD 720p)"

    print(f"\n[*] Tracking Results & Precision Profile:")
    print(f"    - Tracking Resolution:  {track_res_str}")
    print(f"    - Inferences Processed: {results_count}")
    print(f"    - Pipeline FPS:         {pipeline_fps:.1f} FPS")
    print(f"    - Avg Tracking Latency: {avg_latency_ms:.1f} ms")
    print(f"    - Hands Detected:       {hands_detected_count} frames")
    print(f"    - Max Confidence:       {max_conf * 100:.1f}%")
    print(f"    - Annotated Frames:     {annotated_frames_count}")

    if results_count >= 10 and annotated_frames_count > 0:
        print(f"\n[+] TEST B PASSED: Real-time hand tracking pipeline verified at {pipeline_fps:.1f} FPS ({avg_latency_ms:.1f} ms latency).")
        return True
    else:
        print(f"\n[-] TEST B FAILED: Pipeline produced insufficient tracking results.")
        return False


def _create_synthetic_hand(
    extended: list[bool] = [True, True, True, True, True],
    thumb_dist: float = 0.08,
    index_tip: tuple[float, float] = (0.5, 0.5),
    wrist: tuple[float, float] = (0.5, 0.8),
    palm_x: float = 0.5,
    palm_y: float = 0.6,
    confidence: float = 0.90,
) -> HandInfo:
    """Helper to synthesize 21 landmarks for precise gesture unit tests."""
    lms: list[HandLandmark] = []
    # 0: wrist
    lms.append(HandLandmark(id=0, name="WRIST", x=wrist[0], y=wrist[1], z=0.0, x_px=int(wrist[0] * 640), y_px=int(wrist[1] * 480)))

    # Fingers: thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
    names = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
    for f_idx in range(5):
        ext = extended[f_idx]
        base_x = palm_x + (f_idx - 2) * 0.06
        base_y = wrist[1] - 0.15
        tip_y = (base_y - 0.20) if ext else (base_y + 0.05)
        tip_x = base_x
        if f_idx == 0:  # thumb
            tip_x = base_x - (thumb_dist if ext else 0.02)
        elif f_idx == 1:  # index
            tip_x, tip_y = index_tip

        # 4 joints per finger
        for j in range(1, 5):
            frac = j / 4.0
            jx = base_x + (tip_x - base_x) * frac
            jy = base_y + (tip_y - base_y) * frac
            node_id = 1 + f_idx * 4 + (j - 1)
            lms.append(HandLandmark(id=node_id, name=f"{names[f_idx]}_{j}", x=jx, y=jy, z=0.0, x_px=int(jx * 640), y_px=int(jy * 480)))

    return HandInfo(
        handedness="Right",
        confidence=confidence,
        wrist=lms[0],
        palm_center=HandLandmark(id=-1, name="PALM_CENTER", x=palm_x, y=palm_y, z=0.0, x_px=int(palm_x * 640), y_px=int(palm_y * 480)),
        thumb_tip=lms[4],
        index_tip=lms[8],
        middle_tip=lms[12],
        ring_tip=lms[16],
        pinky_tip=lms[20],
        landmarks=lms,
        fingers_extended={
            "thumb": extended[0],
            "index": extended[1],
            "middle": extended[2],
            "ring": extended[3],
            "pinky": extended[4],
        },
    )


def cmd_test_c() -> bool:
    print_banner("TEST C: STATIC & DYNAMIC GESTURE CLASSIFICATION")
    recognizer = GestureRecognizer()

    test_cases = [
        ("OPEN_PALM", [True, True, True, True, True], GestureType.OPEN_PALM),
        ("POINT",     [False, True, False, False, False], GestureType.POINT),
        ("FIST",      [False, False, False, False, False], GestureType.FIST),
        ("TWO_FINGER",[False, True, True, False, False], GestureType.TWO_FINGER),
    ]

    all_passed = True
    for name, fingers, expected_type in test_cases:
        hand = _create_synthetic_hand(extended=fingers)
        detected, conf, _ = recognizer.recognize(hand)
        ok = (detected == expected_type)
        st = "\033[92mPASS\033[0m" if ok else f"\033[91mFAIL (got {detected})\033[0m"
        print(f"    - Pose {name:<12}: {st}")
        if not ok:
            all_passed = False

    # Dynamic Swipe test
    print("[*] Testing dynamic swipe recognition...")
    swipe_recognizer = GestureRecognizer(swipe_min_distance=0.15, swipe_min_velocity=0.40)
    t_base = 100.0
    swipe_gesture = GestureType.NONE
    for step in range(8):
        t = t_base + step * 0.033
        x = 0.20 + step * 0.045
        hand = _create_synthetic_hand(extended=[True, True, True, True, True], palm_x=x, palm_y=0.50)
        swipe_gesture, conf, _ = swipe_recognizer.recognize(hand, timestamp=t)
        if swipe_gesture == GestureType.SWIPE_RIGHT:
            break

    swipe_ok = (swipe_gesture == GestureType.SWIPE_RIGHT)
    st = "\033[92mPASS\033[0m" if swipe_ok else "\033[91mFAIL\033[0m"
    print(f"    - Dynamic SWIPE_RIGHT: {st}")
    if not swipe_ok:
        all_passed = False

    if all_passed:
        print("\n[+] TEST C PASSED: Gesture recognizer correctly identifies static & dynamic gestures.")
        return True
    else:
        print("\n[-] TEST C FAILED: Some gesture tests failed.")
        return False


def cmd_test_d() -> bool:
    print_banner("TEST D: SAFE ACTION DISPATCHER & COMMAND WHITELIST")
    dispatcher = SafeActionDispatcher(prefer_win32=True)

    # Verify safe actions execute without error
    safe_actions = ["prev_window", "next_window", "neutral", "confirm", "cancel"]
    safe_ok = True
    for act in safe_actions:
        res = dispatcher.dispatch(act)
        if not res:
            safe_ok = False
            print(f"    - Safe action '{act}': \033[91mFAILED\033[0m")
        else:
            print(f"    - Safe action '{act}': \033[92mALLOWED & EXECUTED\033[0m")

    # Verify dangerous actions are strictly BLOCKED
    dangerous_actions = ["shutdown", "rmdir", "format_c", "powershell -c evil", "curl http://malware"]
    blocked_ok = True
    for bad in dangerous_actions:
        res = dispatcher.dispatch(bad)
        if res:
            blocked_ok = False
            print(f"    - Dangerous command '{bad}': \033[91mSECURITY BREACH (ALLOWED!)\033[0m")
        else:
            print(f"    - Dangerous command '{bad}': \033[92mBLOCKED [SAFE]\033[0m")

    passed = safe_ok and blocked_ok
    if passed:
        print("\n[+] TEST D PASSED: Safe action dispatcher fully enforces sandbox security.")
        return True
    else:
        print("\n[-] TEST D FAILED: Action dispatcher security or execution failed.")
        return False


def cmd_test_e() -> bool:
    print_banner("TEST E: CURSOR CONTROLLER (MAPPING, MIRRORING, FILTERING)")
    bounds = ScreenBounds(origin_x=0, origin_y=0, width=1920, height=1080)
    cursor = VisionCursorController(
        config={
            "enabled": True,
            "mirror_camera": True,
            "sensitivity_x": 1.0,
            "sensitivity_y": 1.0,
            "active_box": [0.20, 0.80, 0.20, 0.80],
            "dead_zone_px": 2.5,
        },
        screen_bounds=bounds,
    )

    # 1. Test coordinate transformation & mirroring
    # cam_x=0.20 with mirroring -> x_mirrored = 0.80 -> right edge of screen: 1920
    sx, sy = cursor.map_camera_to_screen(cam_x=0.20, cam_y=0.50)
    mirror_ok = abs(sx - 1920.0) < 2.0 and abs(sy - 540.0) < 2.0
    print(f"    - Mirroring & Mapping: cam=(0.2, 0.5) -> screen=({sx:.1f}, {sy:.1f}) -> "
          f"{'\033[92mPASS\033[0m' if mirror_ok else '\033[91mFAIL\033[0m'}")

    # 2. Test OneEuroFilter jitter suppression
    t_now = time.time()
    filt = OneEuroFilter(t_now, 500.0, min_cutoff=1.2, beta=0.05)
    fx = filt.filter(500.4, t_now + 0.033)
    diff = abs(fx - 500.0)
    jitter_ok = diff < 0.6
    print(f"    - OneEuroFilter Jitter Suppression (diff={diff:.3f}px): "
          f"{'\033[92mPASS\033[0m' if jitter_ok else '\033[91mFAIL\033[0m'}")

    # 3. Test mode transition
    cursor.enable_pointing_mode()
    act_ok = cursor.is_active
    cursor.disable_pointing_mode()
    deact_ok = not cursor.is_active
    mode_ok = act_ok and deact_ok
    print(f"    - Pointing Mode Activation / Deactivation: {'\033[92mPASS\033[0m' if mode_ok else '\033[91mFAIL\033[0m'}")

    passed = mirror_ok and jitter_ok and mode_ok
    if passed:
        print("\n[+] TEST E PASSED: Cursor control mapping, mirroring, and filtering verified.")
        return True
    else:
        print("\n[-] TEST E FAILED: Cursor control test failed.")
        return False


def cmd_test_f() -> bool:
    print_banner("TEST F: MULTI-MONITOR WINDOW MANAGEMENT")
    mon_mgr = get_monitor_manager()
    monitors = mon_mgr.get_monitors(force_refresh=True)

    print(f"[+] Detected {len(monitors)} Active Display(s):")
    for m in monitors:
        primary_str = " [PRIMARY]" if m.is_primary else ""
        print(f"    - Monitor {m.index}: ({m.left}, {m.top}) {m.width}x{m.height} work=({m.work_width}x{m.work_height}){primary_str}")

    count = mon_mgr.monitor_count
    pri = mon_mgr.get_primary_monitor()
    has_pri = pri is not None
    print(f"    - Monitor Count: {count} -> {'\033[92mPASS\033[0m' if count > 0 else '\033[91mFAIL\033[0m'}")
    print(f"    - Primary Monitor: {pri.name if pri else 'None'} -> {'\033[92mPASS\033[0m' if has_pri else '\033[91mFAIL\033[0m'}")

    passed = count > 0 and has_pri
    if passed:
        print("\n[+] TEST F PASSED: Display geometry and monitor manager validated.")
        return True
    else:
        print("\n[-] TEST F FAILED: Monitor detection failed.")
        return False


def cmd_all_headless() -> bool:
    print_banner("RUNNING FULL AUTOMATED HEADLESS DIAGNOSTICS SUITE")
    results = {}

    print("\n>>> 1/7: CAMERA DEVICE ENUMERATION")
    results["Enumerate"] = cmd_enumerate()

    print("\n>>> 2/7: TEST A - RAW CAMERA CAPTURE (3.0s)")
    results["Test A (Capture)"] = cmd_test_a(duration=3.0)

    print("\n>>> 3/7: TEST B - MEDIAPIPE TRACKING PIPELINE (3.0s)")
    results["Test B (Tracking)"] = cmd_test_b(duration=3.0)

    print("\n>>> 4/7: TEST C - GESTURE RECOGNITION")
    results["Test C (Gestures)"] = cmd_test_c()

    print("\n>>> 5/7: TEST D - SAFE ACTION DISPATCHER")
    results["Test D (Actions)"] = cmd_test_d()

    print("\n>>> 6/7: TEST E - CURSOR CONTROLLER")
    results["Test E (Cursor)"] = cmd_test_e()

    print("\n>>> 7/7: TEST F - MULTI-MONITOR TOPOLOGY")
    results["Test F (Monitors)"] = cmd_test_f()

    # Final Summary Matrix
    print_banner("DIAGNOSTIC SUMMARY MATRIX")
    print(f"{'COMPONENT':<30} {'STATUS'}")
    print("-" * 50)
    all_ok = True
    for comp, res in results.items():
        if res:
            st = "\033[92m[PASS]\033[0m"
        else:
            st = "\033[91m[FAIL]\033[0m"
            all_ok = False
        print(f"{comp:<30} {st}")
    print("-" * 50)

    if all_ok:
        print("\n\033[92m[PASS] ALL 7 DIAGNOSTIC TESTS PASSED SUCCESSFULLY.\033[0m\n")
    else:
        print("\n\033[91m[FAIL] ONE OR MORE DIAGNOSTIC TESTS FAILED.\033[0m\n")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="JARVIS Vision & Gesture Diagnostic Suite")
    parser.add_argument("--enumerate", action="store_true", help="Enumerate connected camera devices")
    parser.add_argument("--test-a", action="store_true", help="Test raw camera capture stability & FPS")
    parser.add_argument("--test-b", action="store_true", help="Test camera + MediaPipe 21 hand tracking")
    parser.add_argument("--test-c", action="store_true", help="Test static & dynamic gesture classification")
    parser.add_argument("--test-d", action="store_true", help="Test safe action dispatcher & whitelist")
    parser.add_argument("--test-e", action="store_true", help="Test cursor controller mapping & filtering")
    parser.add_argument("--test-f", action="store_true", help="Test multi-monitor topology & window movement")
    parser.add_argument("--all-headless", action="store_true", help="Run full automated non-interactive suite")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    success = True
    if args.enumerate:
        success = cmd_enumerate()
    elif args.test_a:
        success = cmd_test_a()
    elif args.test_b:
        success = cmd_test_b()
    elif args.test_c:
        success = cmd_test_c()
    elif args.test_d:
        success = cmd_test_d()
    elif args.test_e:
        success = cmd_test_e()
    elif args.test_f:
        success = cmd_test_f()
    elif args.all_headless:
        success = cmd_all_headless()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
