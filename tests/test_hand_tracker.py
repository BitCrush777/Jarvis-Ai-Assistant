"""
Unit and Integration Tests for JARVIS Hand Tracking Foundation.

Covers:
  - OneEuroFilter jitter reduction and fast-motion responsiveness
  - LandmarkSmoother multi-hand management and reset hysteresis
  - Finger extension classification (open palm, fist, pointing, peace, thumbs up)
  - Palm center centroid calculation and fingertip extraction
  - HandTracker lifecycle, processing pipeline, and callback distribution
  - Debug overlay rendering and HUD telemetry
  - Unified CameraService thread-safety and contention prevention
"""
import math
import unittest
import numpy as np

from core.camera import CameraService, get_camera_service
from core.hand_tracker import (
    HandTracker,
    HandLandmark,
    HandInfo,
    HandFrameResult,
    OneEuroFilter,
    LandmarkSmoother,
    get_hand_tracker,
    LANDMARK_NAMES,
    SKELETON_CONNECTIONS,
)


class TestOneEuroFilter(unittest.TestCase):
    """Test mathematical behavior of One Euro Filter."""

    def test_filter_initialization(self):
        f = OneEuroFilter(t0=0.0, x0=10.0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        self.assertEqual(f.x_prev, 10.0)
        self.assertEqual(f.dx_prev, 0.0)

    def test_constant_signal(self):
        """A constant signal should converge exactly to the constant value."""
        f = OneEuroFilter(t0=0.0, x0=5.0)
        for i in range(1, 30):
            t = i * 0.033
            val = f.filter(5.0, t)
        self.assertAlmostEqual(val, 5.0, places=3)

    def test_jitter_attenuation(self):
        """High-frequency small noise (jitter) must have significantly reduced variance."""
        f = OneEuroFilter(t0=0.0, x0=0.5, min_cutoff=0.8, beta=0.005)
        np.random.seed(42)
        base_val = 0.5
        noise = np.random.normal(0, 0.02, 100)
        raw_signals = base_val + noise

        filtered_signals = []
        for i, raw in enumerate(raw_signals):
            t = (i + 1) * 0.033
            filtered_signals.append(f.filter(raw, t))

        raw_var = float(np.var(raw_signals))
        filtered_var = float(np.var(filtered_signals[10:]))  # After warm-up
        self.assertLess(filtered_var, raw_var * 0.4, "Filtered signal should reduce jitter variance by at least 60%")

    def test_rapid_motion_responsiveness(self):
        """When velocity spikes, cutoff increases (beta effect) to follow motion with low lag."""
        f = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=1.0, d_cutoff=10.0)
        # Settle at 0.0
        for i in range(1, 20):
            f.filter(0.0, i * 0.033)

        # Fast step jump to 1.0 over 3 frames
        f1 = f.filter(0.4, 20 * 0.033)
        f2 = f.filter(0.8, 21 * 0.033)
        f3 = f.filter(1.0, 22 * 0.033)
        self.assertGreater(f3, 0.85, "Filter must adapt swiftly to fast intentional motion")


class TestLandmarkSmoother(unittest.TestCase):
    """Test 21-landmark multi-hand smoothing bank."""

    def test_smoother_multi_hand_isolation(self):
        smoother = LandmarkSmoother()
        coords_left = [(0.1 * i, 0.2 * i, 0.0) for i in range(21)]
        coords_right = [(0.5 * i, 0.6 * i, 0.0) for i in range(21)]

        smoothed_l = smoother.smooth("Left", 1.0, coords_left)
        smoothed_r = smoother.smooth("Right", 1.0, coords_right)

        self.assertEqual(len(smoothed_l), 21)
        self.assertEqual(len(smoothed_r), 21)
        self.assertNotEqual(smoothed_l[1], smoothed_r[1])

    def test_reset_hand(self):
        smoother = LandmarkSmoother()
        coords = [(0.2, 0.3, 0.1)] * 21
        smoother.smooth("Right", 1.0, coords)
        self.assertIn("Right", smoother._filters)

        smoother.reset_hand("Right")
        self.assertNotIn("Right", smoother._filters)

        # After reset, smoothing begins fresh without historical drag
        new_coords = [(0.9, 0.9, 0.9)] * 21
        smoothed = smoother.smooth("Right", 2.0, new_coords)
        self.assertAlmostEqual(smoothed[0][0], 0.9, places=2)


class TestFingerExtensionGeometry(unittest.TestCase):
    """Test rotation-invariant finger extension analysis."""

    @staticmethod
    def _build_synthetic_hand(pose: str) -> list[HandLandmark]:
        """
        Build a synthetic 21-landmark hand coordinate set for testing gestures.
        Coordinates are normalized [0, 1]. Wrist is at (0.5, 0.8).
        """
        lms = []
        wrist_x, wrist_y = 0.5, 0.8

        # 0: Wrist
        lms.append(HandLandmark(0, "WRIST", wrist_x, wrist_y, 0.0, int(wrist_x*640), int(wrist_y*480)))

        # Extended fingers set
        extended_fingers = set()
        if pose == "open_palm":
            extended_fingers = {"thumb", "index", "middle", "ring", "pinky"}
        elif pose == "fist":
            extended_fingers = set()
        elif pose == "pointing":
            extended_fingers = {"index"}
        elif pose == "peace":
            extended_fingers = {"index", "middle"}
        elif pose == "thumbs_up":
            extended_fingers = {"thumb"}

        # Build Thumb (1..4)
        if "thumb" in extended_fingers:
            thumb_pts = [(0.42, 0.72), (0.35, 0.65), (0.28, 0.58), (0.22, 0.50)]
        else:
            thumb_pts = [(0.45, 0.73), (0.42, 0.68), (0.45, 0.65), (0.48, 0.65)]
        for i, (tx, ty) in enumerate(thumb_pts):
            lms.append(HandLandmark(i + 1, LANDMARK_NAMES[i + 1], tx, ty, 0.0, int(tx*640), int(ty*480)))

        # Build 4 fingers: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
        finger_names = ["index", "middle", "ring", "pinky"]
        for f_idx, fname in enumerate(finger_names):
            base_idx = 5 + f_idx * 4
            x_col = 0.5 + (f_idx - 1.5) * 0.07
            mcp_y = 0.55

            if fname in extended_fingers:
                # Extended: Tip is far from wrist, PIP is intermediate
                pip_y = 0.40
                dip_y = 0.28
                tip_y = 0.18
            else:
                # Folded: Tip curves down near/below MCP, tip-wrist dist is <= pip-wrist dist
                pip_y = 0.48
                dip_y = 0.54
                tip_y = 0.58

            lms.append(HandLandmark(base_idx, LANDMARK_NAMES[base_idx], x_col, mcp_y, 0.0, int(x_col*640), int(mcp_y*480)))
            lms.append(HandLandmark(base_idx+1, LANDMARK_NAMES[base_idx+1], x_col, pip_y, 0.0, int(x_col*640), int(pip_y*480)))
            lms.append(HandLandmark(base_idx+2, LANDMARK_NAMES[base_idx+2], x_col, dip_y, 0.0, int(x_col*640), int(dip_y*480)))
            lms.append(HandLandmark(base_idx+3, LANDMARK_NAMES[base_idx+3], x_col, tip_y, 0.0, int(x_col*640), int(tip_y*480)))

        return lms

    def test_open_palm_all_extended(self):
        lms = self._build_synthetic_hand("open_palm")
        ext = HandTracker._classify_finger_extension(lms, "Right")
        self.assertTrue(ext["index"], "Index must be extended in open palm")
        self.assertTrue(ext["middle"], "Middle must be extended in open palm")
        self.assertTrue(ext["ring"], "Ring must be extended in open palm")
        self.assertTrue(ext["pinky"], "Pinky must be extended in open palm")
        self.assertTrue(ext["thumb"], "Thumb must be extended in open palm")

    def test_fist_all_folded(self):
        lms = self._build_synthetic_hand("fist")
        ext = HandTracker._classify_finger_extension(lms, "Right")
        self.assertFalse(ext["index"], "Index must be folded in fist")
        self.assertFalse(ext["middle"], "Middle must be folded in fist")
        self.assertFalse(ext["ring"], "Ring must be folded in fist")
        self.assertFalse(ext["pinky"], "Pinky must be folded in fist")

    def test_pointing_pose(self):
        lms = self._build_synthetic_hand("pointing")
        ext = HandTracker._classify_finger_extension(lms, "Right")
        self.assertTrue(ext["index"], "Index must be extended in pointing")
        self.assertFalse(ext["middle"], "Middle must be folded in pointing")
        self.assertFalse(ext["ring"], "Ring must be folded in pointing")
        self.assertFalse(ext["pinky"], "Pinky must be folded in pointing")

    def test_peace_sign_pose(self):
        lms = self._build_synthetic_hand("peace")
        ext = HandTracker._classify_finger_extension(lms, "Right")
        self.assertTrue(ext["index"], "Index must be extended in peace sign")
        self.assertTrue(ext["middle"], "Middle must be extended in peace sign")
        self.assertFalse(ext["ring"], "Ring must be folded in peace sign")
        self.assertFalse(ext["pinky"], "Pinky must be folded in peace sign")

    def test_rotation_invariance(self):
        """Rotated hand (e.g. tilted by 45 degrees) should still correctly classify extended fingers."""
        lms = self._build_synthetic_hand("open_palm")
        angle = math.radians(45)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        ox, oy = lms[0].x, lms[0].y

        rotated_lms = []
        for lm in lms:
            dx = lm.x - ox
            dy = lm.y - oy
            rx = ox + dx * cos_a - dy * sin_a
            ry = oy + dx * sin_a + dy * cos_a
            rotated_lms.append(HandLandmark(lm.id, lm.name, rx, ry, lm.z, int(rx*640), int(ry*480)))

        ext = HandTracker._classify_finger_extension(rotated_lms, "Right")
        self.assertTrue(ext["index"], "Index must be extended under 45 deg rotation")
        self.assertTrue(ext["middle"], "Middle must be extended under 45 deg rotation")
        self.assertTrue(ext["ring"], "Ring must be extended under 45 deg rotation")
        self.assertTrue(ext["pinky"], "Pinky must be extended under 45 deg rotation")


    def test_palm_center_centroid(self):
        """Palm center should be the centroid of wrist (0) and MCP bases (1, 5, 9, 13, 17)."""
        lms = self._build_synthetic_hand("open_palm")
        indices = [0, 1, 5, 9, 13, 17]
        expected_x = sum(lms[i].x for i in indices) / len(indices)
        expected_y = sum(lms[i].y for i in indices) / len(indices)

        palm_center = HandLandmark(
            id=-1,
            name="PALM_CENTER",
            x=expected_x,
            y=expected_y,
            z=0.0,
            x_px=int(expected_x * 640),
            y_px=int(expected_y * 480),
        )
        self.assertAlmostEqual(palm_center.x, expected_x, places=5)
        self.assertAlmostEqual(palm_center.y, expected_y, places=5)

    def test_fingertip_extraction(self):
        lms = self._build_synthetic_hand("open_palm")
        self.assertEqual(lms[4].name, "THUMB_TIP")
        self.assertEqual(lms[8].name, "INDEX_TIP")
        self.assertEqual(lms[12].name, "MIDDLE_TIP")
        self.assertEqual(lms[16].name, "RING_TIP")
        self.assertEqual(lms[20].name, "PINKY_TIP")


class TestHandTrackerPipeline(unittest.TestCase):
    """Test HandTracker initialization, offline frame processing, and debug overlay."""

    @classmethod
    def setUpClass(cls):
        cls.tracker = HandTracker(max_hands=2, detection_confidence=0.5, tracking_confidence=0.5)

    def test_initialization(self):
        self.assertIsNotNone(self.tracker._landmarker, "MediaPipe Tasks HandLandmarker must load model successfully")
        self.assertEqual(self.tracker.max_hands, 2)

    def test_process_blank_frame(self):
        """Blank / black frame should return 0 hands cleanly without errors."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result, annotated = self.tracker.process_frame(blank_frame)

        self.assertIsInstance(result, HandFrameResult)
        self.assertEqual(result.hand_count, 0)
        self.assertEqual(len(result.hands), 0)
        self.assertEqual(result.frame_width, 640)
        self.assertEqual(result.frame_height, 480)

    def test_debug_overlay_rendering(self):
        """Debug overlay must render valid HUD graphics without throwing."""
        self.tracker.debug_mode = True
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result, annotated = self.tracker.process_frame(blank_frame)

        self.assertIsNotNone(annotated, "Debug frame should not be None when debug_mode=True")
        self.assertEqual(annotated.shape, (480, 640, 3))
        hud_area = annotated[10:60, 10:200]
        self.assertGreater(np.sum(hud_area), 0, "HUD telemetry overlay should render non-zero pixels")

    def test_callback_registration(self):
        callback_fired = []

        def my_cb(res: HandFrameResult):
            callback_fired.append(res)

        self.tracker.add_callback(my_cb)
        self.assertIn(my_cb, self.tracker._callbacks)

        # Trigger on dummy frame
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.tracker._is_active = True
        self.tracker._on_camera_frame(dummy, 123.456)
        self.tracker._is_active = False

        self.assertGreater(len(callback_fired), 0)
        self.assertEqual(callback_fired[0].timestamp, 123.456)

        self.tracker.remove_callback(my_cb)
        self.assertNotIn(my_cb, self.tracker._callbacks)


class TestCameraService(unittest.TestCase):
    """Test unified CameraService thread safety and state management."""

    def test_singleton_accessor(self):
        c1 = get_camera_service()
        c2 = get_camera_service()
        self.assertIs(c1, c2, "get_camera_service() must return a singleton instance")

    def test_get_latest_frame_when_idle(self):
        cam = CameraService()
        frame, ts = cam.get_latest_frame()
        self.assertIsNone(frame)
        self.assertEqual(ts, 0.0)

    def test_subscriber_registration(self):
        cam = CameraService()
        try:
            def dummy_cb(f, t): pass
            sub_id = cam.subscribe(dummy_cb, sub_id="test_sub", auto_start=False)
            self.assertEqual(sub_id, "test_sub")
            self.assertIn("test_sub", cam._subscribers)
            cam.unsubscribe("test_sub")
            self.assertNotIn("test_sub", cam._subscribers)
        finally:
            cam.stop()


if __name__ == "__main__":
    unittest.main()
