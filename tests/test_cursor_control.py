"""
Unit Tests for Phase 4: JARVIS Vision Cursor Control.

Covers:
  - Coordinate mapping, camera mirroring, and active box normalization
  - Multi-monitor virtual screen projection
  - One Euro Filter smoothing and dead zone jitter suppression
  - Anti-jump latching on hand entrance/exit
  - Pinch click vs drag state machine transitions
  - Mode gating and non-interference guarantees
"""
import math
import time
import unittest
from unittest.mock import MagicMock, patch

from core.hand_tracker import HandLandmark, HandInfo, LANDMARK_NAMES
from core.cursor_control import (
    ScreenBounds,
    PinchState,
    VisionCursorController,
    get_virtual_screen_bounds,
)


class TestCoordinateMapping(unittest.TestCase):
    """Test camera-to-screen geometry, mirroring, and boundary projection."""

    def setUp(self):
        self.bounds = ScreenBounds(origin_x=0, origin_y=0, width=1920, height=1080)
        self.controller = VisionCursorController(
            config={
                "enabled": True,
                "mirror_camera": True,
                "sensitivity_x": 1.0,
                "sensitivity_y": 1.0,
                "active_box": [0.20, 0.80, 0.20, 0.80],
                "dead_zone_px": 2.5,
            },
            screen_bounds=self.bounds,
        )

    def test_camera_mirroring(self):
        """Moving hand to the right in physical space (left in camera feed, x=0.2) moves cursor right."""
        # cam_x=0.20 with mirroring -> x_mirrored = 0.80
        # In active box [0.20, 0.80], 0.80 maps to u = 1.0 (right edge of screen: 1920)
        sx, sy = self.controller.map_camera_to_screen(cam_x=0.20, cam_y=0.50)
        self.assertAlmostEqual(sx, 1920.0, delta=2.0)
        self.assertAlmostEqual(sy, 540.0, delta=2.0)

    def test_center_mapping(self):
        """Center of active box (cam_x=0.5, cam_y=0.5) must map exactly to center of screen."""
        sx, sy = self.controller.map_camera_to_screen(cam_x=0.50, cam_y=0.50)
        self.assertAlmostEqual(sx, 960.0, delta=2.0)
        self.assertAlmostEqual(sy, 540.0, delta=2.0)

    def test_boundary_clamping(self):
        """Coordinates outside the active box must clamp safely to [0, width] and [0, height]."""
        # Far outside left (cam_x = 0.95 -> mirrored 0.05)
        sx_left, sy = self.controller.map_camera_to_screen(cam_x=0.95, cam_y=0.50)
        self.assertEqual(sx_left, 0.0)

        # Far outside top (cam_y = 0.05)
        _, sy_top = self.controller.map_camera_to_screen(cam_x=0.50, cam_y=0.05)
        self.assertEqual(sy_top, 0.0)


class TestMultiMonitorSupport(unittest.TestCase):
    """Test coordinate projection across dual monitor layouts."""

    def test_dual_monitor_left_extended(self):
        """Primary monitor (1920x1080) at (0, 0) + Secondary monitor (1920x1080) at (-1920, 0)."""
        dual_bounds = ScreenBounds(origin_x=-1920, origin_y=0, width=3840, height=1080)
        controller = VisionCursorController(
            config={
                "enabled": True,
                "mirror_camera": False,
                "sensitivity_x": 1.0,
                "sensitivity_y": 1.0,
                "active_box": [0.0, 1.0, 0.0, 1.0],
            },
            screen_bounds=dual_bounds,
        )
        # Leftmost position (x=0) -> -1920
        sx_left, _ = controller.map_camera_to_screen(0.0, 0.5)
        self.assertAlmostEqual(sx_left, -1920.0)

        # Center position (x=0.5) -> 0 (boundary between monitors)
        sx_mid, _ = controller.map_camera_to_screen(0.5, 0.5)
        self.assertAlmostEqual(sx_mid, 0.0)

        # Rightmost position (x=1.0) -> +1920
        sx_right, _ = controller.map_camera_to_screen(1.0, 0.5)
        self.assertAlmostEqual(sx_right, 1920.0)


class TestJitterAndDeadZone(unittest.TestCase):
    """Test One Euro Filter smoothing and dead zone threshold."""

    @staticmethod
    def _create_hand(index_x: float, index_y: float, pinch_dist: float = 0.15) -> HandInfo:
        lms = []
        for i in range(21):
            lms.append(HandLandmark(i, LANDMARK_NAMES[i], index_x, index_y, 0.0, int(index_x*640), int(index_y*480)))
        # Landmark 4: Thumb tip
        lms[4] = HandLandmark(4, "THUMB_TIP", index_x - pinch_dist, index_y, 0.0, int((index_x - pinch_dist)*640), int(index_y*480))
        # Landmark 8: Index tip
        lms[8] = HandLandmark(8, "INDEX_TIP", index_x, index_y, 0.0, int(index_x*640), int(index_y*480))
        wrist = lms[0]
        palm = HandLandmark(-1, "PALM_CENTER", index_x, index_y + 0.1, 0.0, int(index_x*640), int((index_y+0.1)*480))
        return HandInfo(
            handedness="Right",
            confidence=0.95,
            wrist=wrist,
            palm_center=palm,
            thumb_tip=lms[4],
            index_tip=lms[8],
            middle_tip=lms[12],
            ring_tip=lms[16],
            pinky_tip=lms[20],
            landmarks=lms,
            fingers_extended={"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False},
        )

    @patch("core.cursor_control._set_os_cursor_pos")
    def test_dead_zone_locks_cursor(self, mock_set_cursor):
        """Micro-movements smaller than dead_zone_px (2.5px) must not move cursor."""
        controller = VisionCursorController(
            config={
                "enabled": True,
                "dead_zone_px": 3.0,
                "active_box": [0.0, 1.0, 0.0, 1.0],
                "sensitivity_x": 1.0,
                "sensitivity_y": 1.0,
            },
            screen_bounds=ScreenBounds(0, 0, 1000, 1000),
        )
        controller.enable_pointing_mode()

        t0 = 100.0
        h1 = self._create_hand(0.50, 0.50)
        controller.update(h1, timestamp=t0)
        init_x = controller._last_screen_x
        init_y = controller._last_screen_y

        # Move by 0.001 in normalized space (approx 1 pixel on 1000px screen) < 3px dead zone
        h_tiny = self._create_hand(0.501, 0.50)
        controller.update(h_tiny, timestamp=t0 + 0.033)

        # Position should remain locked at initial coordinates
        self.assertEqual(controller._last_screen_x, init_x)
        self.assertEqual(controller._last_screen_y, init_y)

    @patch("core.cursor_control._set_os_cursor_pos")
    def test_anti_jump_latching_on_entrance(self, mock_set_cursor):
        """When hand enters or mode activates, filter initializes at target without jump."""
        controller = VisionCursorController(screen_bounds=ScreenBounds(0, 0, 1920, 1080))
        controller.enable_pointing_mode()

        hand = self._create_hand(0.60, 0.40)
        pos = controller.update(hand, timestamp=100.0)
        self.assertIsNotNone(pos)
        self.assertTrue(controller._has_previous_pos)
        self.assertIsNotNone(controller._filter_x)

        # Simulate hand leaving frame
        controller.update(None, timestamp=101.0)
        self.assertFalse(controller._has_previous_pos)
        self.assertIsNone(controller._filter_x)

        # Hand re-enters at new position: cleanly initializes without jumping from old position
        hand2 = self._create_hand(0.30, 0.70)
        pos2 = controller.update(hand2, timestamp=102.0)
        self.assertIsNotNone(pos2)
        self.assertTrue(controller._has_previous_pos)


class TestPinchClickAndDrag(unittest.TestCase):
    """Test pinch-click and pinch-drag state machine."""

    @patch("core.cursor_control._send_mouse_down")
    @patch("core.cursor_control._send_mouse_up")
    @patch("core.cursor_control._send_mouse_click")
    @patch("core.cursor_control._set_os_cursor_pos")
    def test_quick_pinch_triggers_single_click(self, mock_set_pos, mock_click, mock_up, mock_down):
        """Pinch held for < 0.25s then released must trigger a single click without drag."""
        controller = VisionCursorController(
            config={"drag_hold_threshold": 0.25, "pinch_click_distance": 0.045},
            screen_bounds=ScreenBounds(0, 0, 1920, 1080),
        )
        controller.enable_pointing_mode()

        t0 = 100.0
        # 1. Normal pointing (pinch_dist = 0.10)
        h_point = TestJitterAndDeadZone._create_hand(0.50, 0.50, pinch_dist=0.10)
        controller.update(h_point, timestamp=t0)
        self.assertEqual(controller.pinch_state, PinchState.RELEASED)

        # 2. Pinch starts (pinch_dist = 0.03 <= 0.045)
        h_pinch = TestJitterAndDeadZone._create_hand(0.50, 0.50, pinch_dist=0.03)
        controller.update(h_pinch, timestamp=t0 + 0.033)
        self.assertEqual(controller.pinch_state, PinchState.PINCHING)
        mock_down.assert_not_called()

        # 3. Pinch released 0.10s later (< 0.25s drag threshold)
        controller.update(h_point, timestamp=t0 + 0.133)
        self.assertEqual(controller.pinch_state, PinchState.RELEASED)
        mock_click.assert_called_once()
        mock_down.assert_not_called()
        mock_up.assert_not_called()

    @patch("core.cursor_control._send_mouse_down")
    @patch("core.cursor_control._send_mouse_up")
    @patch("core.cursor_control._send_mouse_click")
    @patch("core.cursor_control._set_os_cursor_pos")
    def test_held_pinch_triggers_drag_mode(self, mock_set_pos, mock_click, mock_up, mock_down):
        """Pinch held for > 0.25s must engage drag mode and release on unpinch."""
        controller = VisionCursorController(
            config={"drag_hold_threshold": 0.25, "pinch_click_distance": 0.045},
            screen_bounds=ScreenBounds(0, 0, 1920, 1080),
        )
        controller.enable_pointing_mode()

        t0 = 100.0
        h_point = TestJitterAndDeadZone._create_hand(0.50, 0.50, pinch_dist=0.10)
        h_pinch = TestJitterAndDeadZone._create_hand(0.50, 0.50, pinch_dist=0.03)

        # 1. Pinch starts
        controller.update(h_pinch, timestamp=t0)
        self.assertEqual(controller.pinch_state, PinchState.PINCHING)

        # 2. Held past 0.25s (e.g. t = t0 + 0.30)
        controller.update(h_pinch, timestamp=t0 + 0.30)
        self.assertEqual(controller.pinch_state, PinchState.DRAGGING)
        mock_down.assert_called_once()

        # 3. Pinch released
        controller.update(h_point, timestamp=t0 + 0.50)
        self.assertEqual(controller.pinch_state, PinchState.RELEASED)
        mock_up.assert_called_once()
        mock_click.assert_not_called()


class TestModeGatingAndNonInterference(unittest.TestCase):
    """Test that cursor is not controlled unless pointing mode is explicitly active."""

    @patch("core.cursor_control._set_os_cursor_pos")
    def test_zero_interference_when_mode_inactive(self, mock_set_pos):
        controller = VisionCursorController()
        self.assertFalse(controller.is_active)

        hand = TestJitterAndDeadZone._create_hand(0.5, 0.5)
        res = controller.update(hand)
        self.assertIsNone(res, "When pointing mode is inactive, update() must return None")
        mock_set_pos.assert_not_called()

    @patch("core.cursor_control._send_mouse_up")
    def test_disabling_mode_releases_active_drag(self, mock_mouse_up):
        controller = VisionCursorController()
        controller.enable_pointing_mode()
        controller._pinch_state = PinchState.DRAGGING

        controller.disable_pointing_mode()
        self.assertFalse(controller.is_active)
        mock_mouse_up.assert_called_once()


if __name__ == "__main__":
    unittest.main()
