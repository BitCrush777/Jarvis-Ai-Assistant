"""
Unit Tests for JARVIS Gesture-to-Computer Control Layer.

Covers:
  - Static pose recognition (OPEN_PALM, FIST, TWO_FINGER, PINCH, THUMBS_UP, THUMBS_DOWN)
  - Dynamic trajectory swipe recognition (SWIPE_LEFT, SWIPE_RIGHT, velocity & drift filtering, SwipeRecord)
  - Deterministic 7-State Lifecycle:
    DISABLED -> READY -> TRACKING -> GESTURE_DETECTED -> ACTION_EXECUTING -> COOLDOWN -> RESET -> READY
  - Stability debounce enforcement (single-frame noise rejection)
  - Single-fire enforcement (holding gesture produces exactly 1 action)
  - Neutral re-arming requirement (must return to OPEN_PALM before re-triggering)
  - Safe actions whitelist validation and execution dispatch
  - Multi-monitor window movement dispatch integration
  - Native Win32 dispatch acceleration
"""
import math
import time
import unittest
from unittest.mock import MagicMock, patch

from core.hand_tracker import HandLandmark, HandInfo, LANDMARK_NAMES
from core.gesture_control import (
    GestureType,
    GestureState,
    GestureEvent,
    GestureRecognizer,
    SafeActionDispatcher,
    GestureController,
    get_gesture_controller,
)


class TestGestureRecognizer(unittest.TestCase):
    """Test geometric pose and trajectory recognition."""

    @staticmethod
    def _create_synthetic_hand(
        pose: str,
        palm_x: float = 0.5,
        palm_y: float = 0.5,
        confidence: float = 0.90,
    ) -> HandInfo:
        """Create synthetic HandInfo matching target gesture pose."""
        lms = []
        wrist = HandLandmark(0, "WRIST", palm_x, palm_y + 0.2, 0.0, int(palm_x*640), int((palm_y+0.2)*480))
        lms.append(wrist)

        # Fingers extension defaults
        fe = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
        if pose == "OPEN_PALM":
            fe = {k: True for k in fe}
        elif pose == "TWO_FINGER":
            fe["index"] = True
            fe["middle"] = True
        elif pose in ("THUMBS_UP", "THUMBS_DOWN"):
            fe["thumb"] = True

        # Build Thumb (1..4)
        if pose == "THUMBS_UP":
            pts = [(palm_x - 0.05, palm_y + 0.1), (palm_x - 0.08, palm_y + 0.05), (palm_x - 0.10, palm_y), (palm_x - 0.12, palm_y - 0.10)]
        elif pose == "THUMBS_DOWN":
            pts = [(palm_x - 0.05, palm_y + 0.1), (palm_x - 0.08, palm_y + 0.15), (palm_x - 0.10, palm_y + 0.20), (palm_x - 0.12, palm_y + 0.30)]
        elif pose == "PINCH":
            # Thumb tip touches index tip at (0.48, 0.40)
            pts = [(palm_x - 0.05, palm_y + 0.1), (palm_x - 0.06, palm_y + 0.05), (palm_x - 0.04, palm_y + 0.02), (0.48, 0.40)]
        else:
            pts = [(palm_x - 0.05, palm_y + 0.1), (palm_x - 0.08, palm_y + 0.05), (palm_x - 0.10, palm_y), (palm_x - 0.12, palm_y - 0.05)]

        for i, (tx, ty) in enumerate(pts):
            lms.append(HandLandmark(i + 1, LANDMARK_NAMES[i + 1], tx, ty, 0.0, int(tx*640), int(ty*480)))

        # Build 4 fingers (5..20)
        finger_names = ["index", "middle", "ring", "pinky"]
        for f_idx, fname in enumerate(finger_names):
            base_idx = 5 + f_idx * 4
            x_col = palm_x + (f_idx - 1.5) * 0.06

            if pose == "PINCH" and fname == "index":
                # Index tip touches thumb tip at (0.48, 0.40)
                pip_y, dip_y, tip_y = 0.46, 0.43, 0.40
                tip_x = 0.48
            elif fe[fname]:
                pip_y, dip_y, tip_y = palm_y - 0.10, palm_y - 0.20, palm_y - 0.30
                tip_x = x_col
            else:
                pip_y, dip_y, tip_y = palm_y + 0.05, palm_y + 0.10, palm_y + 0.15
                tip_x = x_col

            lms.append(HandLandmark(base_idx, LANDMARK_NAMES[base_idx], x_col, palm_y, 0.0, int(x_col*640), int(palm_y*480)))
            lms.append(HandLandmark(base_idx+1, LANDMARK_NAMES[base_idx+1], x_col, pip_y, 0.0, int(x_col*640), int(pip_y*480)))
            lms.append(HandLandmark(base_idx+2, LANDMARK_NAMES[base_idx+2], x_col, dip_y, 0.0, int(x_col*640), int(dip_y*480)))
            lms.append(HandLandmark(base_idx+3, LANDMARK_NAMES[base_idx+3], tip_x, tip_y, 0.0, int(tip_x*640), int(tip_y*480)))

        palm_center = HandLandmark(-1, "PALM_CENTER", palm_x, palm_y, 0.0, int(palm_x*640), int(palm_y*480))
        return HandInfo(
            handedness="Right",
            confidence=confidence,
            wrist=wrist,
            palm_center=palm_center,
            thumb_tip=lms[4],
            index_tip=lms[8],
            middle_tip=lms[12],
            ring_tip=lms[16],
            pinky_tip=lms[20],
            landmarks=lms,
            fingers_extended=fe,
        )

    def test_open_palm_recognition(self):
        recognizer = GestureRecognizer()
        hand = self._create_synthetic_hand("OPEN_PALM")
        gesture, conf, _ = recognizer.recognize(hand)
        self.assertEqual(gesture, GestureType.OPEN_PALM)
        self.assertGreater(conf, 0.8)

    def test_fist_recognition(self):
        recognizer = GestureRecognizer()
        hand = self._create_synthetic_hand("FIST")
        gesture, conf, _ = recognizer.recognize(hand)
        self.assertEqual(gesture, GestureType.FIST)
        self.assertGreater(conf, 0.8)

    def test_two_finger_recognition(self):
        recognizer = GestureRecognizer()
        hand = self._create_synthetic_hand("TWO_FINGER")
        gesture, conf, _ = recognizer.recognize(hand)
        self.assertEqual(gesture, GestureType.TWO_FINGER)

    def test_pinch_recognition(self):
        recognizer = GestureRecognizer(pinch_distance_threshold=0.05)
        hand = self._create_synthetic_hand("PINCH")
        gesture, conf, _ = recognizer.recognize(hand)
        self.assertEqual(gesture, GestureType.PINCH)

    def test_thumbs_up_recognition(self):
        recognizer = GestureRecognizer()
        hand = self._create_synthetic_hand("THUMBS_UP")
        gesture, conf, _ = recognizer.recognize(hand)
        self.assertEqual(gesture, GestureType.THUMBS_UP)

    def test_thumbs_down_recognition(self):
        recognizer = GestureRecognizer()
        hand = self._create_synthetic_hand("THUMBS_DOWN")
        gesture, conf, _ = recognizer.recognize(hand)
        self.assertEqual(gesture, GestureType.THUMBS_DOWN)

    def test_swipe_right_detection_and_record(self):
        recognizer = GestureRecognizer(swipe_min_distance=0.15, swipe_min_velocity=0.40)
        t_base = 100.0
        for step in range(8):
            t = t_base + step * 0.033
            x = 0.20 + step * 0.045
            hand = self._create_synthetic_hand("OPEN_PALM", palm_x=x, palm_y=0.50)
            gesture, conf, _ = recognizer.recognize(hand, timestamp=t)
            if gesture == GestureType.SWIPE_RIGHT:
                break
        self.assertEqual(gesture, GestureType.SWIPE_RIGHT)
        self.assertIsNotNone(recognizer.last_swipe)
        self.assertEqual(recognizer.last_swipe.direction, "RIGHT")
        self.assertGreaterEqual(recognizer.last_swipe.displacement_x, 0.15)

    def test_swipe_left_detection_and_record(self):
        recognizer = GestureRecognizer(swipe_min_distance=0.15, swipe_min_velocity=0.40)
        t_base = 100.0
        for step in range(8):
            t = t_base + step * 0.033
            x = 0.70 - step * 0.045
            hand = self._create_synthetic_hand("OPEN_PALM", palm_x=x, palm_y=0.50)
            gesture, conf, _ = recognizer.recognize(hand, timestamp=t)
            if gesture == GestureType.SWIPE_LEFT:
                break
        self.assertEqual(gesture, GestureType.SWIPE_LEFT)
        self.assertIsNotNone(recognizer.last_swipe)
        self.assertEqual(recognizer.last_swipe.direction, "LEFT")
        self.assertLessEqual(recognizer.last_swipe.displacement_x, -0.15)

    def test_swipe_vertical_drift_rejection(self):
        recognizer = GestureRecognizer(swipe_min_distance=0.15, swipe_min_velocity=0.40)
        t_base = 100.0
        # Move diagonally (x from 0.20 to 0.50, y from 0.20 to 0.60 -> dy = 0.40 > 0.18 drift limit)
        for step in range(8):
            t = t_base + step * 0.033
            x = 0.20 + step * 0.045
            y = 0.20 + step * 0.060
            hand = self._create_synthetic_hand("OPEN_PALM", palm_x=x, palm_y=y)
            gesture, _, _ = recognizer.recognize(hand, timestamp=t)
        self.assertNotEqual(gesture, GestureType.SWIPE_RIGHT)


class TestAntiAccidentStateMachine(unittest.TestCase):
    """Test 7-State lifecycle, debounce, cooldown, and single-shot execution."""

    def setUp(self):
        self.controller = GestureController(
            config={
                "enabled": True,
                "confidence_threshold": 0.65,
                "stability_frames": 4,
                "cooldown_seconds": 0.6,
                "mappings": {
                    "FIST": "minimize_window",
                    "TWO_FINGER": "activate_window_mode",
                    "OPEN_PALM": "neutral",
                },
            },
            sync_dispatch=True,
        )
        self.controller.dispatcher.dispatch = MagicMock(return_value=True)

    def test_initial_state(self):
        self.assertEqual(self.controller.state, GestureState.READY)
        self.assertIn("READY", self.controller.latest_status)

    def test_disabled_gating(self):
        """When disabled, hand tracking produces zero actions and state is DISABLED."""
        self.controller.disable()
        self.assertEqual(self.controller.state, GestureState.DISABLED)
        fist_hand = TestGestureRecognizer._create_synthetic_hand("FIST")
        ev = self.controller.process_hand(fist_hand, timestamp=100.0)
        self.assertIsNone(ev)
        self.controller.dispatcher.dispatch.assert_not_called()

    def test_stability_debounce(self):
        """A static pose appearing for only 1, 2, or 3 frames must NOT trigger action dispatch."""
        fist_hand = TestGestureRecognizer._create_synthetic_hand("FIST")
        t0 = 100.0

        # Frame 1..3: State transitions to TRACKING, no action dispatched
        for i in range(3):
            ev = self.controller.process_hand(fist_hand, timestamp=t0 + i * 0.033)
            self.assertEqual(ev.state, GestureState.TRACKING)
            self.controller.dispatcher.dispatch.assert_not_called()

        # Frame 4: Reaches stability threshold -> enters COOLDOWN with action dispatched
        ev4 = self.controller.process_hand(fist_hand, timestamp=t0 + 3 * 0.033)
        self.assertEqual(ev4.state, GestureState.COOLDOWN)
        self.assertEqual(ev4.action, "minimize_window")
        self.controller.dispatcher.dispatch.assert_called_once_with("minimize_window")

    def test_single_fire_and_repetition_lock(self):
        """Holding FIST for 30 consecutive frames must fire EXACTLY once."""
        fist_hand = TestGestureRecognizer._create_synthetic_hand("FIST")
        t0 = 100.0

        for i in range(30):
            self.controller.process_hand(fist_hand, timestamp=t0 + i * 0.033)

        self.assertEqual(self.controller.dispatcher.dispatch.call_count, 1)

    def test_neutral_rearm_requirement(self):
        """
        After action execution and cooldown expiry, the user MUST return to OPEN_PALM
        before any subsequent action can trigger.
        """
        fist_hand = TestGestureRecognizer._create_synthetic_hand("FIST")
        palm_hand = TestGestureRecognizer._create_synthetic_hand("OPEN_PALM")
        t0 = 100.0

        # 1. Trigger first fist (4 frames)
        for i in range(4):
            self.controller.process_hand(fist_hand, timestamp=t0 + i * 0.033)
        self.assertEqual(self.controller.dispatcher.dispatch.call_count, 1)

        # 2. Advance time past cooldown (0.8s later), still holding FIST
        t_after_cooldown = t0 + 0.8
        for i in range(10):
            ev = self.controller.process_hand(fist_hand, timestamp=t_after_cooldown + i * 0.033)
            self.assertEqual(self.controller.dispatcher.dispatch.call_count, 1)
            self.assertEqual(ev.state, GestureState.RESET)

        # 3. Transition to OPEN_PALM (neutral reset)
        t_neutral = t_after_cooldown + 0.5
        ev_neutral = self.controller.process_hand(palm_hand, timestamp=t_neutral)
        self.assertFalse(self.controller._rearm_required)

        # 4. Now a second FIST can fire after 4 stable frames
        t_second_fist = t_neutral + 0.2
        for i in range(4):
            self.controller.process_hand(fist_hand, timestamp=t_second_fist + i * 0.033)
        self.assertEqual(self.controller.dispatcher.dispatch.call_count, 2)

    def test_window_mode_toggle(self):
        two_finger = TestGestureRecognizer._create_synthetic_hand("TWO_FINGER")
        t0 = 100.0
        for i in range(4):
            self.controller.process_hand(two_finger, timestamp=t0 + i * 0.033)
        self.assertTrue(self.controller.is_window_mode)


class TestSafeActionDispatcher(unittest.TestCase):
    """Test safety filters and dispatch logic."""

    def test_dangerous_actions_rejected(self):
        dispatcher = SafeActionDispatcher()
        dangerous_payloads = [
            "shutdown",
            "format_c",
            "rmdir",
            "powershell -c evil",
            "delete_all",
            "reboot",
        ]
        for dangerous in dangerous_payloads:
            result = dispatcher.dispatch(dangerous)
            self.assertFalse(result, f"Dangerous action '{dangerous}' MUST be rejected")

    @patch("core.gesture_control.pyautogui")
    def test_safe_actions_fallback(self, mock_pyautogui):
        dispatcher = SafeActionDispatcher(prefer_win32=False)
        dispatcher.dispatch("next_window")
        mock_pyautogui.hotkey.assert_called_with("alt", "tab")

        dispatcher.dispatch("prev_window")
        mock_pyautogui.hotkey.assert_called_with("alt", "shift", "tab")

        dispatcher.dispatch("click")
        mock_pyautogui.click.assert_called_once()

        dispatcher.dispatch("confirm")
        mock_pyautogui.press.assert_called_with("enter")

        dispatcher.dispatch("cancel")
        mock_pyautogui.press.assert_called_with("esc")

    @patch("core.gesture_control.get_monitor_manager")
    def test_multi_monitor_action_dispatch(self, mock_get_mm):
        mock_mm = MagicMock()
        mock_get_mm.return_value = mock_mm
        dispatcher = SafeActionDispatcher()

        dispatcher.dispatch("move_window_next_monitor")
        mock_mm.move_active_window_to_monitor.assert_called_with(direction="next")

        dispatcher.dispatch("move_window_prev_monitor")
        mock_mm.move_active_window_to_monitor.assert_called_with(direction="prev")


if __name__ == "__main__":
    unittest.main()
