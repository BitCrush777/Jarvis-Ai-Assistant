import unittest
import sys
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

import ui
from ui import _GestureHandVisualizer, _GestureControlPanel, MainWindow, HudCanvas
from core.gesture_control import get_gesture_controller


class TestGestureUI(unittest.TestCase):
    def setUp(self):
        self.visualizer = _GestureHandVisualizer()
        self.panel = _GestureControlPanel()
        self.panel.show()

    def test_visualizer_standby_and_tracking(self):
        self.visualizer.set_telemetry(hand_detected=False, landmarks=[], gesture='NONE')
        self.assertFalse(self.visualizer._hand_detected)
        self.assertEqual(len(self.visualizer._landmarks), 0)

        mock_lms = [(0.05 * i, 0.04 * i) for i in range(21)]
        self.visualizer.set_telemetry(
            hand_detected=True,
            landmarks=mock_lms,
            gesture='POINT',
            handedness='Right',
        )
        self.assertTrue(self.visualizer._hand_detected)
        self.assertEqual(len(self.visualizer._landmarks), 21)
        self.assertEqual(self.visualizer._gesture, 'POINT')
        self.assertEqual(self.visualizer._handedness, 'Right')
        self.assertGreater(self.visualizer._pulse_r, 0.0)

        self.visualizer._step_pulse()
        self.assertGreaterEqual(self.visualizer._pulse_r, 5.0)

    def test_panel_state_transitions(self):
        states_to_test = [
            ('DISABLED', False, '○ DISABLED'),
            ('READY', True, '● READY'),
            ('TRACKING', True, '● TRACKING'),
            ('GESTURE_DETECTED', True, '● ACTIVE'),
            ('ACTION_EXECUTING', True, '● EXECUTING'),
            ('COOLDOWN', True, '● COOLDOWN'),
            ('CAMERA_OFFLINE', True, '○ CAMERA OFFLINE'),
        ]

        for state_str, enabled, expected_text in states_to_test:
            self.panel.update_telemetry({
                'enabled': enabled,
                'state': state_str,
                'hand_detected': False,
                'fps': 0.0,
                'gesture': 'NONE',
                'confidence': 0.0,
            })
            self.assertEqual(self.panel._state_pill.text(), expected_text)

    def test_panel_telemetry_fields(self):
        self.panel.update_telemetry({
            'enabled': True,
            'state': 'TRACKING',
            'hand_detected': True,
            'handedness': 'Right',
            'confidence': 0.85,
            'fps': 29.8,
            'tracking_active': True,
            'gesture': 'POINT',
            'cursor_active': True,
            'cursor_pos': (1440, 900),
            'displays_count': 2,
            'landmarks': [(0.5, 0.5)] * 21,
        })

        self.assertIn('DETECTED (R)', self.panel._hand_lbl.text())
        self.assertIn('29.8 FPS', self.panel._fps_lbl.text())
        self.assertEqual(self.panel._gesture_lbl.text(), 'GESTURE: POINT')
        self.assertIn('85%', self.panel._conf_lbl.text())
        self.assertIn('ACTIVE (1440, 900)', self.panel._cursor_lbl.text())
        self.assertEqual(self.panel._displays_lbl.text(), 'DISPLAYS: 2 ACTIVE')

    def test_panel_action_feedback(self):
        self.panel.show_action_feedback('SWIPE RIGHT -> NEXT WINDOW')
        self.assertFalse(self.panel._action_badge.isHidden())
        self.assertIn('SWIPE RIGHT', self.panel._action_badge.text())

        self.panel.show_action_feedback('move_window_next_monitor', is_window_move=True, src_mon=1, dest_mon=2)
        self.assertIn('MONITOR 2', self.panel._action_badge.text())
        self.assertIn('[ 1 ] ➔ [ 2 ★ ]', self.panel._action_badge.text())

        self.panel._clear_action_feedback()
        self.assertTrue(self.panel._action_badge.isHidden())

    def test_panel_map_toggle(self):
        self.assertTrue(self.panel._map_drawer.isHidden())
        self.assertEqual(self.panel._map_btn.text(), '[?] MAP')

        self.panel._toggle_map()
        self.assertFalse(self.panel._map_drawer.isHidden())
        self.assertEqual(self.panel._map_btn.text(), '[X] MAP')

        self.panel._toggle_map()
        self.assertTrue(self.panel._map_drawer.isHidden())
        self.assertEqual(self.panel._map_btn.text(), '[?] MAP')

    def test_hud_canvas_gesture_pulse(self):
        hud = HudCanvas(face_path='')
        initial_pulses = len(hud._pulses)
        hud.gesture_pulse(1.5)
        self.assertEqual(len(hud._pulses), initial_pulses + 1)
        self.assertGreaterEqual(hud._tgt_scale, 1.03)

    def test_mainwindow_gesture_integration(self):
        win = MainWindow(face_path='')
        if hasattr(win, '_gesture_boot_timer'):
            win._gesture_boot_timer.stop()
        self.assertTrue(hasattr(win, '_gesture_panel'))

        win._on_gesture_action('SWIPE_RIGHT -> NEXT_WINDOW')
        self.assertFalse(win._gesture_panel._action_badge.isHidden())

        try:
            import unittest.mock as mock
            gc = get_gesture_controller()
            initial_enabled = gc.enabled
            with mock.patch("core.camera.get_camera_service"), mock.patch("core.hand_tracker.get_hand_tracker"):
                win._toggle_gesture_control()
                self.assertEqual(gc.enabled, not initial_enabled)
                win._toggle_gesture_control()
                self.assertEqual(gc.enabled, initial_enabled)
        finally:
            try:
                from core.camera import get_camera_service
                from core.hand_tracker import get_hand_tracker
                get_hand_tracker().stop()
                get_camera_service().stop()
            except Exception:
                pass


if __name__ == '__main__':
    unittest.main()
