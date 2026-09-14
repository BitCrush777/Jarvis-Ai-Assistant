"""
Unit Tests for Dynamic Multi-Monitor Window Management Engine.

Covers:
  - Dynamic display enumeration and geometric arrangement
  - Multi-monitor count and primary monitor identification
  - Window rect and monitor matching
  - Proportional coordinate calculation across displays
  - Single monitor system fallback
  - Cycle next and prev monitor repositioning
"""
import unittest
from unittest.mock import MagicMock, patch

from core.window_manager import (
    MonitorInfo,
    WindowRect,
    MonitorManager,
    get_monitor_manager,
)


class TestWindowManager(unittest.TestCase):
    """Test monitor discovery, geometric mapping, and proportional window repositioning."""

    def setUp(self):
        self.mm = MonitorManager()
        # Mock dual monitor configuration:
        # Monitor 0 (Left/Primary): 0,0 to 1920,1080 (work: 0,0 to 1920,1040)
        # Monitor 1 (Right): 1920,0 to 3840,1080 (work: 1920,0 to 3840,1040)
        self.mon0 = MonitorInfo(
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
        self.mon1 = MonitorInfo(
            index=1,
            name="DISPLAY2",
            is_primary=False,
            left=1920,
            top=0,
            right=3840,
            bottom=1080,
            width=1920,
            height=1080,
            work_left=1920,
            work_top=0,
            work_right=3840,
            work_bottom=1040,
            work_width=1920,
            work_height=1040,
        )
        self.mm._cached_monitors = [self.mon0, self.mon1]
        self.mm.get_monitors = MagicMock(return_value=[self.mon0, self.mon1])

    def test_multi_monitor_count_and_primary(self):
        self.assertEqual(self.mm.monitor_count, 2)
        pri = self.mm.get_primary_monitor()
        self.assertIsNotNone(pri)
        self.assertTrue(pri.is_primary)
        self.assertEqual(pri.left, 0)

    def test_single_monitor_fallback(self):
        """When only 1 monitor is present, cross-monitor movement safely returns False."""
        self.mm.get_monitors = MagicMock(return_value=[self.mon0])
        self.mm.get_active_window = MagicMock(
            return_value=WindowRect(100, "Test App", 100, 100, 900, 700, 800, 600)
        )
        success = self.mm.move_active_window_to_monitor(direction="next")
        self.assertFalse(success)

    @patch("core.window_manager.user32")
    def test_move_window_to_next_monitor_proportional(self, mock_user32):
        """Moving window from Monitor 0 (left) to Monitor 1 (right) scales work coordinates proportionally."""
        # Window centered on Monitor 0: left=480, top=260, width=960, height=520
        # rel_x = 480/1920 = 0.25, rel_y = 260/1040 = 0.25
        active_win = WindowRect(
            hwnd=12345,
            title="Code Editor",
            left=480,
            top=260,
            right=1440,
            bottom=780,
            width=960,
            height=520,
        )
        self.mm.get_active_window = MagicMock(return_value=active_win)
        self.mm.get_monitor_for_window = MagicMock(return_value=self.mon0)
        self.mm.get_window_rect = MagicMock(
            return_value=WindowRect(12345, "Code Editor", 2400, 260, 3360, 780, 960, 520)
        )
        mock_user32.SetWindowPos = MagicMock(return_value=1)

        success = self.mm.move_active_window_to_monitor(direction="next")
        self.assertTrue(success)

        # Expected destination: Monitor 1 work area (left=1920) + 0.25*1920 (480) = 2400
        mock_user32.SetWindowPos.assert_called_once()
        args, _ = mock_user32.SetWindowPos.call_args
        hwnd_call, _, new_x, new_y, new_w, new_h, flags = args
        self.assertEqual(hwnd_call, 12345)
        self.assertEqual(new_x, 2400)
        self.assertEqual(new_y, 260)
        self.assertEqual(new_w, 960)
        self.assertEqual(new_h, 520)

    @patch("core.window_manager.user32")
    def test_move_window_prev_cycle(self, mock_user32):
        """Moving prev from Monitor 0 cycles to Monitor 1."""
        active_win = WindowRect(
            hwnd=555,
            title="Browser",
            left=100,
            top=100,
            right=900,
            bottom=700,
            width=800,
            height=600,
        )
        self.mm.get_active_window = MagicMock(return_value=active_win)
        self.mm.get_monitor_for_window = MagicMock(return_value=self.mon0)
        self.mm.get_window_rect = MagicMock(return_value=active_win)
        mock_user32.SetWindowPos = MagicMock(return_value=1)

        success = self.mm.move_active_window_to_monitor(direction="prev")
        self.assertTrue(success)
        mock_user32.SetWindowPos.assert_called_once()


if __name__ == "__main__":
    unittest.main()
