import unittest
from unittest.mock import patch

import cv2
import numpy as np

from training_monitor import TrainingCharts, TrainingMonitor, with_status_bar


class MonitorTests(unittest.TestCase):
    def test_status_never_overwrites_content(self):
        content = np.random.default_rng(7).integers(0, 256, (630, 480, 3), dtype=np.uint8)
        original = content.copy()
        output = with_status_bar(content, ['Decision 10,000,000', 'Episode 9,999', 'Score 100'])
        self.assertGreater(output.shape[0], content.shape[0])
        np.testing.assert_array_equal(output[-630:], original)
        np.testing.assert_array_equal(content, original)

    def test_empty_and_populated_charts(self):
        charts = TrainingCharts()
        charts.update([], [], [], [], [])
        self.assertTrue(all(label.get_visible() for label in charts.placeholders))
        charts.update([1, 3, 5], [100, 150, 120], [1, 69], [0.5, 0.25], [1.2, 1.8])
        np.testing.assert_array_equal(charts.mean_line.get_ydata(), [1, 2, 3])
        self.assertTrue(all(not label.get_visible() for label in charts.placeholders))
        for image in charts.images():
            self.assertEqual(image.shape, (600, 600, 3))

    def test_rolling_mean_and_live_history_limit(self):
        charts = TrainingCharts()
        charts.update(list(range(30)), [10] * 30, [1, 2, 3], [0.3, 0.2, 0.1], [1, 2, 3], max_points=2)
        np.testing.assert_array_equal(charts.score_line.get_xdata(), [29, 30])
        np.testing.assert_array_equal(charts.mean_line.get_ydata(), [18.5, 19.5])
        np.testing.assert_array_equal(charts.q_line.get_xdata(), [2, 3])

    def test_close_button_stops_and_destroys_only_owned_windows(self):
        monitor = TrainingMonitor()
        monitor.opened = list(monitor.names)
        monitor.ready = True
        with patch('training_monitor.cv2.waitKey', return_value=-1), \
             patch('training_monitor.cv2.getWindowProperty', side_effect=[1, 0, 1]), \
             patch('training_monitor.cv2.destroyWindow') as destroy:
            self.assertTrue(monitor.poll_stop())
            monitor.close()
            self.assertEqual(destroy.call_count, 3)
            self.assertEqual(monitor.opened, [])

    def test_escape_stops_even_between_redraws(self):
        monitor = TrainingMonitor()
        with patch('training_monitor.cv2.waitKey', return_value=27):
            self.assertTrue(monitor.poll_stop())

    def test_initial_visibility_does_not_count_as_close(self):
        monitor = TrainingMonitor()
        monitor.opened = list(monitor.names)
        with patch('training_monitor.cv2.waitKey', return_value=-1), \
             patch('training_monitor.cv2.getWindowProperty', return_value=0) as visibility:
            self.assertFalse(monitor.poll_stop())
            visibility.assert_not_called()

    def test_startup_waits_for_all_windows(self):
        monitor = TrainingMonitor()
        monitor.opened = list(monitor.names)
        monitor.shown = True
        with patch('training_monitor.cv2.waitKey', return_value=-1), \
             patch('training_monitor.cv2.getWindowProperty', side_effect=[0, 0, 0, 1, 1, 1]):
            self.assertTrue(monitor.wait_until_visible())
        self.assertTrue(monitor.ready)

    def test_startup_timeout_reports_missing_windows(self):
        monitor = TrainingMonitor()
        monitor.opened = list(monitor.names)
        monitor.shown = True
        with patch('training_monitor.cv2.waitKey', return_value=-1), \
             patch('training_monitor.cv2.getWindowProperty', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'Training has not started'):
                monitor.wait_until_visible(timeout=0)
        self.assertFalse(monitor.ready)

    def test_gui_error_is_not_silently_swallowed(self):
        monitor = TrainingMonitor()
        with patch('training_monitor.cv2.waitKey', side_effect=cv2.error('backend failed')):
            with self.assertRaisesRegex(RuntimeError, 'GUI error'):
                monitor.poll_stop()

    def test_open_failure_is_explicit_and_cleans_up(self):
        monitor = TrainingMonitor()
        with patch('training_monitor.cv2.namedWindow', side_effect=cv2.error('GUI missing')), \
             patch('training_monitor.cv2.waitKey', return_value=-1):
            with self.assertRaisesRegex(RuntimeError, 'local desktop kernel'):
                monitor.open()
        self.assertEqual(monitor.opened, [])


if __name__ == '__main__':
    unittest.main()
