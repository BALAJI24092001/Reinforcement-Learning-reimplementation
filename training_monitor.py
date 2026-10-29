"""Responsive OpenCV popups with Matplotlib charts and separate status bars.

Matplotlib uses off-screen FigureCanvasAgg here, so Jupyter's inline backend does
not prevent native popup windows from opening. Only OpenCV owns desktop windows.
"""
import time

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


GAME_WINDOW = "Breakout - Gameplay"
PERFORMANCE_WINDOW = "Breakout - Episode performance"
LEARNING_WINDOW = "Breakout - Learning progress"


def with_status_bar(content, lines):
    """Prepend a dedicated header; never draw onto gameplay or plot pixels."""
    width = content.shape[1]
    header = np.full((18 + 27 * len(lines), width, 3), (35, 30, 25), np.uint8)
    for row, text in enumerate(lines):
        scale = 0.48
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if text_width > width - 24:
            scale *= (width - 24) / text_width
        cv2.putText(header, text, (12, 27 + 27 * row), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate((header, content), axis=0)


class TrainingCharts:
    """Two figures, two useful plots each; reused for live and final charts."""
    def __init__(self):
        self.performance = Figure(figsize=(6, 6), dpi=100, layout="constrained")
        self.learning = Figure(figsize=(6, 6), dpi=100, layout="constrained")
        self.performance_canvas = FigureCanvasAgg(self.performance)
        self.learning_canvas = FigureCanvasAgg(self.learning)
        self.score_ax, self.length_ax = self.performance.subplots(2, 1)
        self.loss_ax, self.q_ax = self.learning.subplots(2, 1)
        self.score_line, = self.score_ax.plot([], [], color="#8cb9dc", linewidth=1)
        self.mean_line, = self.score_ax.plot([], [], color="#dd7924", linewidth=2)
        self.length_line, = self.length_ax.plot([], [], color="#397aab", linewidth=1.4)
        self.loss_line, = self.loss_ax.plot([], [], color="#9c4c89", linewidth=1.4)
        self.q_line, = self.q_ax.plot([], [], color="#27836f", linewidth=1.4)
        self.score_ax.set(title="Episode score (blue) and rolling mean (orange)",
                          xlabel="Completed episode", ylabel="Breakout points")
        self.length_ax.set(title="Episode length", xlabel="Completed episode",
                           ylabel="Agent decisions")
        self.loss_ax.set(title="Mean squared TD loss", xlabel="Optimizer update",
                         ylabel="Loss")
        self.q_ax.set(title="Mean predicted Q for sampled actions", xlabel="Optimizer update",
                      ylabel="Q-value")
        self.axes = (self.score_ax, self.length_ax, self.loss_ax, self.q_ax)
        self.placeholders = []
        for ax in self.axes:
            ax.grid(True, alpha=0.22)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            self.placeholders.append(ax.text(
                0.5, 0.5, "Waiting for data", transform=ax.transAxes,
                ha="center", va="center", color="#777777",
            ))

    def update(self, returns, lengths, updates, losses, q_values, max_points=500):
        episode_start = max(0, len(returns) - max_points)
        episodes = np.arange(1, len(returns) + 1)
        self.score_line.set_data(episodes[episode_start:], returns[episode_start:])
        # A trailing mean of up to 20 episodes is available from the first episode.
        values = np.asarray(returns, dtype=float)
        if len(values):
            cumsum = np.concatenate(([0.0], np.cumsum(values)))
            ends = np.arange(1, len(values) + 1)
            starts = np.maximum(0, ends - 20)
            mean = (cumsum[ends] - cumsum[starts]) / (ends - starts)
            self.mean_line.set_data(episodes[episode_start:], mean[episode_start:])
        else:
            self.mean_line.set_data([], [])
        self.length_line.set_data(episodes[episode_start:], lengths[episode_start:])
        start = max(0, len(updates) - max_points)
        self.loss_line.set_data(updates[start:], losses[start:])
        self.q_line.set_data(updates[start:], q_values[start:])
        for ax, placeholder, has_data in zip(
            self.axes, self.placeholders, (bool(returns), bool(lengths), bool(updates), bool(updates))
        ):
            placeholder.set_visible(not has_data)
            ax.relim()
            ax.autoscale_view()
            ax.margins(x=0.04, y=0.15)
        self.score_ax.set_ylim(bottom=0)
        self.length_ax.set_ylim(bottom=0)
        self.loss_ax.set_ylim(bottom=0)

    def images(self):
        images = []
        for canvas in (self.performance_canvas, self.learning_canvas):
            canvas.draw()
            images.append(cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR))
        return images


class TrainingMonitor:
    def __init__(self, game_fps=15, graph_interval=1.0):
        self.charts = TrainingCharts()
        self.game_interval = 1.0 / game_fps
        self.graph_interval = graph_interval
        self.last_game = self.last_graph = float("-inf")
        self.names = (GAME_WINDOW, PERFORMANCE_WINDOW, LEARNING_WINDOW)
        self.opened = []
        self.stopped = False
        self.graph_images = None
        self.shown = False
        self.ready = False

    def open(self):
        try:
            # Fit all three windows to the Windows work area where available.
            width, height = 1600, 900
            left, top = 0, 0
            try:
                import ctypes
                from ctypes.wintypes import RECT
                area = RECT()
                if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(area), 0):
                    left, top = area.left, area.top
                    width, height = area.right - area.left, area.bottom - area.top
            except (AttributeError, OSError):
                pass
            scale = min(1.0, (width - 50) / 1680, (height - 90) / 756)
            x = left + 10
            for name, content_width in zip(self.names, (480, 600, 600)):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                self.opened.append(name)
                window_width = int(content_width * scale)
                cv2.resizeWindow(name, window_width, int(738 * scale))
                cv2.moveWindow(name, x, top + 10)
                x += window_width + 10
        except cv2.error as exc:
            self.close()
            raise RuntimeError(
                "Desktop monitor windows could not open. Run the notebook in a local "
                "desktop kernel with opencv-python (not opencv-python-headless), then "
                "restart the kernel. Set SHOW_LIVE_WINDOWS=False only for a headless run."
            ) from exc

    def refresh(self, render_frame, *, step, total_steps, episode, score, episode_length,
                epsilon, updates, replay_size, replay_capacity, elapsed,
                returns, lengths, optimization_steps, losses, q_values, force=False):
        if self.stopped:
            return False
        # Pump native events on EVERY call, even when no image needs repainting.
        if self.shown and self.poll_stop():
            return False
        now = time.monotonic()
        if force or now - self.last_game >= self.game_interval:
            frame = render_frame()
            content = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (480, 630),
                                 interpolation=cv2.INTER_NEAREST)
            self.game_image = with_status_bar(content, [
                f"Decision {step:,} / {total_steps:,}",
                f"Episode {episode:,} | Score {score:g} | Length {episode_length:,}",
                f"Epsilon {epsilon:.3f} | Q / Esc or close a window to stop",
            ])
            cv2.imshow(GAME_WINDOW, self.game_image)
            self.last_game = now
        if force or now - self.last_graph >= self.graph_interval:
            self.charts.update(returns, lengths, optimization_steps, losses, q_values)
            performance, learning = self.charts.images()
            self.graph_images = (
                with_status_bar(performance, [
                    f"Completed episodes: {len(returns):,} | Current episode: {episode:,}",
                    f"Decision {step:,} / {total_steps:,} | Elapsed {elapsed / 60:.1f} min",
                    "Training scores | Orange line: trailing mean of up to 20 episodes",
                ]),
                with_status_bar(learning, [
                    f"Optimizer updates: {updates:,} | Epsilon: {epsilon:.3f}",
                    f"Replay: {replay_size:,} / {replay_capacity:,} | Logged points: {len(losses):,}",
                    "Loss and Q are learning diagnostics; episode score measures gameplay",
                ]),
            )
            for name, image in zip(self.names[1:], self.graph_images):
                cv2.imshow(name, image)
            self.last_graph = now
        self.shown = True
        return not self.poll_stop()

    def wait_until_visible(self, timeout=5.0):
        """Pump startup events before treating visibility loss as a user close.

        namedWindow/imshow may return before the native window is mapped. A zero
        visibility value during that interval is not a request to stop training.
        """
        if not self.shown or len(self.opened) != len(self.names):
            raise RuntimeError("Open and render all three monitor windows before waiting for startup.")
        deadline = time.monotonic() + timeout
        while True:
            if self.poll_stop():
                return False
            try:
                visible = [cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
                           for name in self.opened]
                if all(visible):
                    self.ready = True
                    return True
            except cv2.error as exc:
                raise RuntimeError(
                    "OpenCV could not verify monitor startup. Check the Python/kernel path "
                    "printed above and use a local desktop kernel with GUI-enabled OpenCV."
                ) from exc
            if time.monotonic() >= deadline:
                missing = [name for name, shown in zip(self.opened, visible) if not shown]
                raise RuntimeError(
                    f"Monitor windows did not become visible within {timeout:g}s: {missing}. "
                    "Training has not started. Check that the selected kernel runs on your "
                    "local desktop, rather than WSL, SSH, a container or a remote server."
                )

    def poll_stop(self):
        if self.stopped:
            return True
        try:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self.stopped = True
                return True
            # Startup visibility is handled by wait_until_visible, not by the
            # stop detector. Only already-visible windows can be closed by users.
            closed = self.ready and any(
                cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                for name in self.opened
            )
            self.stopped = closed
        except cv2.error as exc:
            # Some HighGUI builds report a destroyed native window as NULL.
            if self.ready and exc.code == -27 and "NULL window" in str(exc):
                self.stopped = True
            else:
                raise RuntimeError(
                    "OpenCV monitor event handling failed. This is a GUI error, not a "
                    "user stop request; see the original OpenCV exception below."
                ) from exc
        return self.stopped

    def close(self):
        for name in self.opened:
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass
        self.opened.clear()
        self.stopped = True
        self.ready = False
        # Flush close events without destroying unrelated notebook windows.
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass
