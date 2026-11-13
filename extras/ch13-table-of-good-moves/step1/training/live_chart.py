"""A window that redraws the learning curve while a player trains."""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from training.charts import (
    FAINT,
    INK,
    SOFT_INK,
    average_bricks,
    label_at_end,
    moving_average,
    new_chart,
    plain_number,
)


class LiveChart:
    def __init__(self, title, baselines=(), redraw_every=10, window=50):
        plt.ion()
        self.figure, self.axes = new_chart(
            f"Training {title}",
            "frames played while training",
            "bricks broken per game",
            how_to_read=f"Gray dots are single games. The black line is "
            f"the average of the last {window} games.",
        )
        self.redraw_every = redraw_every
        self.window = window
        self.frames = []
        self.bricks = []

        for baseline in baselines:
            average = average_bricks(baseline)
            self.axes.axhline(average, color=SOFT_INK, linewidth=1)
            name = f"{baseline}: {average:,.0f}"
            label_at_end(self.axes, average, name)
        self.dots = self.axes.scatter([], [], s=4, color=FAINT)
        (self.line,) = self.axes.plot([], [], color=INK, linewidth=2)
        self.axes.xaxis.set_major_formatter(FuncFormatter(plain_number))
        plt.show(block=False)

    def add(self, frames_played, bricks):
        self.frames.append(frames_played)
        self.bricks.append(bricks)
        if len(self.bricks) % self.redraw_every == 0:
            self.redraw()

    def redraw(self):
        points = list(zip(self.frames, self.bricks, strict=True))
        self.dots.set_offsets(points)
        average = moving_average(self.bricks, self.window)
        self.line.set_data(self.frames, average)
        self.axes.relim()
        self.axes.autoscale_view()
        plt.pause(0.001)

    def close(self):
        self.redraw()
        plt.ioff()
        plt.close(self.figure)
