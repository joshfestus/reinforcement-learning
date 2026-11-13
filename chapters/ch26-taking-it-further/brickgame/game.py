"""The rules of the brick breaker game.

Nothing here draws to the screen or reads the keyboard. The game moves one
frame each time step() is called, so a person, a window or a training
program can all drive it in the same way:

    game = BrickGame()
    while not game.game_over:
        game.step(RIGHT)

Positions are in pixels. (0, 0) is the top left corner, x grows to the
right and y grows downward.
"""

import math
import random
from dataclasses import dataclass

from brickgame.levels import COLUMNS, level_layout

WIDTH = 600
HEIGHT = 500

PADDLE_WIDTH = 80
PADDLE_HEIGHT = 12
PADDLE_Y = HEIGHT - 40

# The paddle is slower than the ball. A paddle that only chases the ball
# falls behind on steep angles, so a player has to judge where the ball
# will come down and move there early.
PADDLE_SPEED = 5

BALL_RADIUS = 6
BALL_SPEED = 6
BOUNCE_WOBBLE = 3

BRICK_GAP = 4
BRICK_HEIGHT = 20
BRICK_TOP = 60
BRICK_WIDTH = (WIDTH - BRICK_GAP * (COLUMNS + 1)) / COLUMNS

START_LIVES = 3

# Each level makes the ball faster and the paddle narrower. The ball is
# capped at 16 pixels a frame: any faster and it could pass a brick
# without ever overlapping it.
SPEEDUP_PER_LEVEL = 0.25
MAX_BALL_SPEED = 16
SHRINK_PER_LEVEL = 15
MIN_PADDLE_WIDTH = 30

STAY = 0
LEFT = 1
RIGHT = 2
ACTIONS = [STAY, LEFT, RIGHT]


@dataclass
class Brick:
    x: float
    y: float
    row: int
    column: int
    letter: str


class BrickGame:
    def __init__(self, seed=None):
        self.random = random.Random(seed)
        self.reset()

    def reset(self):
        self.score = 0
        self.lives = START_LIVES
        self.level = 1
        self.frames = 0
        self.paddle_hits = 0
        self.game_over = False
        self.start_level()

    def start_level(self):
        faster = 1 + SPEEDUP_PER_LEVEL * (self.level - 1)
        self.ball_speed = min(MAX_BALL_SPEED, BALL_SPEED * faster)

        narrower = SHRINK_PER_LEVEL * (self.level - 1)
        self.paddle_width = max(MIN_PADDLE_WIDTH, PADDLE_WIDTH - narrower)
        self.paddle_x = (WIDTH - self.paddle_width) / 2

        self.bricks = []
        for row, column, letter in level_layout(self.level):
            x = BRICK_GAP + column * (BRICK_WIDTH + BRICK_GAP)
            y = BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
            self.bricks.append(Brick(x, y, row, column, letter))

        self.serve()

    def serve(self):
        """Set the ball on the paddle and send it up at a random angle."""
        self.ball_x = self.paddle_x + self.paddle_width / 2
        self.ball_y = PADDLE_Y - BALL_RADIUS - 1
        angle = math.radians(self.random.uniform(-45, 45))
        self.ball_dx = self.ball_speed * math.sin(angle)
        self.ball_dy = -self.ball_speed * math.cos(angle)

    def step(self, action):
        if self.game_over:
            return
        self.frames += 1

        self.move_paddle(action)
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        self.bounce_off_walls()

        if self.ball_meets_paddle():
            self.bounce_off_paddle()
            self.paddle_hits += 1

        if self.break_brick():
            self.score += 1
            if not self.bricks:
                self.level += 1
                self.start_level()

        if self.ball_y - BALL_RADIUS > HEIGHT:
            self.lives -= 1
            if self.lives == 0:
                self.game_over = True
            else:
                self.serve()

    def move_paddle(self, action):
        if action == LEFT:
            self.paddle_x -= PADDLE_SPEED
        elif action == RIGHT:
            self.paddle_x += PADDLE_SPEED
        right_limit = WIDTH - self.paddle_width
        self.paddle_x = max(0, min(right_limit, self.paddle_x))

    def bounce_off_walls(self):
        if self.ball_x < BALL_RADIUS:
            self.ball_x = BALL_RADIUS
            self.ball_dx = abs(self.ball_dx)
        if self.ball_x > WIDTH - BALL_RADIUS:
            self.ball_x = WIDTH - BALL_RADIUS
            self.ball_dx = -abs(self.ball_dx)
        if self.ball_y < BALL_RADIUS:
            self.ball_y = BALL_RADIUS
            self.ball_dy = abs(self.ball_dy)

    def ball_meets_paddle(self):
        if self.ball_dy < 0:
            return False

        # Compare where the ball's bottom edge is now with where it was a
        # frame ago. A fast ball can move further in one frame than the
        # paddle is thick, so checking only the new position would let it
        # pass straight through.
        bottom_now = self.ball_y + BALL_RADIUS
        bottom_before = bottom_now - self.ball_dy
        crossed_top = (
            bottom_now >= PADDLE_Y
            and bottom_before <= PADDLE_Y + PADDLE_HEIGHT
        )

        left_edge = self.paddle_x - BALL_RADIUS
        right_edge = self.paddle_x + self.paddle_width + BALL_RADIUS
        return crossed_top and left_edge <= self.ball_x <= right_edge

    def bounce_off_paddle(self):
        # Where the ball lands sets its new direction: the center sends it
        # straight up, the ends send it off at up to 60 degrees. This is
        # what lets a player aim.
        center = self.paddle_x + self.paddle_width / 2
        half_width = self.paddle_width / 2 + BALL_RADIUS
        offset = (self.ball_x - center) / half_width
        offset = max(-1.0, min(1.0, offset))

        # A few degrees of chance in every bounce, so that no two rallies
        # are exactly alike and a player has to watch the ball rather
        # than repeat moves it has memorized.
        wobble = self.random.uniform(-BOUNCE_WOBBLE, BOUNCE_WOBBLE)
        angle = math.radians(offset * 60 + wobble)
        self.ball_dx = self.ball_speed * math.sin(angle)
        self.ball_dy = -self.ball_speed * math.cos(angle)
        self.ball_y = PADDLE_Y - BALL_RADIUS

    def break_brick(self):
        """Remove the first brick the ball overlaps and bounce off it."""
        for brick in self.bricks:
            center_x = brick.x + BRICK_WIDTH / 2
            center_y = brick.y + BRICK_HEIGHT / 2
            dx = self.ball_x - center_x
            dy = self.ball_y - center_y
            overlap_x = BRICK_WIDTH / 2 + BALL_RADIUS - abs(dx)
            overlap_y = BRICK_HEIGHT / 2 + BALL_RADIUS - abs(dy)
            if overlap_x <= 0 or overlap_y <= 0:
                continue

            # The ball has pushed less far into the side it came through.
            if overlap_x < overlap_y:
                self.ball_dx = abs(self.ball_dx)
                if dx < 0:
                    self.ball_dx = -self.ball_dx
            else:
                self.ball_dy = abs(self.ball_dy)
                if dy < 0:
                    self.ball_dy = -self.ball_dy
            self.bricks.remove(brick)
            return True
        return False
