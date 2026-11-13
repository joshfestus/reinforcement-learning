"""A player that learns a table of how good each move is, situation by
situation.

The game has far too many positions to list, so the player sorts each
moment into one of a few hundred situations: roughly where the ball is
compared with the paddle, how high it is, and which way it is moving.
For each situation the table holds three numbers, one per action: the
player's current guess of the reward that will follow if it takes that
action and plays well afterwards. These guesses are called Q-values.
"""

import json
import random

from brickgame.env import BALL_DX, BALL_DY, BALL_FROM_PADDLE, BALL_Y
from brickgame.game import ACTIONS, WIDTH

# How far the ball is to the left (negative) or right of the paddle's
# center, in pixels. The buckets are narrow close to the paddle, where
# a few pixels decide a catch, and wide further out.
OFFSET_EDGES = [-150, -80, -40, -20, -10, -4, 0, 4, 10, 20, 40, 80, 150]

# How far down the screen the ball is: -1 is the top, 1 the bottom.
HEIGHT_EDGES = [-0.2, 0.3, 0.7]

# The player picks an action, then keeps it for this many frames. One
# frame moves the paddle 5 pixels, too little for the table to tell the
# three actions apart.
DECIDE_EVERY = 4


def bucket(value, edges):
    """0 below the first edge, 1 past the first, 2 past the second..."""
    number = 0
    for edge in edges:
        if value >= edge:
            number += 1
    return number


def situation(observation):
    offset = observation[BALL_FROM_PADDLE] * WIDTH
    falling = 1 if observation[BALL_DY] > 0 else 0
    moving_right = 1 if observation[BALL_DX] > 0 else 0
    return (
        bucket(offset, OFFSET_EDGES),
        bucket(observation[BALL_Y], HEIGHT_EDGES),
        falling,
        moving_right,
    )


class QTablePlayer:
    name = "qtable"

    def __init__(
        self,
        learning_rate=0.1,
        discount=0.99,
        epsilon=1.0,
        method="q-learning",
        seed=None,
    ):
        """discount is gamma: how much a reward one decision later is
        worth compared with the same reward now. method is "q-learning"
        or "sarsa"."""
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.method = method
        self.random = random.Random(seed)
        self.table = {}
        self.action = 0
        self.frames_left = 0

    def q_values(self, state):
        if state not in self.table:
            self.table[state] = [0.0, 0.0, 0.0]
        return self.table[state]

    def choose(self, state):
        """Usually the action with the highest Q-value; with probability
        epsilon a random one, to keep trying things."""
        if self.random.random() < self.epsilon:
            return self.random.choice(ACTIONS)
        values = self.q_values(state)
        best = max(values)
        best_actions = [a for a in ACTIONS if values[a] == best]
        return self.random.choice(best_actions)

    def learn(self, state, action, reward, new_state, new_action, ended):
        """Move one Q-value toward what the decision turned out to be
        worth: the reward it brought, plus the discounted value of the
        situation it led to."""
        if ended:
            future = 0
        elif self.method == "sarsa":
            future = self.q_values(new_state)[new_action]
        else:
            future = max(self.q_values(new_state))
        target = reward + self.discount * future

        values = self.q_values(state)
        values[action] += self.learning_rate * (target - values[action])

    def act(self, observation):
        """Play one frame, deciding afresh every DECIDE_EVERY frames."""
        if self.frames_left == 0:
            self.action = self.choose(situation(observation))
            self.frames_left = DECIDE_EVERY
        self.frames_left -= 1
        return self.action

    def save(self, path):
        rows = {}
        for state, values in self.table.items():
            key = ",".join(str(part) for part in state)
            rows[key] = [round(value, 4) for value in values]
        path.write_text(json.dumps(rows, indent=1))

    def load(self, path):
        self.table = {}
        for key, values in json.loads(path.read_text()).items():
            state = tuple(int(part) for part in key.split(","))
            self.table[state] = values
