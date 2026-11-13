"""A Deep Q-Network player: the Q-table, with the table replaced by a
neural network.

The network takes the observation and puts out three numbers, one Q-value
per action. Because similar situations give similar inputs, what it learns
in one situation carries over to others it has never met, which a table
cannot do.
"""

import random

import torch
from torch import nn

from brickgame.env import BRICKS_START
from brickgame.game import ACTIONS
from brickgame.levels import COLUMNS, MAX_ROWS


def make_network(inputs, hidden=128):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, len(ACTIONS)),
    )


class DQNPlayer:
    name = "dqn"

    def __init__(
        self,
        see_bricks=True,
        discount=0.99,
        learning_rate=0.0001,
        epsilon=1.0,
        decide_every=4,
        seed=0,
    ):
        """see_bricks=False gives the network only the seven ball and
        paddle numbers, not the forty-eight bricks. decide_every is how
        many frames each chosen action is kept for."""
        torch.manual_seed(seed)
        self.random = random.Random(seed)
        self.see_bricks = see_bricks
        self.inputs = BRICKS_START
        if see_bricks:
            self.inputs += COLUMNS * MAX_ROWS
        self.discount = discount
        self.epsilon = epsilon
        self.decide_every = decide_every

        self.network = make_network(self.inputs)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate
        )
        self.action = 0
        self.frames_left = 0

    def state(self, observation):
        """The part of the observation this player sees."""
        return observation[: self.inputs]

    def choose(self, state):
        if self.random.random() < self.epsilon:
            return self.random.choice(ACTIONS)
        with torch.no_grad():
            q_values = self.network(torch.from_numpy(state))
        return int(q_values.argmax())

    def act(self, observation):
        """Play one frame, deciding afresh every decide_every frames."""
        if self.frames_left == 0:
            self.action = self.choose(self.state(observation))
            self.frames_left = self.decide_every
        self.frames_left -= 1
        return self.action

    def learn(self, state, action, reward, new_state, ended):
        """One step of learning from the decision just made. Returns the
        loss: how far the network's Q-value was from its target."""
        state = torch.from_numpy(state)[None]
        new_state = torch.from_numpy(new_state)[None]

        # What the network now says the chosen action was worth.
        chosen = self.network(state)[0, action]

        # What it should have said: the reward, plus the discounted value
        # of the best action from where the player ended up.
        with torch.no_grad():
            best_next = self.network(new_state).max()
            target = reward + self.discount * best_next * (1 - ended)

        # Huber loss: squared error for small misses, but it grows only
        # in a straight line for large ones, so one wild target cannot
        # throw the weights far off.
        loss = nn.functional.smooth_l1_loss(chosen, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(
            {
                "see_bricks": self.see_bricks,
                "decide_every": self.decide_every,
                "network": self.network.state_dict(),
            },
            path,
        )


def load_dqn(path):
    """A trained player from a saved file, ready to play without
    exploring."""
    saved = torch.load(path)
    player = DQNPlayer(
        see_bricks=saved["see_bricks"],
        decide_every=saved["decide_every"],
        epsilon=0,
    )
    player.network.load_state_dict(saved["network"])
    return player
