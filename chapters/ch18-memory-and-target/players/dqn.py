"""A Deep Q-Network player: the Q-table, with the table replaced by a
neural network.

The network takes the observation and puts out three numbers, one Q-value
per action. Because similar situations give similar inputs, what it learns
in one situation carries over to others it has never met, which a table
cannot do.

Two extra pieces make it train steadily:

    the replay memory keeps the last many decisions, and the network
    learns from a random handful of them at a time, instead of only from
    the latest one;

    the target network is a copy of the network, refreshed only now and
    then, used to value "where the player ended up", so that the target
    the network chases does not move every time it learns.
"""

import random

import numpy as np
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


class ReplayMemory:
    """The most recent decisions, stored in a ring: once full, each new
    decision overwrites the oldest."""

    def __init__(self, size, inputs):
        self.size = size
        self.states = np.zeros((size, inputs), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.new_states = np.zeros((size, inputs), dtype=np.float32)
        self.ended = np.zeros(size, dtype=np.float32)
        self.count = 0

    def add(self, state, action, reward, new_state, ended):
        slot = self.count % self.size
        self.states[slot] = state
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.new_states[slot] = new_state
        self.ended[slot] = ended
        self.count += 1

    def sample(self, how_many, rng):
        filled = min(self.count, self.size)
        picks = rng.integers(0, filled, size=how_many)
        return (
            torch.from_numpy(self.states[picks]),
            torch.from_numpy(self.actions[picks]),
            torch.from_numpy(self.rewards[picks]),
            torch.from_numpy(self.new_states[picks]),
            torch.from_numpy(self.ended[picks]),
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
        self.target = make_network(self.inputs)
        self.refresh_target()
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

    def refresh_target(self):
        self.target.load_state_dict(self.network.state_dict())

    def learn(self, batch, use_target=True):
        """One step of learning from a batch of past decisions. Returns
        the loss: how far the network's Q-values were from their
        targets."""
        states, actions, rewards, new_states, ended = batch

        # What the network now says each chosen action was worth.
        q_values = self.network(states)
        chosen = q_values.gather(1, actions[:, None])[:, 0]

        # What it should have said: the reward, plus the discounted value
        # of the best action from where the player ended up.
        valuer = self.target if use_target else self.network
        with torch.no_grad():
            best_next = valuer(new_states).max(dim=1).values
            targets = rewards + self.discount * best_next * (1 - ended)

        # Huber loss: squared error for small misses, but it grows only
        # in a straight line for large ones, so one wild target cannot
        # throw the weights far off.
        loss = nn.functional.smooth_l1_loss(chosen, targets)
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
        decide_every=saved.get("decide_every", 4),
        epsilon=0,
    )
    player.network.load_state_dict(saved["network"])
    return player
