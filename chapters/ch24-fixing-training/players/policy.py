"""A player that learns which move to make, rather than how good each
move is.

Its network has two outputs. The policy gives a chance for each action;
the player picks an action at random with those chances, so it explores
by itself, with no epsilon. The value is the network's guess of how much
reward is still to come from this position. REINFORCE (chapter 20) uses
only the policy. Actor and critic (chapter 21) and PPO (chapter 22) use
both: the policy acts, the value judges.
"""

import torch
from torch import nn

from brickgame.env import BRICKS_START
from brickgame.game import ACTIONS
from brickgame.levels import COLUMNS, MAX_ROWS

INPUTS = BRICKS_START + COLUMNS * MAX_ROWS


class PolicyNetwork(nn.Module):
    def __init__(self, inputs=INPUTS, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))
        self.value = nn.Linear(hidden, 1)

    def forward(self, states):
        """Chances for each action (as a distribution) and the value."""
        features = self.body(states)
        chances = torch.distributions.Categorical(
            logits=self.policy(features)
        )
        return chances, self.value(features)[:, 0]


class PolicyPlayer:
    name = "policy"

    def __init__(self, decide_every=4, seed=0):
        torch.manual_seed(seed)
        self.network = PolicyNetwork()
        self.decide_every = decide_every
        self.greedy = False
        self.action = 0
        self.frames_left = 0

    def choose(self, state):
        """Pick an action with the policy's chances, or, when greedy,
        the action with the highest chance."""
        with torch.no_grad():
            chances, value = self.network(torch.as_tensor(state)[None])
        if self.greedy:
            return int(chances.probs.argmax())
        return int(chances.sample())

    def act(self, observation):
        if self.frames_left == 0:
            self.action = self.choose(observation)
            self.frames_left = self.decide_every
        self.frames_left -= 1
        return self.action

    def save(self, path):
        torch.save(
            {
                "decide_every": self.decide_every,
                "network": self.network.state_dict(),
            },
            path,
        )


def load_policy(path):
    saved = torch.load(path)
    player = PolicyPlayer(decide_every=saved["decide_every"])
    player.network.load_state_dict(saved["network"])
    player.greedy = True
    return player
