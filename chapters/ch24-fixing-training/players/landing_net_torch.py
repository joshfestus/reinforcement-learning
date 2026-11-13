"""The landing network again, built with PyTorch.

It has the same layers and the same methods as LandingNetwork, so the
two can be swapped. The difference is in train_step: PyTorch works out
every gradient by itself, so the backward pass is one line.
"""

import torch
from torch import nn

from players.landing_net import INPUTS


class TorchLandingNetwork:
    def __init__(self, hidden=32, seed=0, optimizer="plain"):
        """optimizer "plain" steps each weight against its gradient, as
        the NumPy version does. "adam" lets each weight keep its own
        running step size, which usually learns faster."""
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(INPUTS, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.adam = None
        if optimizer == "adam":
            self.adam = torch.optim.Adam(
                self.model.parameters(), lr=0.001
            )

    def predict(self, inputs):
        with torch.no_grad():
            guesses = self.model(
                torch.as_tensor(inputs, dtype=torch.float32)
            )
        return guesses[:, 0].numpy()

    def train_step(self, inputs, answers, learning_rate):
        inputs = torch.as_tensor(inputs, dtype=torch.float32)
        answers = torch.as_tensor(answers, dtype=torch.float32)

        guesses = self.model(inputs)[:, 0]
        loss = ((guesses - answers) ** 2).mean()

        self.model.zero_grad()
        loss.backward()
        if self.adam:
            self.adam.step()
        else:
            with torch.no_grad():
                for weight in self.model.parameters():
                    weight -= learning_rate * weight.grad
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
