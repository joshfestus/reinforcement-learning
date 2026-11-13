"""A small neural network written with NumPy alone.

Four numbers go in: the ball's position and speed. One number comes out:
where the network thinks the ball will land. Between them is one hidden
layer of units, each of which adds up the inputs with its own weights and
passes the total through tanh, a curve that squashes any number into the
range -1 to 1.
"""

import numpy as np

INPUTS = 4


class LandingNetwork:
    def __init__(self, hidden=32, seed=0):
        # Small random starting weights. Scaling by the square root of
        # the number of inputs keeps the totals in tanh's useful range.
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 1 / np.sqrt(INPUTS), (INPUTS, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, 1))
        self.b2 = np.zeros(1)

    def predict(self, inputs):
        """One guess per row of inputs."""
        hidden = np.tanh(inputs @ self.w1 + self.b1)
        return (hidden @ self.w2 + self.b2)[:, 0]

    def train_step(self, inputs, answers, learning_rate):
        """Make one small correction from a batch of examples.

        Returns the batch's mean squared error: the average of
        (guess - answer) squared, before the correction.
        """
        # Forward: the same steps as predict(), keeping the hidden values
        # because the backward pass needs them.
        hidden = np.tanh(inputs @ self.w1 + self.b1)
        guesses = (hidden @ self.w2 + self.b2)[:, 0]
        errors = guesses - answers
        loss = np.mean(errors**2)

        # Backward: for every weight, how much the loss would change if
        # that weight grew a little. This is the chain rule, applied one
        # layer at a time, from the output back to the inputs.
        d_guesses = 2 * errors / len(answers)
        d_w2 = hidden.T @ d_guesses[:, None]
        d_b2 = d_guesses.sum(keepdims=True)
        d_hidden = d_guesses[:, None] @ self.w2.T
        d_totals = d_hidden * (1 - hidden**2)
        d_w1 = inputs.T @ d_totals
        d_b1 = d_totals.sum(axis=0)

        # Nudge each weight in the direction that lowers the loss.
        self.w1 -= learning_rate * d_w1
        self.b1 -= learning_rate * d_b1
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2
        return loss

    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load(self, path):
        weights = np.load(path)
        self.w1 = weights["w1"]
        self.b1 = weights["b1"]
        self.w2 = weights["w2"]
        self.b2 = weights["b2"]
