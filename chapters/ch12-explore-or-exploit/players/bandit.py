"""A learner that picks which player to send out for each game.

It has a few players to choose from (its "arms") and does not know which
one scores best. After every game it updates its estimate of the chosen
arm. Most of the time it picks the arm with the best estimate; with
probability epsilon it picks one at random, to keep checking the others.
"""

import random


class Bandit:
    def __init__(self, arm_names, epsilon=0.1, decay=1.0, seed=None):
        """decay multiplies epsilon after every game: 1.0 keeps it fixed,
        0.99 shrinks it a little each time."""
        self.arm_names = arm_names
        self.epsilon = epsilon
        self.decay = decay
        self.random = random.Random(seed)
        self.counts = [0] * len(arm_names)
        self.estimates = [0.0] * len(arm_names)

    def choose(self):
        """The number of the arm to play next."""
        arms = range(len(self.arm_names))
        if self.random.random() < self.epsilon:
            return self.random.choice(arms)

        best = max(self.estimates)
        best_arms = [arm for arm in arms if self.estimates[arm] == best]
        return self.random.choice(best_arms)

    def learn(self, arm, score):
        # The new estimate is the average of every score this arm has
        # had, updated without storing the scores: move the old average
        # toward the new score by 1/count of the gap.
        self.counts[arm] += 1
        gap = score - self.estimates[arm]
        self.estimates[arm] += gap / self.counts[arm]
        self.epsilon *= self.decay
