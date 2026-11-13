"""The environment as a learning library sees it: one step per decision.

A library such as Stable-Baselines3 calls step() once per action. This
wrapper makes each of those steps hold the action for several frames,
as the book's own players do, and, while training, reports a lost life
as the end of an episode. When the library then asks for a reset, the
game carries on with its next life; only a finished game really starts
over.
"""

import gymnasium as gym

from training.decisions import play_decision


class DecisionEnv(gym.Wrapper):
    def __init__(self, env, decide_every=4, lost_life_ends=True):
        super().__init__(env)
        self.decide_every = decide_every
        self.lost_life_ends = lost_life_ends
        self.game_finished = True

    def reset(self, seed=None, options=None):
        # A reset that names a seed always starts a fresh game.
        if self.game_finished or seed is not None:
            self.game_finished = False
            return self.env.reset(seed=seed, options=options)
        return (
            self.env.unwrapped.observe(),
            self.env.unwrapped.information(),
        )

    def step(self, action):
        observation, reward, played, finished, ended, info = (
            play_decision(self.env, int(action), self.decide_every)
        )
        info["played"] = played
        self.game_finished = finished
        terminated = info.get("ending") == "lost"
        truncated = finished and not terminated
        if self.lost_life_ends and ended and not finished:
            terminated = True
        return observation, reward, terminated, truncated, info
