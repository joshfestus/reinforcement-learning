"""Train PPO from the Stable-Baselines3 library.

The settings match train_ppo.py, so the two can be compared on one
chart. The library's own log table is switched off; add --library-log
to see it.

Run it with:  uv run train_library_ppo.py
Options:
    --frames N       train for N frames instead of 6 million
    --library-log    print the library's own table after every update
"""

import sys

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from brickgame.decision_env import DecisionEnv
from brickgame.env import BrickEnv
from players import play_game
from training.decisions import sliding
from training.progress import ProgressLog

TOTAL_FRAMES = 6_000_000
TRAINING_FRAMES = 10_000
GAMES_AT_ONCE = 8
DECIDE_EVERY = 4
TEST_EVERY = 250_000
TEST_GAMES = 10


class LibraryPlayer:
    """Lets a trained library model play through play_game()."""

    name = "library ppo"

    def __init__(self, model):
        self.model = model
        self.action = 0
        self.frames_left = 0

    def act(self, observation):
        if self.frames_left == 0:
            action, _ = self.model.predict(
                observation, deterministic=True
            )
            self.action = int(action)
            self.frames_left = DECIDE_EVERY
        self.frames_left -= 1
        return self.action


class BookLog(BaseCallback):
    """Writes the library's games into the book's own log, and plays
    test games every TEST_EVERY frames."""

    def __init__(self, total_frames=TOTAL_FRAMES):
        super().__init__()
        self.total_frames = total_frames
        self.log = ProgressLog("ppo-library", report_every=40)
        self.frames = 0
        self.next_test = TEST_EVERY
        self.best_average = -1

    def _on_step(self):
        for info in self.locals["infos"]:
            self.frames += info["played"]
            if "ending" in info:
                self.log.game_over(info)

        if self.frames >= self.next_test:
            self.next_test += TEST_EVERY
            env = BrickEnv()
            player = LibraryPlayer(self.model)
            scores = []
            for game in range(TEST_GAMES):
                info = play_game(player, env, seed=100_000 + game)
                scores.append(info["score"])
            self.log.test_result(scores)
            average = sum(scores) / len(scores)
            if average > self.best_average:
                self.best_average = average
                self.model.save(self.log.folder / "best")
                print(f"       new best network: {average:.1f} bricks")
        return self.frames < self.total_frames

    def _on_training_end(self):
        self.log.close()


def make_env():
    return DecisionEnv(
        BrickEnv(max_frames=TRAINING_FRAMES), decide_every=DECIDE_EVERY
    )


def main():
    torch.set_num_threads(1)
    total_frames = TOTAL_FRAMES
    if "--frames" in sys.argv:
        total_frames = int(sys.argv[sys.argv.index("--frames") + 1])
    envs = make_vec_env(make_env, n_envs=GAMES_AT_ONCE, seed=1)
    model = PPO(
        "MlpPolicy",
        envs,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=lambda left: sliding(0.0, 0.0003, left),
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1 if "--library-log" in sys.argv else 0,
        seed=0,
    )
    # The library counts decisions, not frames, and slides the learning
    # rate down over the decisions it is asked for. Each decision is
    # DECIDE_EVERY frames.
    model.learn(
        total_timesteps=total_frames // DECIDE_EVERY,
        callback=BookLog(total_frames),
    )


if __name__ == "__main__":
    main()
