"""Train a policy player with an actor and a critic.

Eight games run side by side. After every 32 decisions in each, the
critic's value guesses are used to judge those 256 decisions: each one
is scored by how much better things went over the next few steps than
the critic expected. The actor's policy is moved once towards the better
actions, the critic is moved towards the values that really came, and
play carries on from where it stopped. Nothing waits for a life to end.

Run it with:  uv run train_actor_critic.py
Options:
    --frames N    train for N frames instead of 6 million
"""

import sys

import numpy as np
import torch

from brickgame.env import BrickEnv
from players import play_game
from players.policy import PolicyPlayer
from training.decisions import play_decision, sliding
from training.progress import ProgressLog

TOTAL_FRAMES = 6_000_000
TRAINING_FRAMES = 10_000
GAMES_AT_ONCE = 8
DECISIONS_PER_UPDATE = 32
DISCOUNT = 0.99
SMOOTHING = 0.95
LEARNING_RATE = 0.0007
VALUE_WEIGHT = 0.5
ENTROPY_BONUS = 0.01
TEST_EVERY = 250_000
TEST_GAMES = 10


class SideBySide:
    """Several copies of the game, played together one decision at a
    time. A copy that finishes starts a new game straight away."""

    def __init__(self, games, log, **settings):
        """settings are passed on to every BrickEnv, such as a different
        paddle_hit_reward."""
        self.envs = [
            BrickEnv(max_frames=TRAINING_FRAMES, **settings)
            for _ in range(games)
        ]
        self.observations = []
        for number, env in enumerate(self.envs):
            self.observations.append(env.reset(seed=number + 1)[0])
        self.next_seed = games + 1
        self.log = log
        self.frames = 0

    def play(self, player, decisions, extra):
        """Play `decisions` decisions in every copy. Returns arrays with
        one row per decision and one column per copy, plus the critic's
        values for the positions the copies stopped in."""
        shape = (decisions, len(self.envs))
        states = np.zeros(
            (*shape, len(self.observations[0])), dtype=np.float32
        )
        actions = np.zeros(shape, dtype=np.int64)
        log_chances = np.zeros(shape, dtype=np.float32)
        values = np.zeros(shape, dtype=np.float32)
        rewards = np.zeros(shape, dtype=np.float32)
        cuts = np.zeros(shape, dtype=np.float32)

        for step in range(decisions):
            states[step] = np.array(self.observations)
            with torch.no_grad():
                chances, value = player.network(
                    torch.as_tensor(states[step])
                )
                chosen = chances.sample()
            actions[step] = chosen.numpy()
            log_chances[step] = chances.log_prob(chosen).numpy()
            values[step] = value.numpy()

            for number, env in enumerate(self.envs):
                action = int(chosen[number])
                observation, reward, played, finished, ended, info = (
                    play_decision(env, action, player.decide_every)
                )
                self.frames += played
                rewards[step, number] = reward
                cuts[step, number] = ended or finished
                if finished:
                    self.log.game_over(info, extra=extra)
                    observation = env.reset(seed=self.next_seed)[0]
                    self.next_seed += 1
                self.observations[number] = observation

        with torch.no_grad():
            last = torch.as_tensor(np.array(self.observations))
            last_values = player.network(last)[1].numpy()
        return (
            states,
            actions,
            log_chances,
            values,
            rewards,
            cuts,
            last_values,
        )


def advantages_of(rewards, values, cuts, last_values):
    """Generalized advantage estimation, for each game copy separately.

    The one-step surprise is delta = reward + gamma * value(next) -
    value(now). The advantage adds up the surprises that follow, each
    shrunk by gamma * lambda per step, stopping where a life ended.
    """
    steps, games = rewards.shape
    advantages = np.zeros((steps, games), dtype=np.float32)
    following = np.zeros(games, dtype=np.float32)
    next_values = last_values
    for step in reversed(range(steps)):
        carry_on = 1.0 - cuts[step]
        delta = (
            rewards[step]
            + DISCOUNT * next_values * carry_on
            - values[step]
        )
        following = delta + DISCOUNT * SMOOTHING * carry_on * following
        advantages[step] = following
        next_values = values[step]
    return advantages


def flatten(*arrays):
    """Turn (decisions, copies, ...) arrays into one long batch each."""
    batch = []
    for array in arrays:
        batch.append(torch.as_tensor(array.reshape(-1, *array.shape[2:])))
    return batch


def improve(player, optimizer, states, actions, advantages, returns):
    chances, values = player.network(states)
    log_chances = chances.log_prob(actions)
    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )
    policy_loss = -(log_chances * advantages).mean()
    value_loss = ((values - returns) ** 2).mean()
    entropy = chances.entropy().mean()
    loss = (
        policy_loss + VALUE_WEIGHT * value_loss - ENTROPY_BONUS * entropy
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(player.network.parameters(), 0.5)
    optimizer.step()
    return entropy.item()


def test(player, games, **settings):
    player.greedy = True
    env = BrickEnv(**settings)
    scores = []
    for game in range(games):
        scores.append(
            play_game(player, env, seed=100_000 + game)["score"]
        )
    player.greedy = False
    return scores


def keep_if_best(player, log, best_average, **settings):
    """Play the test games, log them, and save the network if it beat
    every earlier test. Returns the best average so far. settings are
    passed on to the test games' BrickEnv, so that a player trained
    without seeing the bricks is tested without them too."""
    scores = test(player, TEST_GAMES, **settings)
    log.test_result(scores)
    average = sum(scores) / len(scores)
    if average > best_average:
        player.save(log.folder / "best.pt")
        print(f"       new best network: {average:.1f} bricks")
        return average
    return best_average


def main():
    torch.set_num_threads(1)
    total_frames = TOTAL_FRAMES
    if "--frames" in sys.argv:
        total_frames = int(sys.argv[sys.argv.index("--frames") + 1])

    player = PolicyPlayer()
    optimizer = torch.optim.Adam(
        player.network.parameters(), LEARNING_RATE
    )
    log = ProgressLog("actor-critic", extra="entropy", report_every=40)
    games = SideBySide(GAMES_AT_ONCE, log)

    entropy = 0.0
    next_test = TEST_EVERY
    best_average = -1
    while games.frames < total_frames:
        for group in optimizer.param_groups:
            group["lr"] = sliding(
                LEARNING_RATE, 0.0, games.frames / total_frames
            )
        states, actions, _, values, rewards, cuts, last_values = (
            games.play(player, DECISIONS_PER_UPDATE, round(entropy, 4))
        )
        advantages = advantages_of(rewards, values, cuts, last_values)
        returns = advantages + values
        entropy = improve(
            player,
            optimizer,
            *flatten(states, actions, advantages, returns),
        )

        if games.frames >= next_test:
            next_test += TEST_EVERY
            best_average = keep_if_best(player, log, best_average)
    log.close()


if __name__ == "__main__":
    main()
