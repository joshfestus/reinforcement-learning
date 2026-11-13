"""Train a policy player with REINFORCE.

REINFORCE plays a few games, then makes every action it took a little
more likely if the reward that followed it was better than usual, and
a little less likely if it was worse.

Run it with:  uv run train_reinforce.py
"""

import numpy as np
import torch

from brickgame.env import BrickEnv
from players import play_game
from players.policy import PolicyPlayer
from training.decisions import play_decision
from training.progress import ProgressLog

TOTAL_FRAMES = 6_000_000
TRAINING_FRAMES = 5_000
GAMES_PER_UPDATE = 8
DISCOUNT = 0.99
LEARNING_RATE = 0.0005
ENTROPY_BONUS = 0.01
TEST_EVERY = 250_000
TEST_GAMES = 10


def play_and_record(player, env, seed):
    """Play one game. Returns the states, actions, rewards and ends of
    every decision, and the final info."""
    states, actions, rewards, ends = [], [], [], []
    observation, info = env.reset(seed=seed)
    finished = False
    while not finished:
        action = player.choose(observation)
        states.append(observation)
        actions.append(action)
        observation, reward, played, finished, ended, info = (
            play_decision(env, action, player.decide_every)
        )
        rewards.append(reward)
        ends.append(ended)
    return states, actions, rewards, ends, info


def returns_of(rewards, ends):
    """For each decision, the discounted sum of the rewards from there to
    the end of that life: reward + gamma * next + gamma^2 * the one after,
    and so on. Worked backwards, so each step reuses the one after it."""
    returns = [0.0] * len(rewards)
    following = 0.0
    for step in reversed(range(len(rewards))):
        if ends[step]:
            following = 0.0
        following = rewards[step] + DISCOUNT * following
        returns[step] = following
    return returns


def update(player, optimizer, states, actions, returns):
    states = torch.as_tensor(np.array(states))
    actions = torch.as_tensor(actions)
    returns = torch.as_tensor(returns, dtype=torch.float32)

    chances, _ = player.network(states)
    log_chances = chances.log_prob(actions)

    # Better than usual means better than the average return in this
    # batch of games.
    advantages = returns - returns.mean()
    advantages = advantages / (advantages.std() + 1e-8)

    policy_loss = -(log_chances * advantages).mean()
    entropy = chances.entropy().mean()
    loss = policy_loss - ENTROPY_BONUS * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(player.network.parameters(), 0.5)
    optimizer.step()
    return entropy.item()


def test(player, games):
    player.greedy = True
    env = BrickEnv()
    scores = []
    for game in range(games):
        scores.append(
            play_game(player, env, seed=100_000 + game)["score"]
        )
    player.greedy = False
    return scores


def main():
    torch.set_num_threads(1)
    player = PolicyPlayer()
    optimizer = torch.optim.Adam(
        player.network.parameters(), LEARNING_RATE
    )
    env = BrickEnv(max_frames=TRAINING_FRAMES)
    log = ProgressLog("reinforce", extra="entropy", report_every=40)

    frames = 0
    game = 0
    entropy = 0.0
    next_test = TEST_EVERY
    best_average = -1
    while frames < TOTAL_FRAMES:
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(GAMES_PER_UPDATE):
            game += 1
            states, actions, rewards, ends, info = play_and_record(
                player, env, seed=game
            )
            batch_states += states
            batch_actions += actions
            batch_returns += returns_of(rewards, ends)
            frames += info["frames"]
            log.game_over(info, extra=round(entropy, 4))
        entropy = update(
            player,
            optimizer,
            batch_states,
            batch_actions,
            batch_returns,
        )

        if frames >= next_test:
            next_test += TEST_EVERY
            scores = test(player, TEST_GAMES)
            log.test_result(scores)
            average = sum(scores) / len(scores)
            if average > best_average:
                best_average = average
                player.save(log.folder / "best.pt")
                print(f"       new best network: {average:.1f} bricks")
    log.close()


if __name__ == "__main__":
    main()
