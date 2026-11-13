"""Train a policy player with PPO, written out in full.

It plays like the actor and critic of train_actor_critic.py, eight games
side by side, but collects 256 decisions in each before learning, and
then learns from those 2,048 decisions in several passes, never letting
any action's chance move by more than 20% in one update. That limit is
the "proximal" in Proximal Policy Optimization, and it is what keeps one
unlucky batch from undoing what was learned before.

Run it with:  uv run train_ppo.py
Options:
    --frames N         train for N frames instead of 6 million
    --decide-every N   keep each action for N frames (normally 4)
"""

import sys

import torch

from players.policy import PolicyPlayer
from train_actor_critic import (
    SideBySide,
    advantages_of,
    flatten,
    keep_if_best,
)
from training.decisions import sliding
from training.progress import ProgressLog

TOTAL_FRAMES = 6_000_000
GAMES_AT_ONCE = 8
DECISIONS_PER_UPDATE = 256
PASSES = 4
MINIBATCH = 256
CLIP = 0.2
LEARNING_RATE = 0.0003
VALUE_WEIGHT = 0.5
ENTROPY_BONUS = 0.01
TEST_EVERY = 250_000


def improve(player, optimizer, batch):
    """Several passes over the batch, in shuffled minibatches."""
    states, actions, old_log_chances, advantages, returns = batch
    size = len(actions)
    for _ in range(PASSES):
        order = torch.randperm(size)
        for start in range(0, size, MINIBATCH):
            picks = order[start : start + MINIBATCH]
            chances, values = player.network(states[picks])
            log_chances = chances.log_prob(actions[picks])

            # How much more (or less) likely each action is now than when
            # it was taken. 1 means unchanged.
            ratio = torch.exp(log_chances - old_log_chances[picks])
            advantage = advantages[picks]
            advantage = (advantage - advantage.mean()) / (
                advantage.std() + 1e-8
            )
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantage
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = ((values - returns[picks]) ** 2).mean()
            entropy = chances.entropy().mean()
            loss = (
                policy_loss
                + VALUE_WEIGHT * value_loss
                - ENTROPY_BONUS * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                player.network.parameters(), 0.5
            )
            optimizer.step()
    return entropy.item()


def main():
    torch.set_num_threads(1)
    total_frames = TOTAL_FRAMES
    if "--frames" in sys.argv:
        total_frames = int(sys.argv[sys.argv.index("--frames") + 1])

    decide_every = 4
    run_name = "ppo-own"
    if "--decide-every" in sys.argv:
        decide_every = int(sys.argv[sys.argv.index("--decide-every") + 1])
        run_name = f"ppo-own-every-{decide_every}"

    player = PolicyPlayer(decide_every=decide_every)
    optimizer = torch.optim.Adam(
        player.network.parameters(), LEARNING_RATE
    )
    log = ProgressLog(run_name, extra="entropy", report_every=40)
    games = SideBySide(GAMES_AT_ONCE, log)

    entropy = 0.0
    next_test = TEST_EVERY
    best_average = -1
    while games.frames < total_frames:
        for group in optimizer.param_groups:
            group["lr"] = sliding(
                LEARNING_RATE, 0.0, games.frames / total_frames
            )
        played = games.play(
            player, DECISIONS_PER_UPDATE, round(entropy, 4)
        )
        states, actions, log_chances, values, rewards, cuts, last = played
        advantages = advantages_of(rewards, values, cuts, last)
        returns = advantages + values
        batch = flatten(states, actions, log_chances, advantages, returns)
        entropy = improve(player, optimizer, batch)

        if games.frames >= next_test:
            next_test += TEST_EVERY
            best_average = keep_if_best(player, log, best_average)
    log.close()


if __name__ == "__main__":
    main()
