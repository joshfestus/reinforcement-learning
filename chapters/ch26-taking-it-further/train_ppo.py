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
    --return-bonus R   reward for each return of the ball (normally 0.1)
    --frame-cost C     reward taken away every frame (normally 0.001)
    --no-scaling       give the player the numbers in pixels, unscaled
    --no-bricks        hide the bricks from the player
    --no-paddle-size   hide the paddle's width from the player
    --constant-rate    keep the learning rate fixed instead of sliding
    --discount G       gamma, the discount per decision (normally 0.99)
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


def option(name):
    """The word typed after name on the command line, or None."""
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return None


def main():
    torch.set_num_threads(1)
    total_frames = TOTAL_FRAMES
    if option("--frames"):
        total_frames = int(option("--frames"))

    # Each option that changes training also changes the run's name, so
    # that runs with different settings never share a folder.
    decide_every = 4
    run_name = "ppo-own"
    settings = {}
    if option("--decide-every"):
        decide_every = int(option("--decide-every"))
        run_name += f"-every-{decide_every}"
    if option("--return-bonus"):
        settings["paddle_hit_reward"] = float(option("--return-bonus"))
        run_name += f"-bonus-{option('--return-bonus')}"
    if option("--frame-cost"):
        settings["frame_cost"] = float(option("--frame-cost"))
        run_name += f"-cost-{option('--frame-cost')}"
    hidden = {
        "--no-scaling": "scaled",
        "--no-bricks": "see_bricks",
        "--no-paddle-size": "see_paddle_size",
    }
    for switch, setting in hidden.items():
        if switch in sys.argv:
            settings[setting] = False
            run_name += switch[1:]
    discount = 0.99
    if option("--discount"):
        discount = float(option("--discount"))
        run_name += f"-discount-{option('--discount')}"
    constant_rate = "--constant-rate" in sys.argv
    if constant_rate:
        run_name += "-constant-rate"

    player = PolicyPlayer(decide_every=decide_every)
    optimizer = torch.optim.Adam(
        player.network.parameters(), LEARNING_RATE
    )
    log = ProgressLog(run_name, extra="entropy", report_every=40)
    games = SideBySide(GAMES_AT_ONCE, log, **settings)

    entropy = 0.0
    next_test = TEST_EVERY
    best_average = -1
    while games.frames < total_frames:
        if not constant_rate:
            for group in optimizer.param_groups:
                group["lr"] = sliding(
                    LEARNING_RATE, 0.0, games.frames / total_frames
                )
        played = games.play(
            player, DECISIONS_PER_UPDATE, round(entropy, 4)
        )
        states, actions, log_chances, values, rewards, cuts, last = played
        advantages = advantages_of(rewards, values, cuts, last, discount)
        returns = advantages + values
        batch = flatten(states, actions, log_chances, advantages, returns)
        entropy = improve(player, optimizer, batch)

        if games.frames >= next_test:
            next_test += TEST_EVERY
            best_average = keep_if_best(
                player, log, best_average, **settings
            )
    log.close()


if __name__ == "__main__":
    main()
