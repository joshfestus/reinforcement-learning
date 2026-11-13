"""The return: the reward still to come from each frame, with rewards
further ahead counted for less."""


def discounted_returns(rewards, discount):
    """For each frame, its own reward plus `discount` times the return
    of the frame after it. Worked backwards from the last frame, so
    each frame reuses the answer for the one after it."""
    returns = [0.0] * len(rewards)
    following = 0.0
    for frame in reversed(range(len(rewards))):
        following = rewards[frame] + discount * following
        returns[frame] = following
    return returns
