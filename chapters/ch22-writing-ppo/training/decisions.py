"""Playing the game one decision at a time, as the learning players do."""


def play_decision(env, action, frames):
    """Repeat one action for up to `frames` frames.

    Stops early if the game finishes or a life is lost, so that every
    decision belongs to one life. Returns the new observation, the reward
    added up over the frames, the number of frames played, whether the
    game has finished, whether the decision ended a life (or the game),
    and the latest info.
    """
    lives_before = env.game.lives
    total_reward = 0
    played = 0
    for _ in range(frames):
        result = env.step(action)
        observation, reward, terminated, truncated, info = result
        total_reward += reward
        played += 1
        if terminated or truncated or env.game.lives < lives_before:
            break

    finished = terminated or truncated
    ended = terminated or env.game.lives < lives_before
    return observation, total_reward, played, finished, ended, info


def sliding(start, end, progress):
    """A value that moves from start to end as progress goes from 0 to
    1, and stays at end after that."""
    progress = min(1.0, progress)
    return start + progress * (end - start)
