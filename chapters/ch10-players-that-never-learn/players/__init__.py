from players.chaser import Chaser
from players.predictor import Predictor
from players.random_player import RandomPlayer

# The players that never learn. Each one is ready the moment it is made.
PLAYERS = {
    "random": RandomPlayer,
    "chaser": Chaser,
    "predictor": Predictor,
}


def play_game(player, env, seed=None):
    """Play one game to the end. Returns the final info dictionary."""
    observation, info = env.reset(seed=seed)
    while True:
        action = player.act(observation)
        result = env.step(action)
        observation, reward, terminated, truncated, info = result
        if terminated or truncated:
            return info
