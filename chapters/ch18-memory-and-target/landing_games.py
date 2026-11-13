"""Play 20 benchmark games steered by each trained landing network.

Run it with:  uv run landing_games.py
(after training the NumPy, PyTorch and Adam versions)
"""

from collections import Counter

from brickgame.env import BrickEnv
from players import play_game
from players.landing_net import LandingNetwork
from players.landing_net_torch import TorchLandingNetwork
from players.landing_player import LandingPlayer
from training.progress import RUNS_FOLDER

GAMES = 20


def load_networks():
    numpy_network = LandingNetwork()
    numpy_network.load(RUNS_FOLDER / "landing-numpy" / "network.npz")
    torch_network = TorchLandingNetwork()
    torch_network.load(RUNS_FOLDER / "landing-torch" / "network.pt")
    adam_network = TorchLandingNetwork(optimizer="adam")
    adam_network.load(RUNS_FOLDER / "landing-adam" / "network.pt")
    return {
        "NumPy": numpy_network,
        "PyTorch": torch_network,
        "PyTorch with Adam": adam_network,
    }


def main():
    env = BrickEnv()
    for name, network in load_networks().items():
        player = LandingPlayer(network)
        games = [play_game(player, env, seed=g) for g in range(1, 21)]
        bricks = sum(game["score"] for game in games) / GAMES
        lives = sum(game["lives_lost"] for game in games) / GAMES
        endings = Counter(game["ending"] for game in games)
        print(
            f"{name:18} {bricks:5.1f} bricks, {lives:.1f} lives lost, "
            f"{endings['lost']} lost, {endings['stuck']} stuck"
        )


if __name__ == "__main__":
    main()
