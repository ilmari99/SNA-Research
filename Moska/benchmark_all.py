import multiprocessing
import random
import numpy as np
from MoskaGame import MoskaGame
from MoskaPlayer import MoskaPlayer
from MoskaHumanPlayer import MoskaHumanPlayer
from MoskaHeuristicPlayer import MoskaHeuristicPlayer
from MoskaNNPlayer import MoskaNNPlayer

import multiprocessing
import os
import random
import numpy as np


def game_constructor(i, model_paths = []):
    return MoskaGame(
        timeout=15,
        logger_args = None,
        render_mode = "",
        gather_data = "",
        model_paths=model_paths,
        )

def players_constructor(i, model_path = ""):
    random_players = [MoskaPlayer(name=f"Player{j}_{i}",
                                    logger_args=None,
                                    )
                for j in range(3)]
    if not model_path:
        test_player = MoskaPlayer(name=f"TestPlayer_{i}",
                                    logger_args=None,
                                    )
    else:
        test_player = MoskaNNPlayer(name=f"TestPlayer_{i}",
                                    logger_args=None,
                                    model_path=model_path,
                                    move_selection_temp=0.0,
                                    )
    players = random_players + [test_player]
    random.shuffle(players)
    return players

def run_game(args):
    i, model_path, seed = args
    random.seed(seed)
    np.random.seed(seed)
    game = game_constructor(i, [model_path])
    players = players_constructor(i, model_path)
    res = game.play_game(players)
    return res

if __name__ == "__main__":
    num_games = 600
    num_cpus = 30
    loss_percents = {}
    for model_path in os.listdir("/home/ilmari/RLFramework/MoskaModels/"):
        if not model_path.endswith(".tflite"):
            continue
        model_path = os.path.join("/home/ilmari/RLFramework/MoskaModels/", model_path)
        print(f"Testing model: {model_path}")
        with multiprocessing.Pool(num_cpus) as p:
            results = p.map(run_game, [(i, model_path, random.randint(0,2**32)) for i in range(num_games)])

        # Find how many times the test player won
        num_losses = 0
        total_games = 0
        for result in results:
            print(result)
            test_player_pid = 0
            for player_json in result.player_jsons:
                if "TestPlayer" in player_json["name"]:
                    test_player_pid = player_json["pid"]
                    break
            if not result.successful:
                print(f"Game failed: {result}")
                continue
            if result.finishing_order[-1] == test_player_pid:
                num_losses += 1
            total_games += 1
        
        print(f"Test player lost {num_losses} out of {total_games} games.")
        print(f"Loss rate: {num_losses / total_games}")
        loss_percents[model_path] = num_losses / total_games
    sorted_loss_percents = sorted(loss_percents.items(), key=lambda x: x[1])
    for model_path, loss_percent in sorted_loss_percents:
        print(f"{model_path}: {loss_percent}")
        

