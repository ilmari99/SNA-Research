import multiprocessing
import os
import numpy as np

from MoskaGame import MoskaGame
from MoskaPlayer import MoskaPlayer
from MoskaHumanPlayer import MoskaHumanPlayer
from MoskaNNPlayer import MoskaNNPlayer


def game_constructor(i):
    model_paths = ["C:\\Users\\ilmari\\Desktop\\Python\\RLFramework\\MoskaModels\\model_3.tflite"]
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
        test_player = MoskaNNPlayer(name=f"Player3_{i}",
                                    logger_args=None,
                                    model_path=model_path,
                                    move_selection_temp=0.0,
                                    )
    players = random_players + [test_player]   
    return players

def run_game(args):
    if len(args) == 1:
        i = args[0]
        model_path = ""
    else:
        i, model_path = args
    game = game_constructor(i)
    players = players_constructor(i, model_path)
    res = game.play_game(players)
    return res

if __name__ == "__main__":
    # Run games with multiprocessing pool
    num_games = 50
    model_path = "C:\\Users\\ilmari\\Desktop\\Python\\RLFramework\\MoskaModels\\model_3.tflite"
    num_cpus = 10
    with multiprocessing.Pool(num_cpus) as p:
        results = p.map(run_game, [(i, model_path) for i in range(num_games)])

    # Find how many times the test player won
    num_wins = 0
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
        if result.finishing_order[-1] != test_player_pid:
            num_wins += 1
        total_games += 1
    
    print(f"Test player won {num_wins} out of {total_games} games.")
    print(f"Win rate: {num_wins / total_games}")
