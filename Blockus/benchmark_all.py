import multiprocessing
import random
import numpy as np
from BlockusGame import BlockusGame
from BlockusPlayer import BlockusPlayer
from BlockusNNPlayer import BlockusNNPlayer

import multiprocessing
import os
import random
import numpy as np


def game_constructor(i, model_paths = []):
    return BlockusGame(
        board_size=(14,14),
        timeout=45,
        logger_args = None,
        render_mode = "",
        gather_data = "",
        model_paths=model_paths,
        )

def players_constructor(i, model_path = ""):
    random_players = [BlockusPlayer(name=f"Player{j}_{i}",
                                    logger_args=None,
                                    )
                for j in range(3)]
    if not model_path:
        test_player = BlockusPlayer(name=f"TestPlayer_{i}",
                                    logger_args=None,
                                    )
    else:
        test_player = BlockusNNPlayer(name=f"TestPlayer_{i}",
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
    num_games = 10
    num_cpus = 10
    win_percents = {}
    for model_path in os.listdir("/home/ilmari/python/RLFramework/BlockusModels/"):
    #for model_path in ["model_all_data.tflite"]:
        if not model_path.endswith(".tflite"):
            continue
        model_path = os.path.join("/home/ilmari/python/RLFramework/BlockusModels/", model_path)
        print(f"Testing model: {model_path}")
        with multiprocessing.Pool(num_cpus) as p:
            results = p.map(run_game, [(i, model_path, random.randint(0, 2**32-1)) for i in range(num_games)])

        # Find how many times the test player won
        num_wins = 0
        total_games = 0
        for result in results:
            print(result)
            test_player_pid = 0
            test_player_name = ""
            for player_json in result.player_jsons:
                if "TestPlayer" in player_json["name"]:
                    test_player_pid = player_json["pid"]
                    test_player_name = player_json["name"]
                    break
            if not result.successful:
                print(f"Game failed: {result}")
                continue
            if result.winner == test_player_name:
                num_wins += 1
            total_games += 1
        
        print(f"Test player won {num_wins} out of {total_games} games")
        print(f"Win rate: {num_wins / total_games}")
        win_percents[model_path] = num_wins / total_games
    sorted_win_percents = sorted(win_percents.items(), key=lambda x: x[1], reverse=True)
    for model_path, win_percent in sorted_win_percents:
        print(f"{model_path}: {win_percent}")
        print()
        

