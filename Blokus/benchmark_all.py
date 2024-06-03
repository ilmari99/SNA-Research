import multiprocessing
import random
import time
import numpy as np
import multiprocessing
import os
import random
import numpy as np
import argparse

from BlokusGame import BlokusGame
from BlokusPlayer import BlokusPlayer
from BlokusNNPlayer import BlokusNNPlayer
from BlokusGreedyPlayer import BlokusGreedyPlayer


def game_constructor(i, model_paths = []):
    return BlokusGame(
        board_size=(20,20),
        timeout=45,
        logger_args = None,
        render_mode = "",
        gather_data = "",
        model_paths=model_paths,
        )

def players_constructor(i, model_path = "", opponent_type = "random"):
    assert opponent_type in ["random", "greedy"], f"Player type must be either 'random' or 'greedy', not {opponent_type}"
    if opponent_type == "random":
        opponent_players = [BlokusPlayer(name=f"RandomPlayer{j}_{i}",
                                        logger_args=None,
                                        )
                    for j in range(3)]
    elif opponent_type == "greedy":
        opponent_players = [BlokusGreedyPlayer(name=f"GreedyPlayer{j}_{i}",
                                        logger_args=None,
                                        action_selection_strategy="epsilon_greedy",
                                        action_selection_args=((), {"epsilon" : 0.1}),
                                        )
                    for j in range(3)]
    if not model_path:
        test_player = BlokusPlayer(name=f"TestPlayer_{i}",
                                    logger_args=None,
                                    )
    else:
        test_player = BlokusNNPlayer(name=f"TestPlayer_{i}",
                                    logger_args=None,
                                    model_path=model_path,
                                    action_selection_strategy="greedy",
                                    )
    players = opponent_players + [test_player]
    random.shuffle(players)
    return players

def run_game(args):
    i, model_path, opponent_type, seed = args
    random.seed(seed)
    np.random.seed(seed)
    game = game_constructor(i, [model_path])
    players = players_constructor(i, model_path, opponent_type)
    res = game.play_game(players)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark all models in a directory.')
    parser.add_argument('--folder', type=str, required=True, help='The folder containing the models.')
    parser.add_argument('--num_games', type=int, required=True, help='The number of games to play for each model.')
    parser.add_argument('--num_cpus', type=int, help='The number of CPUs to use.', default=os.cpu_count()-1)
    parser.add_argument("--opponent_type", type=str, help="The type of opponent to use. Must be either 'random' or 'greedy'.", default="random")
    args = parser.parse_args()
    print(args)
    
    t_start = time.time()
    num_games = args.num_games
    num_cpus = args.num_cpus
    win_percents = {}
    folder = os.path.abspath(args.folder)
    for model_path in os.listdir(folder):
        if not model_path.endswith(".tflite"):
            continue
        model_path = os.path.join(folder, model_path)
        print(f"Testing model: {model_path}")
        with multiprocessing.Pool(num_cpus) as p:
            results = p.map(run_game, [(i, model_path, args.opponent_type, random.randint(0, 2**32-1)) for i in range(num_games)])

        # Find how many times the test player won
        num_wins = 0
        num_ties = 0
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
            if result.winner == None:
                num_ties += 1 
            if result.winner == test_player_name:
                num_wins += 1
            total_games += 1
        print(f"Test player won {num_wins} out of {total_games} games")
        print(f"Test player tied {num_ties} out of {total_games} games")
        print(f"Win rate: {num_wins / total_games}")
        print(f"Tie rate: {num_ties / total_games}")
        win_percents[model_path] = num_wins / total_games
    t_end = time.time()
    print(f"Time taken: {t_end - t_start}")
    sorted_win_percents = sorted(win_percents.items(), key=lambda x: x[1], reverse=True)
    for model_path, win_percent in sorted_win_percents:
        print(f"{model_path}: {win_percent}")
        print()
        

