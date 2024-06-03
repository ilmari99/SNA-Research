import multiprocessing
import random
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

def players_constructor(i, model_path = ""):
    random_players = [BlokusNNPlayer(name=f"NNPlayer{j}_{i}",
                                    model_path=model_path,
                                    logger_args=None,
                                    move_selection_temp=0.0,
                                    )
                for j in range(3)]
    test_player = BlokusPlayer(name=f"RandomPlayer_{i}",
                                logger_args=None,
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
    parser = argparse.ArgumentParser(description='Benchmark all models in a directory.')
    parser.add_argument('--folder', type=str, required=True, help='The folder containing the models.')
    parser.add_argument('--num_games', type=int, required=True, help='The number of games to play for each model.')
    parser.add_argument('--num_cpus', type=int, help='The number of CPUs to use.', default=os.cpu_count()-1)
    args = parser.parse_args()
    print(args)

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
            results = p.map(run_game, [(i, model_path, random.randint(0, 2**32-1)) for i in range(num_games)])

        # Find how many times the test player won
        num_wins = 0
        num_ties = 0
        total_games = 0
        for result in results:
            print(result)
            if not result.successful:
                print(f"Game failed: {result}")
                continue
            for i, player_json in enumerate(result.player_jsons):
                if "RandomPlayer" in player_json["name"]:
                    random_player_idx = i
                    break
            
            player_scores = [player_json["score"] for player_json in result.player_jsons]
            winner = result.winner
            if winner is None:
                # Check the random player's score
                random_player_score = result.player_jsons[random_player_idx]["score"]
                if random_player_score == max(player_scores):
                    num_ties += 1
            elif "RandomPlayer" in winner:
                num_wins += 1
            total_games += 1   
            
        print(f"Random player won {num_wins} out of {total_games} games")
        print(f"Random player tied {num_ties} out of {total_games} games")
        print(f"Win rate: {num_wins / total_games}")
        print(f"Tie rate: {num_ties / total_games}")
        

