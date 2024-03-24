"""
This module contains the implementation of a benchmarking tool for TicTacToe game models.
It simulates games between different models and records their performance.
"""

import random
import os
import argparse
from TTTGame import TTTGame
from TTTPlayer import TTTPlayer
from TTTPlayerNeuralNet import TTTPlayerNeuralNet
from TTTResult import TTTResult
from RLFramework import simulate_games


class PickleableFunction:
    """ Create a Callable object that takes in a global function,
    and wraps it by giving it some default arguments.
    """

    def __init__(self, func, **pargs):
        self.func = func
        self.pargs = pargs

    def __call__(self, *args, **kwargs):
        return self.func(*args, **self.pargs, **kwargs)


def game_constructor(i):
    return TTTGame(board_size=(3, 3),
                   logger_args=None,
                   render_mode="",
                   gather_data="",
                   custom_result_class=TTTResult,
                   )


def players_constructor(i, model_path):
    players = [TTTPlayerNeuralNet(model_path=model_path,
                                  name=f"NeuralNetPlayer1_{i}",
                                  move_selection_temp=0,
                                  logger_args=None),
               TTTPlayer(name=f"Player2_{i}", logger_args=None)]
    random.shuffle(players)
    return players


def benchmark_all_tflite_files_in_folder(
        folder, num_games, num_proc, num_files):
    """ Benchmark all tflite files in a folder.
    """
    model_performance = {}
    for f in os.listdir(folder):
        if f.endswith(".tflite"):
            model_path = os.path.join(folder, f)
            print(f"Benchmarking model: {model_path}")
            res = simulate_games(game_constructor,
                                 PickleableFunction(
                                     players_constructor, model_path=model_path),
                                 folder, num_games,
                                 num_files=num_files,
                                 num_cpus=num_proc,
                                 return_results=True
                                 )
            player_name_to_wins = {"NeuralNetPlayer1": 0, "Player2": 0}
            for r in res:
                if r.winner:
                    player_name_to_wins[r.winner.split("_")[0]] += 1
            print(player_name_to_wins)
            model_performance[model_path] = player_name_to_wins["NeuralNetPlayer1"] / num_games
    # Sort and print
    sorted_model_performance = sorted(
        model_performance.items(),
        key=lambda x: x[1],
        reverse=True)
    for k, v in sorted_model_performance:
        print(f"{k}: {v}")

    return model_performance


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Simulate TicTacToe games")
    argparser.add_argument(
        "--tflite_folder",
        help="Folder to save the data to",
        default="/home/ilmari/python/RLFramework/models")
    argparser.add_argument(
        "--num_proc",
        help="Number of processes to use",
        default=12)
    argparser.add_argument(
        "--num_games",
        help="Number of games to play",
        default=1000)
    argparser.add_argument(
        "--folder",
        help="Folder to save the data to",
        default="BenchmarkData")
    argparser.add_argument(
        "--num_files",
        help="Number of files to save the data to",
        default=-1)
    argparser = argparser.parse_args()

    num_proc = int(argparser.num_proc)
    num_games = int(argparser.num_games)
    folder = argparser.folder
    num_files = int(argparser.num_files)

    model_performance = benchmark_all_tflite_files_in_folder(
        argparser.tflite_folder, num_games, num_proc, num_files)