import argparse
import os
import random
import numpy as np

from RLFramework.simulate import simulate_games
from MoskaGame import MoskaGame
from MoskaPlayer import MoskaPlayer
from MoskaNNPlayer import MoskaNNPlayer

def game_constructor(i):
    return MoskaGame(
        timeout=25,
        logger_args = None,
        render_mode = "",
        gather_data = f"",
        model_paths=[], #Fetch the model_paths from the NN player
        )

def players_constructor(i, model_file):
    if not os.path.exists(model_file):
        raise Exception(f"No file found: {model_file}")
    random_players = [MoskaPlayer(name=f"Player{j}_{i}",
                                    logger_args=None,
                                    )
                for j in range(3)]
    
    nn_player = MoskaNNPlayer(name=f"NNPlayer_{i}",
                              model_path=model_file,
    )
    players = random_players + [nn_player]
    random.shuffle(players)
    return players

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark all models in a directory.')
    parser.add_argument('--output_file', type=str, required=False, help='File where to save the benchmark result.',default="moska_benchmark.out")
    parser.add_argument('--model_file', type=str, required=True, help='The NN to benchmark.')
    parser.add_argument('--output_folder', type=str, required=False, default="MoskaBenchmarkFolder")
    parser.add_argument('--opponent', type=str, required=False, default="random",help="benchmark type")
    parser.add_argument('--num_games', type=int, required=True, help='The number of games to play for each model.')
    parser.add_argument('--num_cpus', type=int, help='The number of CPUs to use.', default=os.cpu_count()-1)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    model_file = os.path.abspath(args.model_file)
    output_file = os.path.abspath(args.output_file)

    def _game_constructor(i):
        return game_constructor(i)
    
    def _players_constructor(i):
        return players_constructor(i, model_file)

    num_games = args.num_games
    num_cpus = args.num_cpus
    folder = args.output_folder
    res = simulate_games(game_constructor=_game_constructor,
                         players_constructor=_players_constructor,
                         folder=folder,
                         num_games=num_games,
                        num_cpus=num_cpus,
                        num_files=-1,
                        exists_ok=True,
                        return_results=True,
                        )
    
    successful_games = [r for r in res if r.successful]
    print(f"Number of succesful games: {len(successful_games)}")
    
    num_not_losses = {}
    for result in successful_games:
        for player in result.player_jsons:
            pname = player["name"].split("_")[0]
            if pname not in num_not_losses:
                num_not_losses[pname] = 0
            num_not_losses[pname] += player["score"]
            
    print(f"Number of not lost games: ", num_not_losses)
    
    num_losses = {pname:len(successful_games) - num_not_losses[pname] for pname in num_not_losses}
    
    print(f"Number of lost games:", num_losses)
    
    loss_percents = {pname : num_losses[pname] / len(successful_games) for pname in num_not_losses}
    
    print(f"Loss percents:", loss_percents)
    
    
    
    
    
    
    
    