import argparse
import os
import numpy as np

from RLFramework.simulate import simulate_games
from MoskaGame import MoskaGame
from MoskaPlayer import MoskaPlayer
from MoskaNNPlayer import MoskaNNPlayer


def game_constructor(i, model_base_folder):
    model_paths = list(filter(lambda path: path.endswith(".tflite"), os.listdir(model_base_folder)))
    model_paths = [os.path.abspath(os.path.join(model_base_folder,model_path)) for model_path in model_paths]
    return MoskaGame(
        timeout=80,
        logger_args = None,
        render_mode = "",
        gather_data = f"gathered_data_{i}.csv",
        model_paths=model_paths,
        )

def players_constructor(i, model_base_folder):
    
    files_in_model_folder = os.listdir(model_base_folder)
    # Filter only the tflite files
    all_model_paths = [os.path.abspath(os.path.join(model_base_folder, model_file)) for model_file in files_in_model_folder if model_file.endswith(".tflite")]
    if len(all_model_paths) == 0:
        print(f"No models found in {model_base_folder}")
        return [MoskaPlayer(name=f"Player{j}_{i}", logger_args=None) for j in range(4)]
    # Find the epoch number of each model
    epoch_nums = []
    for model_path in all_model_paths:
        model_base_path = model_path.split("/")[-1]
        # Now we have 'model_%d.tflite'
        epoch_num = int(model_base_path.split("_")[1].split(".")[0])
        epoch_nums.append(epoch_num)
    # Now, we assume the model numbers are all 0-epoch_num
    assert set(epoch_nums) == set(range(max(epoch_nums)+1)) == set(range(len(epoch_nums))), f"Epoch numbers are not in the expected range: {epoch_nums}"
    # In the simulation, we play games with the current and previous models
    # To do that, we'll create a dict of players, where the keys are the model paths, and the values are the weights
    # for picking that player. The weight is the epoch number.
    models_weighted_set = {model_path_ : epoch_num_ + 1 for model_path_, epoch_num_ in zip(all_model_paths, epoch_nums)}
    # Softmax the weights
    model_weights = np.array(list(models_weighted_set.values()))
    model_weights = np.exp(model_weights) / np.sum(np.exp(model_weights))
    
    models_weighted_set = {model_path_ : w for model_path_, w in zip(all_model_paths, model_weights)}
    #print(models_weighted_set)
    players = [MoskaNNPlayer(name=f"Player{j}_{i}",
                                    logger_args=None,
                                    model_path=np.random.choice(list(models_weighted_set.keys()), p=list(models_weighted_set.values())),
                                    move_selection_temp=1.0,
                                    )
                for j in range(4)]
    
    return players

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark all models in a directory.')
    parser.add_argument('--folder', type=str, required=True, help='Folder where to save the games.')
    parser.add_argument('--model_base_folder', type=str, required=True, help='The folder containing the models.')
    parser.add_argument('--num_games', type=int, required=True, help='The number of games to play for each model.')
    parser.add_argument('--num_cpus', type=int, help='The number of CPUs to use.', default=os.cpu_count()-1)
    parser.add_argument('--num_files', type=int, help='The number of files to use.', default=-1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    os.makedirs(args.folder, exist_ok=True)
    os.makedirs(args.model_base_folder, exist_ok=True)
    folder = os.path.abspath(args.folder)
    model_base_folder = os.path.abspath(args.model_base_folder)
    
    def _players_constructor(i):
        return players_constructor(i, model_base_folder)
    
    def _game_constructor(i):
        return game_constructor(i, model_base_folder)

    num_games = args.num_games
    num_cpus = args.num_cpus
    print(f"folder: {folder}")
    print(f"model_base_folder: {model_base_folder}")
    num_files = args.num_files
    res = simulate_games(game_constructor=_game_constructor,
                         players_constructor=_players_constructor,
                         folder=folder,
                         num_games=num_games,
                        num_cpus=num_cpus,
                        num_files=num_files,
                        exists_ok=True,
                        return_results=True,
                        )