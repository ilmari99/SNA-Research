from collections import Counter
import gc
import json
import multiprocessing
import os
import random
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PentobiGTP import PentobiGTP
from PentobiPlayers import PentobiInternalPlayer, PentobiNNPlayer, PentobiInternalEpsilonGreedyPlayer
from utils import TFLiteModel
import argparse

def play_pentobi(i, seed, player_maker, save_data_file = "", proc_args = {}):
    
    default_proc_args = {
        "command": None,
        "book": None,
        "config": None,
        "game": "classic",
        "level": 1,
        "seed": seed,
        "showboard": False,
        "nobook": False,
        "noresign": True,
        "quiet": False,
        "threads": 1,
    }
    
    proc = PentobiGTP(**{**default_proc_args, **proc_args})
    
    players = player_maker(proc)
    
    np.random.seed(seed)
    random.seed(seed)
    
    while not proc.is_game_finished():
        pid = proc.pid
        player = players[pid-1]
        player.play_move()
    if save_data_file:
        proc.write_states_to_file(save_data_file)
    score = list(proc.score)
    proc.close()
    return score

def shuffle_players_func(players):
    random.shuffle(players)
    for i, player in enumerate(players):
        player.pid = i+1
    return players

def player_maker(proc, model_paths = [], shuffle_players = True):
    if not model_paths:
        # Make four players using the internal player
        players = [PentobiInternalEpsilonGreedyPlayer(i+1, proc,epsilon=1.0) for i in range(4)]
        return players
    # Load all the models
    models = [TFLiteModel(path) for path in model_paths]
    # Make four players using a random model
    players = []
    for i in range(4):
        model = random.choice(models)
        players.append(PentobiNNPlayer(i+1, proc, model,move_selection_strategy="weighted", move_selection_kwargs={"top_p": 1.0}))
    if shuffle_players:
        players = shuffle_players_func(players)
    return players

def play_pentobi_wrapper(args):
    return play_pentobi(*args)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # Load the environment variables from env.json
    if os.path.exists('env.json'):
        with open('env.json') as f:
            env_vars = json.load(f)
            print(env_vars)
    else:
        env_vars = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=1000, help="Number of games")
    parser.add_argument("--num_cpus", type=int, default=10, help="Number of CPUs")
    parser.add_argument("--pentobi_gtp", type=str, default=env_vars.get('pentobi_gtp', None), help="Path to pentobi-gtp")
    parser.add_argument("--data_folder", type=str, default=env_vars.get('data_folder', "./Data"), help="Path to data folder")
    parser.add_argument("--model_folder", type=str, default=env_vars.get('model_folder', "./Models"), help="Path to model folder")
    args = parser.parse_args()
    
    print(args)
    num_games = args.num_games
    num_cpus = args.num_cpus
    #os.environ["PENTOBI_GTP"] = os.path.abspath(args.pentobi_gtp)
    data_folder = os.path.abspath(args.data_folder)
    model_folder = os.path.abspath(args.model_folder)
    pentobi_gtp = os.path.abspath(args.pentobi_gtp)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)
    
    model_paths = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith(".tflite")]
    
    def _player_maker(proc):
        return player_maker(proc, model_paths)
    
    def arg_generator(num_games):
        kwargs = {
            "command": pentobi_gtp,
            "level": 1,
            "threads": 1,
            "showboard": False,
            "nobook": False,
            "quiet": True,
        }
        for i in range(num_games):
            seed = np.random.randint(2**32)
            file = f"{data_folder}/data_{i}.csv"
            yield (i, seed, _player_maker, file, kwargs)
    
    # Play the games in parallel
    results = []
    with multiprocessing.Pool(num_cpus) as pool:
        gen = pool.imap_unordered(play_pentobi_wrapper, arg_generator(num_games))
        while True:
            try:
                result = next(gen)
                results.append(result)
                if len(results) % 100 == 0:
                    print(f"Games played: {len(results)}", end="\r")
            except StopIteration:
                break
    #print(results)
    
    # Calculate the average score
    results = np.array(results)
    mean_scores = np.mean(results, axis=0)
    winners = np.argmax(results, axis=1) + 1
    # Hm times each player won
    num_wins = Counter(winners)
    num_wins = [num_wins[i+1] for i in range(4)]
    print(f"Mean scores: {mean_scores}")
    print(f"Number of wins: {num_wins}")