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
    
    num_moves = 0
    while not proc.is_game_finished():
        pid = proc.pid
        #print(f"Player {pid} playing")
        player = players[pid-1]
        player.play_move()
        num_moves += 1
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

def player_maker_selfplay(proc, model_paths = [], shuffle_players = True):
    if not model_paths:
        # Make four players using the internal player
        players = [PentobiInternalEpsilonGreedyPlayer(i+1, proc,epsilon=1.0) for i in range(4)]
        return players
    # Load all the models
    models = [TFLiteModel(path) for path in model_paths]
    model_nums = []
    for model_path in model_paths:
        s = model_path.split("_")
        s = s[-1]
        s.split(".")
        num = int(s[0])
        model_nums.append(num)
    w = np.exp(model_nums) / np.sum(np.exp(model_nums))
    # Make four players using a random model
    players = []
    for i in range(4):
        model = np.random.choice(models,p=w)
        players.append(PentobiNNPlayer(i+1, proc, model,move_selection_strategy="weighted", move_selection_kwargs={"top_p": 1.0}))
    if shuffle_players:
        players = shuffle_players_func(players)
    return players

def player_maker_benchmark(proc, model_paths):
    assert len(model_paths) == 1, f"Only one model path is allowed when benchmarking"
    model_path = model_paths[0]
    model = TFLiteModel(model_path)
    player_to_test = PentobiNNPlayer(1,proc,model,move_selection_strategy="best")
    
    opponents = []
    for pid in range(2,5):
        opponents.append(PentobiInternalEpsilonGreedyPlayer(pid,proc,epsilon=0.03))
    
    players = [player_to_test] + opponents
    players = shuffle_players_func(players)
    return players

def player_maker_with_randomly_internal_players(proc,
                                                model_paths = [],
                                                model_weights = [],
                                                internal_player_epsilon=0.1,
                                                internal_player_chance=0.25,
                                                ):
    if not model_paths:
        # Make four players using the internal player
        players = [PentobiInternalEpsilonGreedyPlayer(i+1, proc,epsilon=internal_player_epsilon) for i in range(4)]
        return players
    # Load all the models
    models = [TFLiteModel(path) for path in model_paths]
    if not model_weights:
        model_weights = model_weights_from_iteration_number(model_paths)
        # In this case we cant add an internal player, since the weights are fuged up
    else:
        # If the model weights come from a file we can't add an internal player
        # we'll add a random internal player "model", that has a of 0.25 (win rate against itself)
        models.append("internal")
        model_weights = np.append(model_weights, 0.25)
    
    # Normalize the model weights
    model_weights = np.exp(model_weights) / np.sum(np.exp(model_weights))
    # Make four players using a random model
    players = []
    for i in range(4):
        model = np.random.choice(models,p=model_weights)
        if model == "internal":
            player = PentobiInternalEpsilonGreedyPlayer(i+1, proc,epsilon=internal_player_epsilon)
        else:
            player = PentobiNNPlayer(i+1, proc, model,move_selection_strategy="epsilon_greedy", move_selection_kwargs={"epsilon": internal_player_epsilon})
        players.append(player)
        
    players = shuffle_players_func(players)
    return players

def model_weights_from_iteration_number(model_paths):
    model_nums = []
    for model_path in model_paths:
        s = model_path.split("_")
        s = s[-1]
        s.split(".")
        num = int(s[0])
        model_nums.append(num)
    return model_nums

def player_maker_test_dataset(proc):
    """ Internal players with a random epsilon between 0.01 and 0.1
    """
    players = [PentobiInternalEpsilonGreedyPlayer(i+1, proc,epsilon=np.random.uniform(0.01,0.1)) for i in range(4)]
    return players


def play_pentobi_wrapper(args):
    return play_pentobi(*args)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # Load the environment variables from env.json
    if os.path.exists('env.json'):
        with open('env.json') as f:
            env_vars = json.load(f)
            #print(env_vars)
    else:
        env_vars = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=1000, help="Number of games")
    parser.add_argument("--num_cpus", type=int, default=10, help="Number of CPUs")
    parser.add_argument("--player_maker", type=str, default="selfplay", help="How to make players. Either 'Benchmark' or 'selfplay'")
    parser.add_argument("--pentobi_gtp", type=str, default=env_vars.get('pentobi_gtp', None), help="Path to pentobi-gtp")
    parser.add_argument("--data_folder", type=str, default=env_vars.get('data_folder', "./Data"), help="Path to data folder")
    parser.add_argument("--model_folder", type=str, default=env_vars.get('model_folder', "./Models"), help="Path to model folder")
    parser.add_argument("--level", required=False, default=1,type=int)
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
    models = [TFLiteModel(path) for path in model_paths]
    
    player_maker_map = {
        "selfplay" : player_maker_selfplay,
        "benchmark" : player_maker_benchmark,
        "use_internal" : player_maker_with_randomly_internal_players
    }
    
    assert args.player_maker in player_maker_map.keys(), f"The player_maker argument must be in {player_maker_map.keys()}"
    
    if args.player_maker == "benchmark":
        assert model_folder.endswith(".tflite"), "If the 'benchmark' player_maker is used, the model_folder must be a single file, not a folder."
        args.player_maker = [args.player_maker]
    
    def _player_maker(proc):
        
        if args.player_maker == "use_internal" and os.path.exists(os.path.join(model_folder, "win_rates.json")):
            # We need to read the win rates from the model folder, in win_rates.json
            with open(os.path.join(model_folder, "win_rates.json")) as f:
                # Win rates are stored as a list of floats
                win_rates = json.load(f)
            assert len(win_rates) == len(model_paths), "The number of models and the number of win rates must be the same"
            return player_maker_map[args.player_maker](proc, model_paths=model_paths, model_weights=win_rates)
        
        return player_maker_map[args.player_maker](proc, model_paths=model_paths)
    
    def arg_generator(num_games):
        kwargs = {
            "command": pentobi_gtp,
            "level": args.level,
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