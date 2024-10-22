import json
import multiprocessing
import os
import random
import argparse
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from PentobiGTP import PentobiGTP
from PentobiPlayers import PentobiInternalPlayer, PentobiNNPlayer
from utils import TFLiteModel
import argparse

def play_pentobi(i, seed, player_maker, timeout, save_data_file = "", proc_args = {}):
    
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
    start_t = time.time()
    elapsed_t = 0
    while not proc.is_game_finished() and elapsed_t < timeout:
        pid = proc.pid
        #print(f"Player {pid} playing")
        player = players[pid-1]
        player.play_move()
        num_moves += 1
        elapsed_t = time.time() - start_t
    
    if elapsed_t >= timeout:
        print(f"Game {i} timed out",flush=True)
        proc.close()
        return {}
        
    if save_data_file:
        proc.write_states_to_file(save_data_file)
    score = list(proc.score)
    pl_names = [pl.name for pl in players]
    proc.close()
    return {pl : sc for pl,sc in zip(pl_names, score)}

def shuffle_players_func(players):
    random.shuffle(players)
    for i, player in enumerate(players):
        player.pid = i+1
    return players


def player_maker_benchmark(proc, model_path, num_internal = 3):
   
    opponents = []
    for pid in range(1,num_internal+1):
        opponents.append(PentobiInternalPlayer(pid,
                                               proc,
                                               move_selection_strategy="epsilon_greedy",
                                               move_selection_kwargs={"epsilon" : 0.01},
                                               name="PentobiInternalPlayer" + str(pid))
        )
        
    model = TFLiteModel(model_path)
    nn_players = []
    for pid in range(len(opponents) + 1, 5):
        player_to_test = PentobiNNPlayer(pid,
                                         proc,
                                         model,
                                         move_selection_strategy="epsilon_greedy",
                                         move_selection_kwargs={"epsilon" : 0.01},
                                         name="PlayerToTest"
                                         )
        nn_players.append(player_to_test)
    
    players = nn_players + opponents
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
            #print(env_vars)
    else:
        env_vars = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=1000, help="Number of games")
    parser.add_argument("--num_cpus", type=int, default=10, help="Number of CPUs")
    parser.add_argument("--pentobi_level", type=int, default=1, help="Level of opponent Pentobi players")
    parser.add_argument("--pentobi_gtp", type=str, default=env_vars.get('pentobi_gtp', None), help="Path to pentobi-gtp")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dont_update_win_rate", default=False, action="store_true")
    parser.add_argument("--num_internal", type=int,required=False,default=3)
    parser.add_argument("--game_timeout", type=int, default=60)
    args = parser.parse_args()
    
    print(args)
    num_games = args.num_games
    num_cpus = args.num_cpus
    os.environ["PENTOBI_GTP"] = os.path.abspath(args.pentobi_gtp)
    model_path = os.path.abspath(args.model_path)
    pentobi_gtp = os.path.abspath(args.pentobi_gtp)
    assert args.num_internal < 4 and args.num_internal > 0, "Number of internal players error."

    #model = TFLiteModel(model_path)
    
    def _player_maker(proc):
        return player_maker_benchmark(proc, model_path,args.num_internal)
    
    def arg_generator(num_games):
        kwargs = {
            "command": pentobi_gtp,
            "level": args.pentobi_level,
            "threads": 1,
            "showboard": False,
            "nobook": False,
            "quiet": True,
        }
        for i in range(num_games):
            seed = np.random.randint(2**32)
            file = f""
            yield (i, seed, _player_maker, args.game_timeout, file, kwargs)
    
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

    player_wins = {}
    player_avg_score = {}
    num_games = 0
    games_per_player = {}
    for res in results:
        if not res:
            continue
        scores = list(res.values())
        players = list(res.keys())
        max_sc = max(scores)
        idx = scores.index(max_sc)
        winner_name = players[idx]
        if len([sc for sc in scores if sc == max_sc]) > 1:
            winner_name = "Draw"
        
        for sc, pl in zip(scores,players):
            games_per_player[pl] = games_per_player.get(pl,0) + 1
            player_avg_score[pl] = player_avg_score.get(pl,0) + sc
            if pl == winner_name:
                # Increment the win count
                player_wins[pl] = player_wins.get(pl,0) + 1
        num_games += 1
    player_avg_score = {k : v/games_per_player[k] for k,v in player_avg_score.items()}
    player_wins = {k : v/games_per_player[k] for k,v in player_wins.items()}
    print(f"Wins",player_wins)
    print(f"Average score", player_avg_score)
    
    print(f"Model {model_path} win percent: {player_wins.get('PlayerToTest',0)}")
    
    if not args.dont_update_win_rate:
        # Write the loss percent to a win_rates.json file at the correct index
        model_number = int(model_path.split("/")[-1].split(".")[0].split("_")[-1])
        model_folder = "/".join(model_path.split("/")[:-1])
        win_rate_file = os.path.join(model_folder, "win_rates.json")
        if args.pentobi_level != 1:
            print(f"Not writing win rates, since the level is not 1")
            exit()
        if not os.path.exists(win_rate_file):
            win_rates = {}
            #with open(win_rate_file, "w") as f:
            #    json.dump({model_path : class_wins.get('PentobiNNPlayer',0)}, f)
        else:
            with open(win_rate_file) as f:
                win_rates = json.load(f)
                # If win_rates is a list, convert it to a dictionary
                if isinstance(win_rates, list):
                    win_rates_dict = {}
                    for i,wr in enumerate(win_rates):
                        i_model_path = os.path.join(model_folder, f"model_{i}.tflite")
                        win_rates_dict[i_model_path] = wr
                    print(f"Converted win rates {win_rates} to {win_rates_dict}")
                    win_rates = win_rates_dict
        win_rates[model_path] = player_wins.get('PlayerToTest',0)
        with open(win_rate_file, "w") as f:
            json.dump(win_rates, f)
    
                
            
        
        