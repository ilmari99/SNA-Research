import random
import multiprocessing as mp
import os
import argparse
from typing import Tuple, List
import warnings
import tqdm

from .Game import Game
from .Result import Result
from .Player import Player

def run_game(args):
    i, game_func, players_func = args
    game = game_func(i)
    players = players_func(i)
    random.shuffle(players)
    return game.play_game(players)
 
def _simulate_games_once(game_func,
                   players_func,
                   num_games: int,
                   num_cpus: int = -1,
                   folder: str = '.') -> List[Result]:
    os.makedirs(folder, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(folder)
    if num_cpus == -1:
        num_cpus = mp.cpu_count()
    

    with mp.Pool(num_cpus) as pool:
        res_gen = pool.imap_unordered(run_game, [(i, game_func, players_func) for i in range(num_games)])
        #tqdm_bar = tqdm.tqdm(total=num_games)
        results = []
        while True:
            try:
                res = next(res_gen)
                results.append(res)
                #tqdm_bar.update(1)
            except StopIteration:
                break
            except Exception as e:
                print(e)
    os.chdir(cwd)
    return results

def simulate_games(game_constructor,
                   players_constructor,
                   folder: str,
                   num_games: int,
                   num_files: int = -1,
                   num_cpus: int = -1,
                   exists_ok: bool = True
                   ) -> None:
    """Simulate games using the given game and players constructors.
    In total, this function will simulate num_games games.
    Additionally, you can specify the number of files to save the results to.
    In this case we will save the results to num_files files, each containing num_games/num_files games.
    If num_files is -1, we will the results will be saved to num_games files.

    Args:
        game_constructor (Callable[[int], Game]): A function that returns a game instance given an index.
        players_constructor (Callable[[int], List[Player]]): A function that returns a list of players given an index.
        folder (str): The folder to save the results to.
        num_games (int): The number of games to simulate.
        num_files (int, optional): The number of files to save the results to. Defaults to -1, in which case the results will be saved to num_games files.
        num_cpus (int, optional): The number of cpus to use. Defaults to -1, in which case all cpus will be used.
    """
    if os.path.exists(folder) and not exists_ok:
        raise FileExistsError(f"Folder {folder} already exists.")
    # If num_files is not specified, we run the simulation once.
    # Otherwise, we run num_files games at a time.
    if num_files == -1:
        _simulate_games_once(game_constructor, players_constructor, num_games, num_cpus, folder)
        return
    num_cpus = mp.cpu_count() if num_cpus == -1 else num_cpus
    
    if num_files < num_cpus:
        warnings.warn(f"Number of games per round (num_files={num_files}) is less than the desired number of cpus ({num_cpus}). In this case, all cpus will not be used.")
    print(f"Simulating {num_games} games using {min(num_files, num_cpus)} cpus.")
    print(f"The games will be simulated in {num_games//num_files} rounds.")
    for i in tqdm.tqdm(range(num_games//num_files)):
        _simulate_games_once(game_constructor, players_constructor, num_files, num_cpus, folder)
    return
    