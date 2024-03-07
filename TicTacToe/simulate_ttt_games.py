import random
from TTTGame import TTTGame
from TTTPlayer import TTTPlayer
from TTTPlayerNeuralNet import TTTPlayerNeuralNet
from TTTResult import TTTResult
import multiprocessing as mp
import os
import argparse

def get_game_and_players(i):
    game : TTTGame = TTTGame(board_size=(5,5),
                         logger_args = None,
                         render_mode = "",
                         gather_data = f"gathered_data_{i}.csv",
                         custom_result_class = TTTResult,
                         )
    
    player1 = TTTPlayer(f"Player0_{i}", logger_args=None)
    player2 = TTTPlayer(f"Player1_{i}", logger_args=None)
    return game, [player1, player2]

def play_game(i):
    game, players = get_game_and_players(i)
    random.shuffle(players)
    res = game.play_game(players)
    return res

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description="Simulate TicTacToe games")
    argparser.add_argument("--num_proc", help="Number of processes to use", default=12)
    argparser.add_argument("--num_games", help="Number of games to play", default=1000)
    argparser.add_argument("--folder", help="Folder to save the data to", default="TTTDataset1")
    argparser = argparser.parse_args()
    
    num_proc = int(argparser.num_proc)
    num_games = int(argparser.num_games)
    folder = argparser.folder
    
    
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    # Play num_games, num_proc at a time, unordered
    with mp.Pool(num_proc) as pool:
        res_gen = pool.imap_unordered(play_game, range(num_games))
        while True:
            try:
                res = next(res_gen)
            except StopIteration:
                break
            except Exception as e:
                # with stacktrace
                print(e)
                
    