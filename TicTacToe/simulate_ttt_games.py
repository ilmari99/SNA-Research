import random
from TTTGame import TTTGame
from TTTPlayer import TTTPlayer
from TTTPlayerNeuralNet import TTTPlayerNeuralNet
from TTTResult import TTTResult
import multiprocessing as mp
import os
import argparse
from RLFramework import simulate_games


def game_constructor(i):
    return TTTGame(board_size=(3,3),
                            logger_args = None,
                            render_mode = "",
                            gather_data = f"gathered_data_{i}.csv",
                            custom_result_class = TTTResult,
                            )

def players_constructor(i):
    players = [TTTPlayerNeuralNet(model_path="/home/ilmari/python/RLFramework/models/model_5.tflite",
                                name=f"NeuralNetPlayer1_{i}",
                                move_selection_temp=0,
                                logger_args=None),
               TTTPlayer(name=f"Player2_{i}", logger_args=None)]
    random.shuffle(players)
    return players

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description="Simulate TicTacToe games")
    argparser.add_argument("--num_proc", help="Number of processes to use", default=12)
    argparser.add_argument("--num_games", help="Number of games to play", default=1000)
    argparser.add_argument("--folder", help="Folder to save the data to", default="TTTDataset1")
    argparser.add_argument("--num_files", help="Number of files to save the data to", default=-1)
    argparser = argparser.parse_args()
    
    num_proc = int(argparser.num_proc)
    num_games = int(argparser.num_games)
    folder = argparser.folder
    num_files = int(argparser.num_files)
    
    res = simulate_games(game_constructor, players_constructor, folder, num_games, num_files=num_files, num_cpus=num_proc,return_results=True)
    
    # Count how many times each player won
    player_name_to_wins = {"NeuralNetPlayer1" : 0, "Player2" : 0}
    for r in res:
        if r.winner:
            print(r)
            player_name_to_wins[r.winner.split("_")[0]] += 1
    print(player_name_to_wins)
    
            
            
    