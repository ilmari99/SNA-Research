import multiprocessing
import os
import random
from BlokusGame import BlokusGame
from BlokusPlayer import BlokusPlayer
from BlokusNNPlayer import BlokusNNPlayer
from BlokusGreedyPlayer import BlokusGreedyPlayer
from BlokusHumanPlayer import BlokusHumanPlayer

players = [BlokusNNPlayer(name="Player 0",
                           model_path="/home/ilmari/python/RLFramework/BlokusModels/model.tflite",
                           action_selection_strategy = "greedy",
                        logger_args = {
                           "log_file" : "blokusplayer0.log",
                           "log_level" : 10,
                  }),
           BlokusNNPlayer(name="Player 1",
                           model_path="/home/ilmari/python/RLFramework/BlokusModels/model.tflite",
                           action_selection_strategy = "greedy",
                           logger_args = {
                              "log_file" : "blokusplayer1.log",
                              "log_level" : 10,
                     }),
           BlokusNNPlayer(name="Player 2",
                           model_path="/home/ilmari/python/RLFramework/BlokusModels/model.tflite",
                           action_selection_strategy = "greedy",
                           logger_args = {
                              "log_file" : "blokusplayer2.log",
                              "log_level" : 10,
                     }),
           BlokusHumanPlayer(name="Player 3",
               #model_path="/home/ilmari/python/RLFramework/model.tflite",
               #action_selection_strategy = "weighted",
               #action_selection_args = ((), {"temperature" : 0.0}),
            logger_args = {
               "log_file" : "blokusplayer3.log",
               "log_level" : 10,
      }),
]

game = BlokusGame(board_size=(20,20), logger_args={"log_file" : "blokusgame.log",
                                                      "log_level" : 10,
                              },
                     model_paths=["/home/ilmari/python/RLFramework/BlokusModels/model.tflite"],#,"/home/ilmari/python/RLFramework/BlokusModelsMahti/model_15.tflite"],
                  render_mode = "human",
               gather_data = "blokus_data.csv",
               timeout=10000,
)
#random.shuffle(players)
game.play_game(players)
#with multiprocessing.Pool(1) as p:
#   p.map(game.play_game, [players])