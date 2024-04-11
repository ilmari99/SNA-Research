import multiprocessing
from MoskaGame import MoskaGame
from MoskaPlayer import MoskaPlayer
from MoskaHumanPlayer import MoskaHumanPlayer

for i in range(1):
   players = [MoskaPlayer("Player 0",
                        logger_args = {
                              "log_file" : "moskaplayer0.log",
                              "log_level" : 10,
                              }),
            MoskaPlayer("Player 1",
                           logger_args = {
                              "log_file" : "moskaplayer1.log",
                              "log_level" : 10,
                           }),
            MoskaPlayer("Player 2",
                           logger_args = {
                              "log_file" : "moskaplayer2.log",
                              "log_level" : 10,
                           }),
            MoskaPlayer("Player 3",
                           logger_args = {
                              "log_file" : "moskaplayer3.log",
                              "log_level" : 10,
                           }),
               ]
   
   game = MoskaGame(logger_args={"log_file" : "moskagame.log",
                                 "log_level" : 10,
                                 },
                     render_mode = "",
                  gather_data = "moska_data.csv",
                  timeout=2,
   )
   
   with multiprocessing.Pool(1) as p:
      p.map(game.play_game, [players])