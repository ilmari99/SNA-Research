from MoskaGame import MoskaGame
from MoskaPlayer import MoskaPlayer
from MoskaHumanPlayer import MoskaHumanPlayer

for i in range(1):
   players = [MoskaHumanPlayer("Player 0",
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
               ]
   game = MoskaGame(logger_args={"log_file" : "moskagame.log",
                                 "log_level" : 10,
                                 },
                     render_mode = "text",
                  gather_data = "moska_data.csv"
   )
   game.play_game(players)