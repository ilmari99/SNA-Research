from MoskaGame import MoskaGame
from MoskaPlayer import MoskaPlayer

players = [MoskaPlayer("Player 0",
                       logger_args = {
                           "log_file" : "moskaplayer0.log",
                           "log_level" : 10,
                           }),
           MoskaPlayer("Player 1",
                        logger_args = {
                           "log_file" : "moskaplayer1.log",
                           "log_level" : 10,
                        })
              ]
game = MoskaGame(logger_args={"log_file" : "moskagame.log",
                              "log_level" : 10,
                              })
game.play_game(players)