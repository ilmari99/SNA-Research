
import logging

from PFGame import PFGame
from PFPlayer import PFPlayer
from PFAction import PFAction
from PFResult import PFResult
from PFOptimalPlayer import PFOptimalPlayer
from PFNeuralNetworkPlayer import PFNeuralNetworkPlayer


game : PFGame = PFGame(board_size=(7,7),
                       logger_args = {"log_level" : 10, "log_file" : "PFGame.log"},
                       render_mode = "human",
                       gather_data = "states.csv",
                       custom_result_class = PFResult,
)

player1 = PFPlayer("RandomPlayer1", logger_args={"log_level" : 10, "log_file" : "PFPlayer0.log"})
player2 = PFOptimalPlayer("OptimalPlayer", logger_args={"log_level" : 10, "log_file" : "PFPlayer1.log"})
player3 = PFNeuralNetworkPlayer("NeuralNetPlayer", model_path = "/home/ilmari/python/RLFramework/models/model_3.tflite", logger_args={"log_level" : 10, "log_file" : "PFPlayer2.log"})

res1 = game.play_game([player3])

res_json = res1.as_json(states_as_num = True)
for k,v in res_json.items():
    print(f"{k}: {v}")
