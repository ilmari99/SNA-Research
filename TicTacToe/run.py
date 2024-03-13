from TTTGame import TTTGame
from TTTPlayer import TTTPlayer
from TTTPlayerNeuralNet import TTTPlayerNeuralNet
from TTTAction import TTTAction
from TTTGameState import TTTGameState
from TTTPlayerMaxElems import TTTPlayerMaxElems
from TTTResult import TTTResult
from TTTHumanPlayer import TTTHumanPlayer


game : TTTGame = TTTGame(board_size=(3,3),
                         logger_args = {"log_level" : 10, "log_file" : "TTTGame.log"},
                         render_mode = "human",
                         gather_data = "",
                         custom_result_class = TTTResult,
)

player1 = TTTPlayer("RandomPlayer1", logger_args={"log_level" : 10, "log_file" : "TTTPlayer0.log"})
player2 = TTTPlayerNeuralNet(model_path="/home/ilmari/python/RLFramework/models/model_9.tflite",
                                name=f"NeuralNetPlayer1",
                                move_selection_temp=0,
                                logger_args=None)
human = TTTHumanPlayer("HumanPlayer", logger_args={"log_level" : 10, "log_file" : "TTTPlayer1.log"})

res1 = game.play_game([human, player2])

res_json = res1.as_json(states_as_num = True)
for k,v in res_json.items():
    print(f"{k}: {v}")
