from TTTGame import TTTGame
from TTTPlayer import TTTPlayer
from TTTAction import TTTAction
from TTTGameState import TTTGameState
from TTTPlayerMaxElems import TTTPlayerMaxElems
from TTTResult import TTTResult


game : TTTGame = TTTGame(board_size=(5,5),
                         logger_args = {"log_level" : 10, "log_file" : "TTTGame.log"},
                         render_mode = "human",
                         gather_data = "gathered_data.csv",
                         custom_result_class = TTTResult,
)
player1 = TTTPlayer("Player 0", logger_args={"log_level" : 10, "log_file" : "TTTPlayer0.log"})
player2 = TTTPlayer("Player 1", logger_args={"log_level" : 10, "log_file" : "TTTPlayer1.log"})
player3 = TTTPlayer("Player 2", logger_args={"log_level" : 10, "log_file" : "TTTPlayer2.log"})

res1 = game.play_game([player1, player2, player3])

res_json = res1.as_json(states_as_num = True)
for k,v in res_json.items():
    print(f"{k}: {v}")
    
# Play another game with different players
player1 = TTTPlayer("Player 0", logger_args={"log_level" : 10, "log_file" : "TTTPlayer0.log"})
player2 = TTTPlayer("Player 1", logger_args={"log_level" : 10, "log_file" : "TTTPlayer1.log"})

res2 = game.play_game([player1, player2])
