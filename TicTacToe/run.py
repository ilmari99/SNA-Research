from TTTGame import TTTGame
from TTTPlayer import TTTPlayer
from TTTAction import TTTAction
from TTTGameState import TTTGameState
from TTTPlayerMaxElems import TTTPlayerMaxElems


game : TTTGame = TTTGame(board_size=(10,10),logger_args = {"log_level" : 10, "log_file" : "TTTGame.log"}, render_mode = "human")
player1 = TTTPlayer("Player 0", logger_args={"log_level" : 10, "log_file" : "TTTPlayer0.log"})
player2 = TTTPlayerMaxElems("Player 1", logger_args={"log_level" : 10, "log_file" : "TTTPlayer1.log"})
player3 = TTTPlayerMaxElems("Player 2", logger_args={"log_level" : 10, "log_file" : "TTTPlayer2.log"})

res = game.play_game([player1, player2, player3])

res_json = res.as_json(states_as_num = True)
for k,v in res_json.items():
    print(f"{k}: {v}")