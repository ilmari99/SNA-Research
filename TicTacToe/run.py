from TTTGame import TTTGame
from TTTPlayer import TTTPlayer
from TTTAction import TTTAction
from TTTGameState import TTTGameState


game : TTTGame = TTTGame(board_size=(5,5),logger_args = {"log_level" : 10, "log_file" : "TTTGame.log"})
player1 = TTTPlayer("Player 0")
player2 = TTTPlayer("Player 1")

res = game.play_game([player1, player2])

res_json = res.as_json(states_as_num = True)
for k,v in res_json.items():
    print(f"{k}: {v}")