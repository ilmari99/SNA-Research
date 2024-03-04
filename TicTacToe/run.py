from TTTGame import TTTGame
from TTTPlayer import TTTPlayer
from TTTAction import TTTAction
from TTTGameState import TTTGameState

game = TTTGame()
player1 = TTTPlayer("Player 0")
player2 = TTTPlayer("Player 1")

game.play_game([player1, player2])