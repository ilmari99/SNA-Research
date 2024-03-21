from typing import List
from RLFramework.Action import Action
import numpy as np
from RLFramework.Game import Game
from RLFramework.GameState import GameState
from RLFramework.Player import Player
from TTTAction import TTTAction
from TTTPlayer import TTTPlayer

class TTTHumanPlayer(TTTPlayer):
    """ A class representing a human player of the game TicTacToe.
    """
    def choose_move(self, game: Game) -> Action:
        """ Choose the move to make.
        """
        print(game)
        print(game.get_current_state().to_vector())
        x, y = input("Enter the x and y coordinates of your move: ").split()
        return TTTAction(int(x), int(y))
        