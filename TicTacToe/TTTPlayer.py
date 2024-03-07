from typing import List
import numpy as np
from RLFramework.Game import Game
from RLFramework.GameState import GameState
from RLFramework.Player import Player
from TTTAction import TTTAction

class TTTPlayer(Player):
    """ A class representing a player of the game TicTacToe.
    """
    def __init__(self, name, logger_args = None):
        super().__init__(name, logger_args)
        
    def evaluate_states(self, states) -> List[float]:
        """ Evaluate the states.
        """
        return [np.random.random() for _ in states]
    
    def initialize_player(self, game: Game) -> None:
        """ Called when a game begins.
        """
        self.game : Game = game
        return
    
    def select_action_strategy(self, evaluations):
        """ Select the action with the highest evaluation.
        """
        return self._select_best_action(evaluations)
    
    def __repr__(self):
        return f"TTTPlayer({self.name}, {self.pid})"
        