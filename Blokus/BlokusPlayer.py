from typing import List, TYPE_CHECKING
import numpy as np
from RLFramework.Game import Game
from RLFramework.GameState import GameState
from RLFramework.Player import Player
from BlokusAction import BlokusAction
if TYPE_CHECKING:
    from BlokusGame import BlokusGame

class BlokusPlayer(Player):
    """ A class representing a player of the game TicTacToe.
    """
    def __init__(self, name, logger_args = None):
        super().__init__(name, logger_args)
        self.score = 0
        self.is_finished = False

    @property
    def remaining_pieces(self):
        return self.game.player_remaining_pieces[self.pid]
    
    def initialize_player(self, game: Game) -> None:
        """ Called when a game begins.
        """
        self.game : BlokusGame = game
        self.is_finished = False
        return
    
    def evaluate_states(self, states) -> List[float]:
        """ Evaluate the states.
        """
        return [np.random.random() for _ in states]
    
    def select_action_strategy(self, evaluations):
        """ Select the action with the highest evaluation.
        """
        return self._select_weighted_action(evaluations, temperature=0.0)
    
    def __repr__(self):
        return f"BlokusPlayer(name={self.name}, pid={self.pid}, score={self.score}, remaining_pieces={self.remaining_pieces})"
        