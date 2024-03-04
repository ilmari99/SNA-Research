from typing import List, SupportsFloat

import numpy as np
from RLFramework.Game import Game
from RLFramework.GameState import GameState

class TTTGameState(GameState):
    """ A class representing the state of the game TicTacToe.
    """
    
    def game_to_state_json(cls, game):
        """ Convert a Game to a state_json.
        """
        state_json = {
            "board" : game.board,
        }
        return state_json
    
    def to_vector(self) -> List[SupportsFloat]:
        """ Convert the state to a vector.
        """
        board_arr = np.array(self.board)
        return [self.current_player] + board_arr.flatten().tolist()
    
    def __repr__(self):
        return f"TTTGameState(\n{np.array(self.board)}\n)"
    