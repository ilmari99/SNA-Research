from typing import List, SupportsFloat

import numpy as np
from RLFramework.Game import Game
from RLFramework.GameState import GameState

class TTTGameState(GameState):
    """ A class representing the state of the game TicTacToe.
    """
    def __init__(self, state_json):
        super().__init__(state_json)
        self.board = state_json["board"]
        
    
    def game_to_state_json(cls, game, player):
        """ Convert a Game to a state_json.
        """
        state_json = {
            "board" : game.board,
        }
        return state_json
    
    def to_vector(self, perspective_pid = None) -> List[SupportsFloat]:
        """ Convert the state to a vector.
        """
        if perspective_pid is None:
            perspective_pid = self.perspective_pid
        board_arr = np.array(self.board)
        return [perspective_pid] + [self.current_pid] + board_arr.flatten().tolist()
    
    #def __repr__(self):
    #    return f"TTTGameState(\n{np.array(self.board)}\n)"
    