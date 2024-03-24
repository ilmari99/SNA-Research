from typing import List, SupportsFloat

import numpy as np
from RLFramework.Game import Game
from RLFramework.GameState import GameState

class PFGameState(GameState):
    """ A class representing the state of the game PathFinder.
    """
    
    def game_to_state_json(cls, game, player):
        """ Convert a Game to a state_json.
        """
        state_json = {
            "board" : game.board,
            "goal" : game.goal,
            "num_moves" : game.num_moves,
            "current_player_pos" : player.position,
        }
        return state_json
    
    def to_vector(self, perspective_pid = None) -> List[SupportsFloat]:
        """ The state as a vector, is the flattened board.
        """
        #print(f"Player scores: {self.player_scores}")
        return self.player_scores + np.array(self.board).flatten().tolist()
    
    def __repr__(self):
        return f"PFGameState(\n{np.array(self.board)}\n)"