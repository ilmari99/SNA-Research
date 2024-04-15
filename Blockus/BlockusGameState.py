from typing import List, SupportsFloat, TYPE_CHECKING

import numpy as np
from RLFramework.Game import Game
from RLFramework.GameState import GameState
if TYPE_CHECKING:
    from BlockusGame import BlockusGame

class BlockusGameState(GameState):
    """ A class representing the state of the game TicTacToe.
    """
    def __init__(self, state_json):
        super().__init__(state_json)
        self.board = state_json["board"]
        self.player_remaining_pieces : List[List[int]] = state_json["player_remaining_pieces"]
        self.finished_players : List[int] = state_json["finished_players"]
    
    def game_to_state_json(cls, game : 'BlockusGame', player):
        """ Convert a Game to a state_json.
        """
        state_json = {
            "board" : game.board,
            "player_remaining_pieces" : game.player_remaining_pieces,
            "finished_players" : game.finished_players,
        }
        return state_json
    
    def to_vector(self, perspective_pid = None) -> List[SupportsFloat]:
        """ Convert the state to a vector.
        """
        if perspective_pid is None:
            perspective_pid = self.perspective_pid
        board_arr = np.array(self.board)
        return [perspective_pid] + [self.current_pid] + board_arr.flatten().tolist()