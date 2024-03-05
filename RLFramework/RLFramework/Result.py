from typing import Dict, Any, TYPE_CHECKING, List

import numpy as np
if TYPE_CHECKING:
    from .GameState import GameState
    from .Action import Action
    from .Player import Player

class Result:
    """ This class is used to describe
    what was simulated, and what we're the results.
    """
    def __init__(self,
                 successful : bool = None,
                 player_jsons : List[Dict[str, Any]] = None,
                 finishing_order : List[int] = None,
                 logger_args : Dict[str, Any] = None,
                 game_state_class : 'GameState' = None,
                 game_states : List['GameState'] = None,
                 previous_turns : List[int] = None,
        ):
        self.successful = successful
        self.player_jsons = player_jsons
        self.finishing_order = finishing_order
        self.logger_args = logger_args
        self.game_state_class = game_state_class
        self.game_states = game_states
        self.previous_turns = previous_turns

    def save_to_file(self, file_path : str) -> None:
        """ Take all the game states as vectors (X), and label them with the final score of the player.
        """
        assert file_path.endswith(".csv"), f"file_path must end with .csv, not {file_path}"
        player_final_scores = self.game_states[-1].player_scores
        Xs = []
        ys = []
        for game_state in self.game_states:
            pid = game_state.current_player
            Xs.append(game_state.to_vector())
            ys.append(player_final_scores[pid])
        Xs = np.array(Xs, dtype=np.int32)
        ys = np.array(ys, dtype=np.int32)
        # Save to csv
        np.savetxt(file_path, np.hstack((Xs, ys.reshape(-1, 1))), delimiter = ",", fmt = "%d")
        




    def as_json(self, states_as_num = False) -> Dict[str, Any]:
        """ Return the result as a json.
        """
        return {"successful" : self.successful,
                "player_jsons" : self.player_jsons,
                "finishing_order" : self.finishing_order,
                "logger_args" : self.logger_args,
                "game_state_class" : self.game_state_class.__name__,
                "game_states" : len(self.game_states) if states_as_num else self.game_states,
                #"previous_turns" : self.previous_turns,
                }
