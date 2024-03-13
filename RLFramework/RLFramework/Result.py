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
                 winner : str = None,
        ):
        self.successful = successful
        self.player_jsons = player_jsons
        self.finishing_order = finishing_order
        self.logger_args = logger_args
        self.game_state_class = game_state_class
        self.game_states = game_states
        self.previous_turns = previous_turns
        self.winner = winner

    def save_game_states_to_file(self, file_path : str) -> None:
        """ Take all the game states as vectors (X), and label them with the final score of the player.
        """
        assert file_path.endswith(".csv"), f"file_path must end with .csv, not {file_path}"
        player_final_scores = self.game_states[-1].player_scores
        #print(f"Final scores: {player_final_scores}")
        #print(f"Number of game states: {len(self.game_states)}")
        Xs = []
        ys = []
        for game_state in self.game_states:
            x = game_state.to_vector()
            # Save the state from each player's perspective.
            for perspective_pid in range(len(player_final_scores)):
                Xs.append([perspective_pid] + x)
                ys.append(player_final_scores[perspective_pid])
        Xs = np.array(Xs, dtype=np.float16)
        ys = np.array(ys, dtype=np.float16)
        arr = np.hstack((Xs, ys.reshape(-1, 1)))
        self.save_array_to_file(arr, file_path)
        #print(f"Saved {len(Xs)} states with {Xs.shape[1]} features to {file_path}")
        
    def save_array_to_file(self, arr : np.ndarray, file_path : str) -> None:
        """ Write the array to a file.
        NOTE: Overwrite this if you want to customize how the array is saved.
        """
        arr = arr.astype(np.float16)
        with open(file_path, "a") as f:
            fmt = "%f"
            np.savetxt(f, arr, delimiter=",", fmt=fmt)

    def as_json(self, states_as_num = False) -> Dict[str, Any]:
        """ Return the result as a json.
        """
        return {"successful" : self.successful,
                "player_jsons" : self.player_jsons,
                "finishing_order" : self.finishing_order,
                "logger_args" : self.logger_args,
                "game_state_class" : self.game_state_class.__name__,
                "game_states" : len(self.game_states) if states_as_num else self.game_states,
                "winner" : self.winner,
                #"previous_turns" : self.previous_turns,
                }
        
    def __repr__(self):
        return f"Result(successful={self.successful}, player_scores={[p['score'] for p in self.player_jsons]}, winner={self.winner})"
