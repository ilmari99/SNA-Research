from typing import List
from RLFramework import Result
import numpy as np


class BlockusResult(Result):
    
    def save_game_states_to_file(self, file_path : str) -> None:
        """ Take all the game states as vectors (X), and label them with the final score of the player.
        """
        assert file_path.endswith(".csv"), f"file_path must end with .csv, not {file_path}"
        player_final_scores = self.game_states[-1].player_scores
        #print(f"Final scores: {player_final_scores}")
        #print(f"Number of game states: {len(self.game_states)}")
        Xs = []
        ys = []
        player_final_scores = self.modify_final_scores(player_final_scores)
        total_game_states = len(self.game_states)
        for curr_game_state_index, game_state in enumerate(self.game_states):
            # Save the state from each player's perspective.
            #for perspective_pid in range(len(player_final_scores)):
            perspective_pid = game_state.perspective_pid
            Xs.append(game_state.to_vector(perspective_pid))
            ys.append(player_final_scores[perspective_pid] * (1 - self.discount_factor(game_state, curr_game_state_index + 1, total_game_states)))
        Xs = np.array(Xs, dtype=np.float16)
        ys = np.array(ys, dtype=np.float16)
        arr = np.hstack((Xs, ys.reshape(-1, 1)))
        self.save_array_to_file(arr, file_path)
    
    def modify_final_scores(self, final_scores : List[float]) -> List[float]:
        """ Add +50 to the winner, and normalize the scores to [0,1].
        """
        max_score = max(final_scores)
        winners = [i for i, score in enumerate(final_scores) if score == max_score]
        if len(winners) == 1:
            final_scores[winners[0]] += 50
        # The score can be [0,150], so we normalize it to [0,1]
        final_scores = [score / 139 for score in final_scores]
        return final_scores
    
    
    def save_array_to_file(self, arr: np.ndarray, file_path: str) -> None:
        with open(file_path, "a") as f:
            # Save all values as int, except the last one, which is a float.
            fmt = ["%d" for _ in range(arr.shape[1] - 1)] + ["%f"]
            np.savetxt(f, arr, delimiter=",", fmt=fmt)
            
    def discount_factor(self, game_state, curr_game_state_num: int, total_game_states: int) -> float:
        """ A number between 0 and 1, representing the factor with which
        the final score of the game state should be multiplied.
        """
        # Linear discount, starting from 1 and ending at 0.
        return 0#1 - curr_game_state_num / total_game_states