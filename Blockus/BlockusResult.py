from RLFramework import Result
import numpy as np


class BlockusResult(Result):

    def save_array_to_file(self, arr: np.ndarray, file_path: str) -> None:
        with open(file_path, "a") as f:
            # Save all values as int, except the last one, which is a float.
            fmt = ["%d" for _ in range(arr.shape[1])]# + ["%f"]
            np.savetxt(f, arr, delimiter=",", fmt=fmt)
            
    def discount_factor(self, game_state, curr_game_state_num: int, total_game_states: int) -> float:
        """ A number between 0 and 1, representing the factor with which
        the final score of the game state should be multiplied.
        """
        return 0