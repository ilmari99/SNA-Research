from RLFramework import Result
import numpy as np


class MoskaResult(Result):

    def save_array_to_file(self, arr: np.ndarray, file_path: str) -> None:
        with open(file_path, "a") as f:
            # Save all values as int, except the last one, which is a float.
            fmt = ["%d" for _ in range(arr.shape[1] - 1)] + ["%f"]
            np.savetxt(f, arr, delimiter=",", fmt=fmt)
            
    def save_game_states_to_file(self, file_path : str) -> None:
        """ Take all the game states as vectors (X), and label them with the final score of the player.
        """
        assert file_path.endswith(".csv"), f"file_path must end with .csv, not {file_path}"
        player_final_scores = self.game_states[-1].player_scores
        #print(f"Final scores: {player_final_scores}")
        #print(f"Number of game states: {len(self.game_states)}")
        Xs = []
        ys = []
        total_game_states = len(self.game_states)
        for curr_game_state_index, game_state in enumerate(self.game_states):
            # We want to save the state only in cases where a player (perspective pid) has just made a move
            # which means the next player is still unknown
            # Additionally, we save the state if the target is playing from the deck,
            # in which case we still know the next player
            if game_state.current_pid == -1 or game_state.target_is_kopling:
                perspective_pid = game_state.perspective_pid
                Xs.append(game_state.to_vector(perspective_pid))
                ys.append(player_final_scores[perspective_pid] * (1 - self.discount_factor(game_state, curr_game_state_index + 1, total_game_states)))
        Xs = np.array(Xs, dtype=np.float16)
        ys = np.array(ys, dtype=np.float16)
        arr = np.hstack((Xs, ys.reshape(-1, 1)))
        self.save_array_to_file(arr, file_path)
            
    def discount_factor(self, game_state, curr_game_state_num: int, total_game_states: int) -> float:
        """ A number between 0 and 1, representing the factor with which
        the final score of the game state should be multiplied.
        """
        return 0