from TTTPlayer import TTTPlayer
import numpy as np


class TTTPlayerMaxElems(TTTPlayer):
    """ This TicTacToe players plays the move, that leads to a row/col/diag with the most of its own pieces.
    """
    def evaluate_states(self, states):
        """ Evaluate the states.
        """
        # Find the largest row/col/diag with the most of the players pieces
        evaluations = [self.evaluate_state(state) for state in states]
        return evaluations
    
    def evaluate_state(self, state):
        """ Evaluate the state.
        """
        arr = np.array(state.board)
        # Find how many pieces are on each totally free row/col/diag, and return the max
        row_maxes = []
        for row_idx in range(arr.shape[0]):
            if any(arr[row_idx, i] not in [self.pid, -1] for i in range(arr.shape[1])):
                continue
            row_maxes.append(sum(arr[row_idx] == self.pid))
        col_maxes = []
        for col_idx in range(arr.shape[1]):
            if any(arr[i, col_idx] not in [self.pid, -1] for i in range(arr.shape[0])):
                continue
            col_maxes.append(sum(arr[:, col_idx] == self.pid))
        diag_maxes = []
        for diag in [np.diag(arr), np.diag(np.fliplr(arr))]:
            if any(diag[i] not in [self.pid, -1] for i in range(len(diag))):
                continue
            diag_maxes.append(sum(diag == self.pid))
        all_evals = row_maxes + col_maxes + diag_maxes
        return max(all_evals) if all_evals else 0.0
        