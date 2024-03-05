import numpy as np
from RLFramework.Game import Game
from RLFramework.GameState import GameState
from RLFramework.Player import Player
from TTTAction import TTTAction

class TTTPlayer(Player):
    """ A class representing a player of the game TicTacToe.
    """
    def __init__(self, name, logger_args = None):
        super().__init__(name, logger_args)
        
    def evaluate_states(self, states):
        """ Evaluate the states.
        """
        return [np.random.random() for _ in states]
    
    def initialize_player(self, game: Game) -> None:
        """ Called when a game begins.
        """
        return
    
    def check_is_finished(self, game_state: GameState) -> bool:
        """ Check if the player has a full row/col/diag
        """
        arr = np.array(game_state.board)
        # Check rows and columns
        for row_idx in range(arr.shape[0]):
            if np.all(arr[row_idx] == self.pid):
                self.logger.info(f"{self.name} has won with a row")
                return True
        for col_idx in range(arr.shape[1]):
            if np.all(arr[:, col_idx] == self.pid):
                self.logger.info(f"{self.name} has won with a column")
                return True
        if np.all(np.diag(arr) == self.pid) or np.all(np.diag(np.fliplr(arr)) == self.pid):
            self.logger.info(f"{self.name} has won with a diagonal")
            return True
        if len(game_state.unfinished_players) == 1:
            self.logger.info(f"{self.name} has lost")
            return True
        return False
    
    def select_action_strategy(self, evaluations):
        """ Select the action with the highest evaluation.
        """
        return self._select_best_action(evaluations)
    
    def __repr__(self):
        return f"TTTPlayer({self.name}, {self.pid})"
        