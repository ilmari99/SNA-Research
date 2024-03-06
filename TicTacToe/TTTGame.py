import sys
from RLFramework.GameState import GameState
import numpy as np
from RLFramework.Game import Game
from typing import List, Tuple
from TTTAction import TTTAction
from TTTGameState import TTTGameState
from TTTPlayer import TTTPlayer


class TTTGame(Game):
    """ A class representing the game TicTacToe.
    """
    def __init__(self, board_size : Tuple[int, int] = (3, 3), logger_args : dict = None):
        super().__init__(TTTGameState, logger_args)
        self.board_size = board_size
        #self.board = [[-1 for _ in range(board_size[1])] for _ in range(board_size[0])]
    
    def initialize_game(self, players: List[TTTPlayer]) -> None:
        """ When the game is started, we need to set the board.
        """
        self.board = [[-1 for _ in range(self.board_size[1])] for _ in range(self.board_size[0])]

    def restore_game(self, game_state: TTTGameState) -> None:
        """ Restore the game to the state described by the game_state.
        """
        self.board = game_state.board
    
    def select_turn(self, players: List[TTTPlayer], previous_turns: List[int]) -> int:
        return super()._select_round_turn(players, previous_turns)
    
    def calculate_reward(self, game_state: TTTGameState) -> float:
        """ Calculate the reward for the player.
        If the player wins, the reward is 1.0.
        If the game is a draw, the reward is 0.5
        """
        pid = game_state.current_player
        #print(f"pid: {pid}")
        #print(game_state)
        if pid not in game_state.unfinished_players:
            return 1
        # If there are no -1 in the board, the game is a draw
        if -1 not in np.array(game_state.board).flatten():
            return 0.5
        return 0.0
    
    def check_is_player_finished(self, pid : int, game_state: GameState) -> bool:
        """ Check if a player has a full row/col/diag
        """
        arr = np.array(game_state.board)
        # Check rows and columns
        for row_idx in range(arr.shape[0]):
            if np.all(arr[row_idx] == pid):
                self.logger.info(f"{self.players[pid]} has won with a row")
                return True
        for col_idx in range(arr.shape[1]):
            if np.all(arr[:, col_idx] == pid):
                self.logger.info(f"{self.players[pid]} has won with a column")
                return True
        if np.all(np.diag(arr) == pid) or np.all(np.diag(np.fliplr(arr)) == pid):
            self.logger.info(f"{self.players[pid]} has won with a diagonal")
            return True
        # If we have no free spots, we have finished
        if -1 not in arr:
            self.logger.info(f"No available spots.")
            return True
        # If the other player is finished, we are finished
        other_pid = 1 if pid == 0 else 0
        if other_pid not in game_state.unfinished_players:
            self.logger.info(f"Other player is finished.")
            return True
        
        return False
    
    def get_all_possible_actions(self) -> List[TTTAction]:
        """ Return all possible actions.
        """
        actions = []
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if TTTAction.check_action_is_legal_from_args(self, x, y):
                    actions.append(TTTAction(x, y))
        return actions
    
    def __repr__(self):
        return f"TTTGame(\n{np.array(self.board)}\n)"
    