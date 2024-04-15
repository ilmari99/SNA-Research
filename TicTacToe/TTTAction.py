from typing import TYPE_CHECKING

from RLFramework.Action import Action
from TTTGameState import TTTGameState

if TYPE_CHECKING:
    from .TTTPlayer import TTTPlayer
    from .TTTGame import TTTGame

class TTTAction(Action):
    """ A class representing an action in the game TicTacToe.
    """
    def __init__(self, x : int, y : int):
        self.x = x
        self.y = y
        
    def modify_game(self, game : 'TTTGame') -> 'TTTGameState':
        """ Modify the game instance according to the action.
        """
        game.board[self.x][self.y] = game.current_pid
        game.current_pid = -1# - game.current_pid
        return TTTGameState.from_game(game)
    
    def check_action_is_legal(self, game : 'TTTGame') -> bool:
        """ Check if the action is legal in the given game state.
        """
        return game.board[self.x][self.y] == -1
    
    @classmethod
    def check_action_is_legal_from_args(cls, game: 'TTTGame', x : int, y : int) -> bool:
        """ Check if the action is legal in the given game state.
        """
        return game.board[x][y] == -1
    
    def __repr__(self):
        return f"TTTAction({self.x}, {self.y})"
    