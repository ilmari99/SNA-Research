from typing import TYPE_CHECKING

from RLFramework.Action import Action
from PFGameState import PFGameState

if TYPE_CHECKING:
    from .PFPlayer import PFPlayer
    from .PFGame import PFGame

class PFAction(Action):
    """ A class representing an action in the PF game,
    where the player tries to find the shortest path to the goal.
    """
    def __init__(self, x : int, y : int):
        self.x = x
        self.y = y
        
    def modify_game(self, game : 'PFGame') -> 'PFGameState':
        """ Modify the game instance according to the action.
        """
        # Set the current pos to 0
        game.board[game.player.position[0]][game.player.position[1]] = 0
        game.board[self.x][self.y] = 1
        game.player.position = [self.x, self.y]
        return PFGameState.from_game(game)
    
    def check_action_is_legal(self, game : 'PFGame') -> bool:
        """ Check if the action is legal in the given game state.
        The action is legal, if the cell is next to the player's current position.
        """
        return abs(game.player.position[0] - self.x) + abs(game.player.position[1] - self.y) == 1
    
    def __repr__(self):
        return f"PFAction({self.x}, {self.y})"