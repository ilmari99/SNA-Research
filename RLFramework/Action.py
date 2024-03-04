from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from RLFramework.Game import Game
    from RLFramework.GameState import GameState
    from RLFramework.Player import Player

class Action(ABC):
    """ Action -class contains information about the proposed changes to the game.
    The action class is passed to the the Game.step() -method, which then returns the new state of the game.
    """
    def __init__(self):
        pass

    @abstractmethod
    def modify_game(self, game: Game) -> GameState:
        """ Modify the game instance according to the action.
        """
        pass

    @abstractmethod
    def check_action_is_legal(self, game: Game) -> bool:
        """ Check if the action is legal in the given game state.
        """
        pass