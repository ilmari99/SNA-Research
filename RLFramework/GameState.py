from abc import ABC, abstractmethod
from typing import List, SupportsFloat, TYPE_CHECKING
from copy import deepcopy
if TYPE_CHECKING:
    from RLFramework.Game import Game
    from RLFramework.Action import Action
    from RLFramework.Player import Player


class GameState(ABC):
    """ A class representing a state of a game.
    The game state is a snapshot of the current game. It should contain ALL
    information needed to restore the game to exactly the same state as it was when the snapshot was taken.

    In games with hidden information, this class contains all the information that is available to the player,
    AND the information that is hidden from the player.

    The GameState is used to evaluate the game state, and to restore the game state to a previous state.

    The gamestate must be deepcopiable, and the copy must be independent of the original game state.
    """

    def __init__(self, state_json):
        """ Initialize the game state.
        If copy is True, the values of the GameState will be deepcopies (if possible) of the values of the Game instance.
        """
        self.unfinished_players : List[int] = []
        self.players : List[int] = []

    def check_is_deepcopyable(self) -> None:
        """ Check that the subclass is deepcopyable.
        """
        try:
            deepcopy(self)
        except Exception as e:
            raise Exception(f"GameState must be deepcopyable. Error: {e}")
        
    def deepcopy(self):
        """ Return a deepcopy of the game state.
        """
        return deepcopy(self)

    
    @classmethod
    @abstractmethod
    def from_game(cls, game : Game, copy : bool = False):
        """ Create a GameState from a Game instance.
        If copy is True, the values of the GameState will be deepcopies (if possible) of the values of the Game instance.
        """
        pass

    @abstractmethod
    def check_is_game_equal(self, game : Game) -> bool:
        """ Check if the state of the game matches the state of the GameState.
        """
        pass
    
    @abstractmethod
    def restore_game(self, game : Game) -> None:
        """ Restore the state of the game to match the state of the GameState.
        """
        pass
    
    @abstractmethod
    def to_vector(self) -> List[SupportsFloat]:
        """ Return a vector representation of the game state.
        """
        pass