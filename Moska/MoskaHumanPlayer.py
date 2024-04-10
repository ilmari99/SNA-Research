from typing import Dict, List, Set, TYPE_CHECKING
from RLFramework.Action import Action
from RLFramework.Game import Game
from RLFramework.Player import Player
import numpy as np

if TYPE_CHECKING:
    from MoskaGameState import MoskaGameState
    from MoskaAction import MoskaAction
    from Card import Card
    from MoskaGame import MoskaGame

from MoskaPlayer import MoskaPlayer

class MoskaHumanPlayer(MoskaPlayer):
    def __init__(self, name : str = "MoskaHumanPlayer", max_moves_to_consider : int = 1000, logger_args : dict = None):
        super().__init__(name, max_moves_to_consider, logger_args)
        return
    
    def choose_move(self, game: Game) -> Action:
        """ Show the valid move IDs,
        """
    
