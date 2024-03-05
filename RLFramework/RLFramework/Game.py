from abc import ABC, abstractmethod
import numpy as np
from typing import List, TYPE_CHECKING, Dict, Any
import functools as ft

from .utils import _NoneLogger, _get_logger
from RLFramework.GameState import GameState
from RLFramework.Result import Result
if TYPE_CHECKING:
    from RLFramework.Action import Action
    from RLFramework.Player import Player


class Game(ABC):
    def __init__(self,
                 game_state_class,
                 logger_args: Dict[str, Any] = None,
                ):
        """ Initializes the Game instance.
        This is mainly used to set up the logger.
        """
        self.game_state_class : GameState = game_state_class
        self.logger_args = logger_args
        self.logger = _get_logger(logger_args)
        self.game_states : List[GameState] = []
        self.previous_turns : List[int] = []
        self.current_player : int = 0
        self.current_player_name : str = ""
        self.players : List[Player] = []
        
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.select_turn = cls.select_turn_decorator()(cls.select_turn)

    @classmethod
    def from_game_state(cls, game_state : 'GameState', logger_args: Dict[str, Any] = None):
        """ Create a Game instance from a GameState instance.
        """
        game = cls(logger_args)
        game.restore_game(game_state)
        return game
    
    def check_correct_initialization(self) -> None:
        """ Check that the game is correctly initialized.
        """
        pnames = [p.name for p in self.players]
        assert len(set(pnames)) == len(pnames), f"Player names must be unique, but were {pnames}"

        pids = [p.pid for p in self.players]
        assert pids == list(range(len(self.players))), f"Player pids must be 0, 1, 2, ..., but were {pids}"

    def internal_initialization(self, players : List['Player']) -> None:
        """ Set the players of the game.
        """
        self.players = players
        self.logger.info(f"Initialized game with {len(players)} players.")
        for i, player in enumerate(players):
            player.initialize_player(self)
        self.logger.info(f"Initialized game with {len(players)} players.")
        

    def initialize_game_wrap(self, players : List['Player']) -> None:
        """ Wrap the initialize_game method, so that it checks the initialization and sets internal variables.
        """
        self.internal_initialization(players)
        self.initialize_game(players)
        self.check_correct_initialization()


    def play_game(self, players : List['Player']) -> Result:
        """ Play a game with the given players.
        """
        # Initilize the game by setting internal variables, custom variables, and checking the initialization
        self.initialize_game_wrap(players)
        
        players_with_no_moves = []
        finishing_order = []

        # Play until all players are finished
        while not self.check_is_terminal() and len(players_with_no_moves) < len(players) - len(finishing_order):

            # Select the next player to play
            self.current_player = self.select_turn(players, self.previous_turns)
            self.current_player_name = players[self.current_player].name
            player = players[self.current_player]
            game_state = self.game_state_class.from_game(self, copy = False)

            # IF the player is finished, skip their turn
            if player.check_is_finished(game_state):
                self.logger.debug(f"Player {self.current_player_name} is finished, skipping turn.")
                continue
            self.logger.debug(f"Game state:\n{game_state}")

            # Make an action with the player
            action = player.choose_move(self)
            
            if action is None:
                #raise Exception(f"Player {self.current_player} had no valid moves to play, even though the player is not in a terminal state.")
                self.logger.info(f"Player {self.current_player_name} had no valid moves to play, but is not in a terminal state.")
                players_with_no_moves.append(self.current_player)
                continue
            players_with_no_moves = []
            self.logger.info(f"Player {self.current_player_name} chose action {action}.")
            # Perform the action
            new_state = self.step(action)
            # Save the new state
            self.game_states.append(new_state.deepcopy())
            # Save the previous turn
            self.previous_turns.append(self.current_player)
            if player.check_is_finished(new_state):
                self.logger.info(f"Player {self.current_player_name} finished the game.")
                finishing_order.append(self.current_player)
        
        game_is_draw = False
        # If the game ended, because no remaining players had any moves, then it's a draw
        if len(players_with_no_moves) == len(players) - len(finishing_order):
            self.logger.info("The remaining players are in a draw.")
            game_is_draw = True
        else:
            finishing_order += [i for i in range(len(players)) if i not in finishing_order]

        result = Result(successful = True if self.check_is_terminal() or game_is_draw else False,
                        player_jsons = [player.as_json() for player in players],
                        finishing_order = finishing_order,
                        logger_args = self.logger_args,
                        game_state_class = self.game_state_class,
                        game_states = self.game_states,
                        previous_turns = self.previous_turns,
                        )
        
        s = "Game finished with results:"
        for k,v in result.as_json(states_as_num = True).items():
            s += f"\n{k}: {v}"
        self.logger.info(s)
        return result


    def disable_logging_wrapper(self, f):
        """ A wrapper that disables logging while the function is running.
        """
        ft.wraps(f)
        def wrapper(*args, **kwargs):
            logger = self.logger
            self.logger = _NoneLogger()
            result = f(*args, **kwargs)
            self.logger = logger
            return result
        return wrapper
    
    
    def step(self, action: 'Action', real_move = True) -> 'GameState':
        """ Perform the given action, and calculate what is the next state.
        This step can be used, when we know the next state exactly. So we don't for example have to lift from deck.

        if real_move is False, then we disable logging, save the current state, modify the game,
        restore to the previous state, and enable logging again.
        """
        # If we are making a real move, we can just modify the game
        # If not, we simulate the move and then restore the game state
        curr_state = self.game_state_class.from_game(self, copy = True)
        modify_game = lambda : action.modify_game(self, inplace = real_move)
        # If not real move, then wrap the modify_game function with disable_logging_wrapper
        if not real_move:
            modify_game = self.disable_logging_wrapper(modify_game)
        new_state : 'GameState' = modify_game()
        if not real_move:
            assert curr_state.check_is_game_equal(self), "The game state was not restored correctly."
        return new_state
    

    def _select_random_turn(self, players : List['Player']) -> int:
        """ Select a random player to play.
        """
        return np.random.choice(range(len(players)))
    
    def _select_round_turn(self, players : List['Player']) -> int:
        """ Select the next player to play in a round-robin fashion.
        """
        # Return the next player in the list, wrapping around if necessary
        return (self.previous_turns[-1] + 1) % len(players)
    
    def _select_random_turn_exclude_last(self, players : List['Player']) -> int:
        """ Select a random player to play, excluding the last player.
        """
        return np.random.choice([i for i in range(len(players)) if i != self.previous_turns[-1]])
    
    def _get_finished_players(self):
        """ Return the indices of the players that are finished.
        """
        return [i for i, player in enumerate(self.players) if player.check_is_finished(self.game_state_class.from_game(self, copy = False))]
    
    @staticmethod
    def select_turn_decorator():
        """ Decorator for the select_turn method."""
        def decorator(func):
            ft.wraps(func)
            def wrapper(self, players : List['Player'], previous_turns : List[int]):
                if len(previous_turns) == 0:
                    return 0
                return func(self, players, previous_turns)
            return wrapper
        return decorator

    def check_is_terminal(self) -> bool:
        """ The game is in a terminal state, if all players are finished.
        """
        return all([player.check_is_finished(self.game_state_class.from_game(self, copy = False)) for player in self.players])
        
    

    @abstractmethod
    def initialize_game(self, players : List['Player']) -> None:
        """ Initialize the game right before playing.
        This means dealing cards, etc. etc.
        """
        pass
    
    @abstractmethod
    def restore_game(self, game_state: 'GameState') -> None:
        """ Restore the game to the state described by the game_state.
        """
        pass

    @select_turn_decorator()
    @abstractmethod
    def select_turn(self, players : List['Player'], previous_turns : List[int]) -> int:
        """ Given self, list of players, and the previous turns, select the next player (index) to play.
        """
        pass

    @abstractmethod
    def get_all_possible_actions(self) -> List['Action']:
        """ Return all legal moves that can be made in the current state of the game.
        The Game instance maintains knowledge of the current state of the game, and whose turn it is,
        and should be able calculate all moves that can be made in the current state.
        """
        pass

