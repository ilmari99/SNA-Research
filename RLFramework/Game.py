from abc import ABC, abstractmethod
import random
from typing import List, TYPE_CHECKING, Dict, Any
from utils import _NoneLogger, _get_logger

if TYPE_CHECKING:
    from RLFramework.Action import Action
    from RLFramework.Player import Player
    from RLFramework.GameState import GameState


class Game(ABC):
    def __init__(self,
                 logger_args: Dict[str, Any] = None,
                ):
        """ Initializes the Game instance.
        This is mainly used to set up the logger.
        """
        self.logger = _get_logger(logger_args)
        self.game_states : List[GameState] = []
        self.previous_turns : List[int] = []
        self.current_player : int = 0
        self.players : List[Player] = []
    
    def check_correct_initialization(self) -> None:
        """ Check that the game is correctly initialized.
        """
        pnames = [p.name for p in self.players]
        assert len(set(pnames)) == len(pnames), f"Player names must be unique, but were {pnames}"

        pids = [p.pid for p in self.players]
        assert pids == list(range(len(self.players))), f"Player pids must be 0, 1, 2, ..., but were {pids}"

    def internal_initialization(self, players : List[Player]) -> None:
        """ Set the players of the game.
        """
        self.players = players
        for i, player in enumerate(players):
            player.pid = i
        return

    def initialize_game_wrap(self, players : List[Player]) -> None:
        """ Wrap the initialize_game method, so that it checks the initialization and sets internal variables.
        """
        self.internal_initialization(players)
        self.initialize_game(players)
        self.check_correct_initialization()


    def play_game(self, players : List[Player]) -> List[GameState]:
        """ Play a game with the given players.
        """
        # Initilize the game by setting internal variables, custom variables, and checking the initialization
        self.initialize_game_wrap(players)

        # Play until all players are finished
        while not self.check_is_terminal():

            # Select the next player to play
            self.current_player = self._select_turn_wrap(players, self.previous_turns)
            player = players[self.current_player]

            # IF the player is finished, skip their turn
            if player.check_is_finished_wrap(self):
                continue
            # Make an action with the player
            action = player.choose_move(self)
            if action is None:
                raise Exception(f"Player {self.current_player} had no valid moves to play, even though the player is not in a terminal state.")
            # Perform the action
            new_state = self.step(action)
            # Save the new state
            self.game_states.append(new_state.deepcopy())
            # Save the previous turn
            self.previous_turns.append(self.current_player)
        return self.game_states


    def disable_logging_wrapper(self, f):
        """ A wrapper that disables logging while the function is running.
        """
        def wrapper(*args, **kwargs):
            logger = self.logger
            self.logger = _NoneLogger()
            result = f(*args, **kwargs)
            self.logger = logger
            return result
        return wrapper
    
    def mock_move_wrapper(self, f):
        """ A wrapper that mocks the move, by saving the current state, modifying the game, and restoring the state.
        """
        def wrapper(*args, **kwargs):
            game_state = GameState.from_game(self, copy = True)
            new_state = f(*args, **kwargs)
            self.restore_game(game_state)
            return new_state
        return wrapper
    
    def step(self, action: Action, real_move = True) -> GameState:
        """ Perform the given action, and calculate what is the next state.
        This step can be used, when we know the next state exactly. So we don't for example have to lift from deck.

        if real_move is False, then we disable logging, save the current state, modify the game,
        restore to the previous state, and enable logging again.
        """
        modify_game = lambda : action.modify_game(self)
        # If not real move, then wrap the modify_game function
        # with disable_logging_wrapper, and mock_move_wrapper
        if not real_move:
            modify_game = self.disable_logging_wrapper(modify_game)
            modify_game = self.mock_move_wrapper(modify_game)
        new_state = modify_game(self)
        return new_state
    

    def _select_random_turn(self, players : List[Player]) -> int:
        """ Select a random player to play.
        """
        return random.choice(range(len(players)))
    
    def _select_round_turn(self, players : List[Player]) -> int:
        """ Select the next player to play in a round-robin fashion.
        """
        # Return the next player in the list, wrapping around if necessary
        return (self.previous_turns[-1] + 1) % len(players)
    
    def _select_random_turn_exclude_last(self, players : List[Player]) -> int:
        """ Select a random player to play, excluding the last player.
        """
        return random.choice([i for i in range(len(players)) if i != self.previous_turns[-1]])
    
    def _get_finished_players(self):
        """ Return the indices of the players that are finished.
        """
        return [i for i, player in enumerate(self.players) if player.check_is_finished(GameState.from_game(self, copy = False))]
    
    def _select_turn_wrap(self, players : List[Player], previous_turns : List[int]) -> int:
        """ Wrap the selector function, so that it returns 0 if there are no previous turns.
        """
        if len(previous_turns) == 0:
            return 0
        return self.select_turn(players, previous_turns)

    def check_is_terminal(self) -> bool:
        """ The game is in a terminal state, if all players are finished.
        """
        return all([player.check_is_finished(GameState.from_game(self, copy = False)) for player in self.players])
        
    

    @abstractmethod
    def initialize_game(self, players : List[Player]) -> None:
        """ Initialize the game right before playing.
        This means dealing cards, etc. etc.
        """
        pass
    
    @abstractmethod
    def restore_game(self, game_state: GameState) -> None:
        """ Restore the game to the state described by the game_state.
        """
        pass

    @abstractmethod
    def select_turn(self, players : List[Player], previous_turns : List[int]) -> int:
        """ Given self, list of players, and the previous turns, select the next player (index) to play.
        """
        pass

    @abstractmethod
    def get_all_possible_actions(self) -> List[Action]:
        """ Return all legal moves that can be made in the current state of the game.
        The Game instance maintains knowledge of the current state of the game, and whose turn it is,
        and should be able calculate all moves that can be made in the current state.
        """
        pass

