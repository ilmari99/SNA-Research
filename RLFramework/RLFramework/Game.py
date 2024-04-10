from abc import ABC, abstractmethod
import numpy as np
from typing import List, TYPE_CHECKING, Dict, Any
import functools as ft
import matplotlib.pyplot as plt

from .utils import _NoneLogger, TFLiteModel, _get_logger
from .GameState import GameState
from .Result import Result
if TYPE_CHECKING:
    from .Action import Action
    from .Player import Player


class Game(ABC):
    def __init__(self,
                 game_state_class,
                 logger_args: Dict[str, Any] = None,
                 render_mode : str = "",
                 gather_data : str = "",
                 custom_result_class = None,
                 max_num_total_steps : int = 1000,
                ):
        """ Initializes the Game instance.
        This is mainly used to set up the logger.
        """
        self.result_class = custom_result_class if custom_result_class else Result
        self.gather_data = gather_data
        if render_mode == "human":
            self.init_render_human()
        self.max_num_total_steps = max_num_total_steps
        self.render_mode = render_mode
        self.game_state_class : GameState = game_state_class
        self.logger_args = logger_args
        self.logger = _get_logger(logger_args)
        self.game_states : List[GameState] = []
        self.previous_turns : List[int] = []
        self.unfinished_players : List[int] = []
        self.finishing_order : List[int] = []
        self.player_scores : List[float] = []
        self.current_pid : int = 0
        self.current_player_name : str = ""
        self.players : List[Player] = []
        self.total_num_played_turns = 0
        self.verify_self()
        
    def verify_self(self) -> None:
        """ Verify that the game is correctly initialized.
        """
        assert self.render_mode in ["human", "text", ""], f"render_mode must be 'human', 'text', or '', not {self.render_mode}"
        assert (self.gather_data and len(self.gather_data) > 4) or not self.gather_data, "gather_data must be a string with length > 4, or \"\"."
        assert isinstance(self.game_state_class, type), f"game_state_class must be a class, not {type(self.game_state_class)}"
        assert issubclass(self.game_state_class, GameState), f"game_state_class must be a subclass of GameState, not {self.game_state_class}"
        assert isinstance(self.result_class, type), f"result_class must be a class, not {type(self.result_class)}"
        assert issubclass(self.result_class, Result), f"result_class must be a subclass of Result, not {self.result_class}"
    
    def reset(self) -> None:
        """ Reset the game to the initial state, with no player data.
        """
        self.game_states = []
        self.previous_turns = []
        self.unfinished_players = []
        self.finishing_order = []
        self.player_scores = []
        self.current_pid = 0
        self.current_player_name = ""
        self.players = []
        self.total_num_played_turns = 0
        
        if self.render_mode == "human":
            self.init_render_human()
        return
        
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        #cls.select_turn = cls.select_turn_decorator()(cls.select_turn)
        cls.check_is_player_finished = ft.lru_cache(maxsize = 256)(cls.check_is_player_finished)
        cls.environment_action = cls._environment_action_wrapper(cls.environment_action)

    def __repr__(self) -> str:
        return self.get_current_state().__repr__()
    
    def get_current_state(self, player : 'Player' = None) -> GameState:
        """ Return the current state of the game.
        """
        player = player if player else self.players[self.current_pid]
        return self.game_state_class.from_game(self, player=player, copy = False)
    
    def render_self(self) -> None:
        """ Render the game state.
        """
        print(self)
        
    def render_nothing(self) -> None:
        """ Do nothing.
        """
        pass
    
    def init_render_human(self) -> None:
        """ Create a figure and axis for rendering the game state.
        """
        plt.cla()
        plt.clf()
        plt.close()
        self.fig, self.ax = plt.subplots()
        plt.ion()
        plt.show()
        
    def render_human(self, ax : plt.Axes = None) -> None:
        """ Plot the game state in a human readable way.
        """
        raise NotImplementedError("The render_human method must be implemented in the subclass.")
    
    def render(self):
        """ Render the game state.
        """
        if self.render_mode == "human":
            self.render_human(self.ax)
        elif self.render_mode == "":
            self.render_nothing()
        elif self.render_mode == "text":
            self.render_self()
        else:
            raise ValueError(f"Render mode '{self.render_mode}' not recognized.")
        
        

    @classmethod
    def from_game_state(cls, game_state : 'GameState', *args, **kwargs) -> 'Game':
        """ Create a Game instance from a GameState instance.
        """
        game = cls(*args, **kwargs)
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
        for i, player in enumerate(players):
            player.initialize_player(self)
        self.logger.info(f"Initialized game with {len(players)} players.")
        self.unfinished_players = [i for i in range(len(players))]
        self.player_scores = [0.0 for _ in range(len(players))]
        self.finishing_order = []
        self.total_num_played_turns = 0
        self.current_pid = 0


    def initialize_game_wrap(self, players : List['Player']) -> None:
        """ Wrap the initialize_game method, so that it checks the initialization and sets internal variables.
        """
        self.internal_initialization(players)
        self.initialize_game(players)
        self.check_correct_initialization()


    def play_game(self, players : List['Player']) -> Result:
        """ Play a game with the given players.
        """
        self.reset()
        # Initilize the game by setting internal variables, custom variables, and checking the initialization
        self.initialize_game_wrap(players)
        
        self.render()
        self.game_states.append(self.get_current_state().deepcopy())
        # Play until all players are finished
        while not self.check_is_terminal() and self.total_num_played_turns < self.max_num_total_steps:
            # Select the next player to play
            self.current_player_name = players[self.current_pid].name
            player = players[self.current_pid]
            self.previous_turns.append(self.current_pid)
            game_state = self.get_current_state(player)
            assert game_state.check_is_game_equal(self), ("The game state was not created correctly. The created ",
                                                          "GameState is not equal to the game according to the ",
                                                          "game state's 'check_is_game_equal' method.")

            # Choose an action with the player
            action = player.choose_move(self)
            if action is not None:
                # First, we take the step, which modifies self.
                # We then save this state (after action).
                self.logger.info(f"Player {self.current_player_name} chose action {action}.")
                new_state : 'GameState' = self.step(action)
                new_state = self.get_current_state(player=player)
                self.logger.debug(f"New state after action:\n{new_state}")
                self.game_states.append(new_state.deepcopy())
                # After every action, the environment reacts.
                # For example, we might add cards to players with missing cards, or change the current player.
                s = self.environment_action(new_state)
                # If the environment action returns something other than False, we set the new state to that.
                if s is not False:
                    new_state = s
                    self.game_states.append(new_state.deepcopy())
                    new_state.set_game_state(self)
                    #print(f"New state after environment action:\n{new_state}")
                    assert new_state.check_is_game_equal(self, player=player), ("The game state was not restored correctly. The created ",
                                                                  "GameState is not equal to the game according to the ",
                                                                  "game state's 'check_is_game_equal' method.")
                self.total_num_played_turns += 1
            else:
                new_state = game_state
                self.logger.info(f"Player '{self.current_player_name}' has no moves.")

            self.logger.debug(f"Game state:\n{new_state}")
            self.render()
        
        # Winner is the player with the higher score
        winner = players[np.argmax(self.player_scores)].name
        players_with_max_score = [p.name for p in players if p.score == max(self.player_scores)]

        # If multiple players have the same score, the winner is None
        if len(players_with_max_score) > 1:
            winner = None
        
        result = self.result_class(successful = True if self.check_is_terminal() else False,
                        player_jsons = [player.as_json() for player in players],
                        finishing_order = self.finishing_order,
                        logger_args = self.logger_args,
                        game_state_class = self.game_state_class,
                        game_states = self.game_states,
                        previous_turns = self.previous_turns,
                        winner=winner,
                        )
        s = "Game finished with results:"
        for k,v in result.as_json(states_as_num = True).items():
            s += f"\n{k}: {v}"
        self.logger.info(s)
        if self.gather_data and result.successful:
            result.save_game_states_to_file(self.gather_data)
        return result
    
    def calculate_reward(self, pid : int, new_state : 'GameState'):
        """ Calculate the reward for the player that made the move.
        """
        return 0.0


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
    
    @staticmethod
    def _environment_action_wrapper(f):
        """ A wrapper that wraps 'environment_action' and
        updates the finishing order and player scores.
        """
        ft.wraps(f)
        def wrapper(self : 'Game', game_state : 'GameState'):
            state = f(self, game_state)
            if state is False:
                return False
            self.update_finished_players_in_gamestate(state)
            self.update_player_scores_in_gamestate(state)
            self.update_player_attributes()
            
            self.logger.debug(f"Environment action finished. Game state:\n{state}")
            return state
        return wrapper
    

    @_environment_action_wrapper
    def environment_action(self, game_state : 'GameState') -> 'GameState':
        """ When creating a new game, the user can define what happens after every action.
        In an Action, the user should set all variables, that are exactly known after each action.
        Such as which move was made, what is the board state, etc.

        In this function, the user should define how the environment reacts to being in a certain state.
        For example:
        - If a player is missing cards, we should add cards to the player
        - If the next player is not known, we should choose the next player
        - If there is a fruit missing in Snake, we should add a fruit to the board

        Nevertheless, this method (it's wrapper) is used to update which players have finished,
        and to increment the rewards of the players.
        """
        return False
    
    
    def update_player_attributes(self) -> None:
        for pl in self.players:
            pl.score = self.player_scores[pl.pid]
            pl.is_finished = pl.pid in self.finishing_order
        return

    def step(self, action: 'Action', real_move = True) -> 'GameState':
        """ Perform the given action, and calculate what is the next state.
        This step can be used, when we know the next state exactly. So we don't for example have to lift from deck.

        if real_move is False, then we disable logging, save the current state, modify the game,
        restore to the previous state, and enable logging again.
        """
        # If we are making a real move, we can just modify the game
        # If not, we simulate the move and then restore the game state
        if not real_move:
            curr_state = self.game_state_class.from_game(self, player=self.players[self.current_pid], copy = True)

        def make_action_and_update_vars():
            """ Calculate a game state after making the action.
            Also update the internal variables of the game_state.
            If the move is real, we also apply the move to the game.
            """
            # Calculate the new state after making the action
            new_state = action.modify_game(self, inplace = real_move)
            # If real move, then we also modify self
            if real_move:
                new_state.set_game_state(self)
                self.update_player_attributes()

            return new_state

        # If not real move, then wrap the modify_game function with disable_logging_wrapper
        if not real_move:
            make_action_and_update_vars = self.disable_logging_wrapper(make_action_and_update_vars)
        new_state : 'GameState' = make_action_and_update_vars()
        if not real_move:
            assert curr_state.check_is_game_equal(self), "The game state was not restored correctly."
        return new_state
    
    def _get_finished_players(self, game_state: GameState) -> List[int]:
        """ Return the indices of the players that are finished.
        """
        return [i for i in range(len(self.players)) if self.check_is_player_finished(i, game_state)]
    
    def update_finished_players_in_gamestate(self, game_state: GameState) -> None:
        """ Update the finishing_order and unfinished_players.
        """
        finished_players = self._get_finished_players(game_state)
        # If there are new finished players, append them to the finishing_order
        for pid in finished_players:
            if pid not in game_state.finishing_order:
                game_state.finishing_order.append(pid)
        # Remove the finished players from the unfinished_players
        game_state.unfinished_players = [pid for pid in game_state.unfinished_players if pid not in finished_players]
        return
    
    def update_player_scores_in_gamestate(self, game_state: GameState) -> None:
        """ Add a reward to the players.
        """
        #print(game_state.state_json)
        for pid in range(len(self.players)):
            r = self.calculate_reward(pid, game_state)
            game_state.player_scores[pid] += r
        return

    def check_is_terminal(self) -> bool:
        """ The game is in a terminal state, if all players are finished.
        """
        return len(self.unfinished_players) == 0
    
    @ft.lru_cache(maxsize = 256)
    @abstractmethod
    def check_is_player_finished(self, pid : int, game_state: GameState) -> bool:
        """ Check if a player has finished in the game_state.
        """
        pass

    @abstractmethod
    def initialize_game(self, players : List['Player']) -> None:
        """ Initialize the game right before playing.
        This means dealing cards, etc. etc.
        """
        pass
    
    @abstractmethod
    def restore_game(self, game_state: 'GameState') -> None:
        """ Restore the game to the state described by the game_state.
        The common variables, such as unfinished players, current player, etc. are restored automatically.
        """
        pass

    @abstractmethod
    def get_all_possible_actions(self) -> List['Action']:
        """ Return all legal moves that can be made in the current state of the game.
        The Game instance maintains knowledge of the current state of the game, and whose turn it is,
        and should be able calculate all moves that can be made in the current state.
        """
        pass

