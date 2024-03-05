from typing import Dict, Any, TYPE_CHECKING, List
if TYPE_CHECKING:
    from RLFramework.GameState import GameState
    from RLFramework.Action import Action
    from RLFramework.Player import Player

class Result:
    """ This class is used to describe
    what was simulated, and what we're the results.
    """
    def __init__(self,
                 successful : bool = None,
                 player_jsons : List[Dict[str, Any]] = None,
                 finishing_order : List[int] = None,
                 logger_args : Dict[str, Any] = None,
                 game_state_class : 'GameState' = None,
                 game_states : List['GameState'] = None,
                 previous_turns : List[int] = None,
        ):
        self.successful = successful
        self.player_jsons = player_jsons
        self.finishing_order = finishing_order
        self.logger_args = logger_args
        self.game_state_class = game_state_class
        self.game_states = game_states
        self.previous_turns = previous_turns

    def as_json(self, states_as_num = False) -> Dict[str, Any]:
        """ Return the result as a json.
        """
        return {"successful" : self.successful,
                "player_jsons" : self.player_jsons,
                "finishing_order" : self.finishing_order,
                "logger_args" : self.logger_args,
                "game_state_class" : self.game_state_class.__name__,
                "game_states" : len(self.game_states) if states_as_num else self.game_states,
                #"previous_turns" : self.previous_turns,
                }
