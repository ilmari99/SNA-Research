
from collections import Counter
from typing import List, TYPE_CHECKING

from Card import Card


from RLFramework.Action import Action
#from MoskaGame import MoskaGame
from utils import check_can_kill_card

if TYPE_CHECKING:
    from MoskaGameState import MoskaGameState
    from MoskaGame import MoskaGame

VALID_MOVE_IDS = ["AttackInitial",
                  "AttackOther",
                  "AttackSelf",
                  "KillFromHand",
                  "KillFromDeck",
                  "EndBout",
                  "Skip"
]

def get_moska_action(pid : int, move_id : str, *args, **kwargs) -> 'MoskaAction':
    """ Get a Moska action based on the move_id.
    """
    if move_id == "AttackInitial":
        return AttackInitial(pid, move_id, *args, **kwargs)
    if move_id == "AttackOther":
        return AttackOther(pid, move_id, *args, **kwargs)
    if move_id == "AttackSelf":
        return AttackSelf(pid, move_id, *args, **kwargs)
    if move_id == "KillFromHand":
        return KillFromHand(pid, move_id, *args, **kwargs)
    if move_id == "KillFromDeck":
        return KillFromDeck(pid, move_id, *args, **kwargs)
    if move_id == "EndBout":
        return EndBout(pid, move_id, *args, **kwargs)
    if move_id == "Skip":
        return Skip(pid, move_id, *args, **kwargs)
    raise ValueError(f"Invalid move_id: {move_id}")

class MoskaAction(Action):
    """ Superclass for all Moska actions.
    """
    def __init__(self, pid : int, move_id : str, *args, **kwargs):
        # Check if the move_id is valid
        if move_id not in VALID_MOVE_IDS:
            raise ValueError(f"Invalid move_id: {move_id}")
        self.pid = pid
        self.move_id = move_id
        self.args = args
        self.kwargs = kwargs
        
    def is_move_id_legal(self, game : 'MoskaGame') -> bool:
        """ Check if the move_id in general (no arguments) is
        legal in the game.
        """
        pass
        
    def modify_game(self, game, inplace = False):
        pass
    
    def check_action_is_legal(self, game):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pid={self.pid}, move_id={self.move_id}, args={self.args}, kwargs={self.kwargs})"
    
    def __init_subclass__(cls) -> None:
        # wrap the check_action_is_legal method
        cls.check_action_is_legal = cls.check_action_is_legal_wrapper(cls.check_action_is_legal)
    
    def check_action_is_legal_wrapper(func):
        """ Wrapper for the check_action_is_legal method.
        """
        def wrapper(self, game):
            try:
                return func(self, game)
            except AssertionError as e:
                print(f"AssertionError: {e}", flush=True)
                return False
        return wrapper
    
class EndBout(MoskaAction):
    """ A class for the EndBout action.
    The EndBout action can be made when
    the game has been initiated, and the target is the player,
    and all other cards have been killed.
    """

    def __init__(self, pid : int, move_id : str, cards_to_lift : List[Card]):
        super().__init__(pid, move_id, cards_to_lift)
        self.cards_to_lift = cards_to_lift

    def _lifting_only_cards_to_kill(self, game : 'MoskaGame') -> bool:
        """ Check if the player is lifting only cards to kill.
        """
        return all(card in game.cards_to_kill for card in self.cards_to_lift)

    def _all_others_ready(self, game : 'MoskaGame') -> bool:
        """ Check if all other players are ready.
        """
        return all(game.players_ready[i] for i in range(len(game.players)) if i != self.pid)
    
    def _is_initiated(self, game : 'MoskaGame') -> bool:
        """ Check if the game has been initiated.
        """
        return len(game.cards_to_kill + game.killed_cards) > 0
    
    def _is_target(self, game : 'MoskaGame') -> bool:
        """ Check if the player is the target.
        """
        return self.pid == game.target_pid
    
    def check_action_is_legal(self, game : 'MoskaGame') -> bool:
        assert self._all_others_ready(game), "Not all other players are ready."
        assert self._is_initiated(game), "The game has not been initiated."
        assert self._is_target(game), "The player is not the target."
        return True
    
    def _modify_game_only_lifted_cards_to_kill(self, game : 'MoskaGame') -> 'MoskaGameState':
        """ Modify the game state, if the player is only lifting cards to kill.
        """
        # Add the killed cards to discard
        game.discarded_cards += game.killed_cards
        game.killed_cards = []
        game.cards_to_kill = []
        game.player_full_cards[self.pid] += self.cards_to_lift
        game.player_public_cards[self.pid] += self.cards_to_lift
        game_state = game.game_state_class.from_game(game, copy = False)
        return game_state
    
    def _modify_game_lifted_all_cards(self, game : 'MoskaGame') -> 'MoskaGameState':
        """ Modify the game state, if the player is lifting cards to kill and cards from hand.
        """
        # Empty the table, and add the cards to the player's hand
        game.player_full_cards[self.pid] += self.cards_to_lift
        game.player_public_cards[self.pid] += self.cards_to_lift
        game.cards_to_kill = []
        game.killed_cards = []
        game_state = game.game_state_class.from_game(game, copy = False)
        return game_state


    
    def modify_game(self, game, inplace=False):
        """ We lift the cards to our hand, and update the public cards
        """

class Skip(MoskaAction):
    """ A class for the Skip action.
    A Skip can be made, if the player is not initiaing, or
    if the player is the target, and they need to end the turn.
    """
    
    def is_move_id_legal(self, game: 'MoskaGame') -> bool:
        """ The Skip move can be played as long as
        the player is not initiating, or
        other players are finished, and the player is the target
        """
    
    def check_action_is_legal(self, game):
        player = game.players[self.pid]
    
    def modify_game(self, game, inplace=False):
        game_state = game.game_state_class.from_game(game, copy = False)
        return game_state


class AttackMove(MoskaAction):
    """ A superclass for attacking type moves
    """
    def __init__(self, pid : int, move_id : str, target_pid : int, cards : List[Card]):
        super().__init__(pid, move_id, target_pid, cards)
        self.target_pid = target_pid
        self.cards = cards
        
    def _attack_cards_are_available(self, game):
        """ Check if the cards are available in the player's hand.
        """
        return all(card in game.players[self.pid].hand for card in self.cards)
    
    def _pids_are_valid(self, game):
        """ Check if the player ids are valid and both are
        in the game.
        """
        return self.pid in range(len(game.players)) and self.target_pid in range(len(game.players))
        
    def _target_pid_is_target(self, game):
        """ Check if the target_pid is the target player.
        """
        return self.target_pid == game.target_player
    
    def _ranks_are_in_table(self, game):
        """ Check if the card ranks are in
        cards_to_kill or killed_cards.
        """
        if any(card not in game.cards_to_kill + game.killed_cards for card in self.cards):
            return False
        return True
    
    def _attacker_and_target_are_different(self, game):
        """ Check if the attacker and target are the same player.
        """
        return self.pid != self.target_pid
    
    def _attack_cards_fit(self, game):
        """ Check if the cards fit the cards to kill.
        """
        return len(game.players[self.target_pid].hand) - len(game.cards_to_kill) >= len(self.cards)
        
    
    def check_action_is_legal(self, game):
        assert self._pids_are_valid(game), "The player ids are invalid."
        assert self._target_pid_is_target(game), "The target_pid is not the target player."
        assert self._attack_cards_are_available(game), "The attack cards are not available."
        #assert self._ranks_are_in_table(game), "The card ranks are not in cards_to_kill or killed_cards."
        return True
    
    def modify_game(self, game, inplace=False) -> 'MoskaGameState':
        """ Modify the game instance according to the action.
        """
        # Remove the cards from the player's hand
        game.players[self.pid].hand = [card for card in game.players[self.pid].hand if card not in self.cards]
        # Add the cards to the cards_to_kill
        game.cards_to_kill += self.cards
        game_state = game.game_state_class.from_game(game, copy = False)
        return game_state
    
class AttackInitial(AttackMove):

    
    def _board_is_empty(self, game):
        """ Check if the board is empty.
        """
        return len(game.cards_to_kill + game.killed_cards) == 0
    
    def _attacker_and_target_are_different(self, game):
        """ Check if the attacker and target are the same player.
        """
        return self.pid != self.target_pid
    
    def _cards_are_single_or_multiple(self, game):
        """ In an initial attack, the cards can be either single cards,
        or multiple cards of the same rank.
        """
        c = Counter([c.rank for c in self.cards])
        return len(self.cards) == 1 or all((count >= 2 for count in c.values()))
    
    def check_action_is_legal(self, game) -> bool:
        assert self._cards_are_single_or_multiple(game), "The cards are not single or multiple cards of the same rank."
        assert self._attack_cards_fit(game), "The attack cards do not fit the cards to kill."
        assert self._attacker_and_target_are_different(game), "The attacker and target are the same player."
        assert self._board_is_empty(game), "The board is not empty."
        return super().check_action_is_legal(game)
    
class AttackSelf(AttackMove):
    def _attacker_and_target_are_same(self, game):
        """ Check if the attacker and target are the same player.
        """
        return self.pid == self.target_pid
    
    def check_action_is_legal(self, game):
        assert self._attacker_and_target_are_same(game), "The attacker and target are not the same player."
        assert self._ranks_are_in_table(game), "The card ranks are not in cards_to_kill or killed_cards."
        return super().check_action_is_legal(game)
    
class AttackOther(AttackMove):
    def check_action_is_legal(self, game):
        assert self._attack_cards_fit(game), "The attack cards do not fit the cards to kill."
        assert self._ranks_are_in_table(game), "The card ranks are not in cards_to_kill or killed_cards."
        assert self._attacker_and_target_are_different(game), "The attacker and target are the same player."
        return super().check_action_is_legal(game)
    
class KillFromHand(MoskaAction):
    def __init__(self, pid : int, move_id : str, kill_mapping : dict):
        super().__init__(pid, move_id, kill_mapping)
        self.kill_mapping = kill_mapping
    
    def _killing_cards_are_in_hand(self, game: 'MoskaGame'):
        """ Check if the killing cards are in the player's hand.
        """
        return all(card in game.player_full_cards[self.pid] for card in self.kill_mapping.keys())
    
    def _kill_mapping_is_valid(self, game: 'MoskaGame'):
        """ Check if the kill mapping is valid.
        """
        if not isinstance(self.kill_mapping, dict):
            return False
        if not all(isinstance(c, Card) and isinstance(v, Card) for c, v in self.kill_mapping.items()):
            return False
        if not all(check_can_kill_card(c, v, game.trump_suit) for c, v in self.kill_mapping.items()):
            return False
        return True
    
    def check_action_is_legal(self, game: 'MoskaGame'):
        assert self._killing_cards_are_in_hand(game), "The killing cards are not in the player's hand."
        assert self._kill_mapping_is_valid(game), "The kill mapping is not valid."
        return True
    
    def modify_game(self, game, inplace=False) -> 'MoskaGameState':
        """ Modify the game instance according to the action.
        """
        cards_from_hand = list(self.kill_mapping.keys())
        cards_on_table = list(self.kill_mapping.values())
        # Remove the cards from the player's hand
        game.player_full_cards[self.pid] = [card for card in game.player_full_cards[self.pid] if card not in cards_from_hand]
        # Remove the killed cards from the table
        game.cards_to_kill = [card for card in game.cards_to_kill if card not in cards_on_table]
        # Add the cards from hand, and the killed cards to the killed cards
        cards_to_killed_cards = []
        for hand_card, table_card in self.kill_mapping.items():
            cards_to_killed_cards.append(hand_card)
            cards_to_killed_cards.append(table_card)
        game.killed_cards += cards_to_killed_cards
        game_state = game.game_state_class.from_game(game, copy = False)
        return game_state
        
        
    
        