
from collections import Counter
from typing import List

from Card import Card


from RLFramework.Action import Action

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

class Skip(MoskaAction):
    """ A class for the Skip action.
    A Skip can be made, if the player is not initiaing, or
    if the player is the target, and they need to end the turn.
    """
    
    def is_move_id_legal(self, game: MoskaGame) -> bool:
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
        
        
        
        
    
        