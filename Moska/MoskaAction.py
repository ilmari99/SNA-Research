
from collections import Counter
from typing import List, TYPE_CHECKING, Tuple

from Card import Card


from RLFramework.Action import Action
#from MoskaGame import MoskaGame
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
        
    def _all_others_ready(self, game : 'MoskaGame') -> bool:
        """ Check if all other players are ready.
        """
        return all(game.ready_players[i] for i in range(len(game.players)) if i != self.pid)
    
    def _is_initiated(self, game : 'MoskaGame') -> bool:
        """ Check if the game has been initiated.
        """
        return len(game.cards_to_kill + game.killed_cards) > 0
    
    def _is_target(self, game : 'MoskaGame') -> bool:
        """ Check if the player is the target.
        """
        return self.pid == game.target_pid
    
    def _board_is_empty(self, game : 'MoskaGame'):
        """ Check if the board is empty.
        """
        return len(game.cards_to_kill + game.killed_cards) == 0
        
    def is_move_id_legal(self, game : 'MoskaGame') -> bool:
        """ Check if the move_id in general (no arguments) is
        legal in the game.
        """
        pass
    
    def __init_subclass__(cls) -> None:
        cls.modify_game = cls._do_after_action_wrapper(cls.modify_game)
        return super().__init_subclass__()
    
    def check_has_duplicate_cards_on_players(self, game : 'MoskaGame') -> bool:
        """ Check if there are no duplicate cards in the action.
        """
        for player in game.players:
            if len(player.hand) != len(set(player.hand)):
                duplicated_cards = [card for card, count in Counter(player.hand).items() if count > 1]
                #print(f"Player {player.pid} has duplicate cards in hand: {duplicated_cards}")
                return True
        return False
    
    @staticmethod
    def _do_after_action_wrapper(func):
        def wrapper(self : 'MoskaAction', game : 'MoskaGame'):
            state : 'MoskaGameState' = func(self, game)
            state.set_game_state(game)
            return self.do_after_action(game)
        return wrapper
    
    def check_every_public_card_is_in_full_cards(self, game : 'MoskaGame') -> bool:
        """ Check if every public card is in the full cards.
        """
        for pid in range(len(game.players)):
            pub_cards = game.player_public_cards[pid]
            full_cards = game.player_full_cards[pid]
            if not all(card in full_cards for card in pub_cards):
                return False
        return True
    
    def check_valid_state(self, game : 'MoskaGame') -> bool:
        """ Check if the state is valid.
        """
        msg = ""
        if self.check_has_duplicate_cards_on_players(game):
            msg += "There are duplicate cards on players hands."
        if len(game.deck) != len(set(game.deck)):
            msg += "There are duplicate cards in the deck."
        if len(game.discarded_cards) != len(set(game.discarded_cards)):
            msg += "There are duplicate cards in the discarded cards."
        if len(game.cards_to_kill) != len(set(game.cards_to_kill)):
            msg += "There are duplicate cards in the cards to kill."
        if len(game.killed_cards) != len(set(game.killed_cards)):
            msg += "There are duplicate cards in the killed cards."
        if not self.check_every_public_card_is_in_full_cards(game):
            msg += "Some players have extra cards in their public cards."
        # Check if someone has kopled cards in hand
        for player in game.players:
            if not(player.pid == game.target_pid and game.target_is_kopling) and any(card.kopled for card in player.hand):
                msg += f"Player {player.pid} has kopled cards in hand, but is not the target."
        return msg
        

    def do_after_action(self, game : 'MoskaGame') -> 'MoskaGameState':
        """ After a Moska action is done, we set the next player to unknown (-1),
        since the next player is usually selected randomly. The only exception is 
        when the bout is ended. In that case:
        - If the target picks up any cards, the target is shifted by two and the turn by one.
        - If the target does not pick up any cards, the target is shifted by one, and the turn remains.

        Another exception (not implemented yet), is when a player wants to play from deck.
        In that case, the turn remains, and the state will have a flag indicating that the target is playing from the deck.
        Then, the environment makes an action: Add the card to the table, OR add a card to a separate set of cards.
        Then, the player must kill a card on the table, with that card.
        """
        msg = self.check_valid_state(game)
        gs : 'MoskaGameState' = game.get_current_state()
        if msg:
            msg = f"The game state \n{gs} is not valid: " + msg
            raise ValueError(msg)
        
        # If the target kills cards, and finished, ten we must play the EndBout action.
        if self.pid == game.target_pid and self.move_id != "EndBout":
            if game.check_is_player_finished(self.pid, gs):
                #print(f"Player {self.pid} has finished.")
                #print(game.get_current_state())
                act = EndBout(self.pid, "EndBout", [])
                game.ready_players = [True for _ in range(len(game.players))]
                return act.modify_game(game)
            
        
        if self.move_id == "EndBout":
            assert hasattr(self, "cards_to_lift"), "EndBout action must have cards_to_lift attribute."
            gs : 'MoskaGameState' = game.get_current_state()
            is_finished = game.check_is_player_finished(self.pid, gs)
            current_player_pids = [i for i in range(len(game.players)) if not game.check_is_player_finished(i, gs) or i == self.pid]
            #curr_pid_idx = current_player_pids.index(game.current_pid)
            curr_target_idx = current_player_pids.index(game.target_pid)
            #print(f"Current target: {game.target_pid}, current target index: {curr_target_idx}")
            # If the target finisehs with this EndBout, the target is shifted by two, and the turn by one.
            if is_finished:
                current_player_pids = [i for i in range(len(game.players)) if not game.check_is_player_finished(i, gs)]
                # If everyone is finished, game is over.
                if not current_player_pids:
                    return game.game_state_class.from_game(game, copy = False)
                game.target_pid = current_player_pids[(curr_target_idx + 2) % len(current_player_pids)]
                game.current_pid = current_player_pids[(curr_target_idx + 1) % len(current_player_pids)]
                #print(f"Target finished.")
            # If the player lifts any cards, the target is shifted by two, and the turn by one.
            elif len(self.cards_to_lift) > 0:
                game.target_pid = current_player_pids[(curr_target_idx + 2) % len(current_player_pids)]
                game.current_pid = current_player_pids[(curr_target_idx + 1) % len(current_player_pids)]
                #print(f"Lifted cards")
            # If the player does not lift any cards, and does not finish, the target is shifted by one, and the turn remains.
            else:
                game.target_pid = current_player_pids[(curr_target_idx + 1) % len(current_player_pids)]
                game.current_pid = current_player_pids[curr_target_idx]
                #print(f"Did not lift cards")
            #print(f"New target pid: {game.target_pid}, new current pid: {game.current_pid}")
        elif self.move_id == "KillFromDeck":
            # If the target is killing from the deck, we do not change the pid here
            pass
        
        # In all other cases, the current_pid is set to -1, so the environment decides the next player.
        else:
            game.current_pid = -1

        if game.check_is_player_finished(self.pid, gs):
            game.logger.info(f"Player {self.pid} has finished.")
        return game.game_state_class.from_game(game, copy = False)
        
    def modify_game(self, game, inplace = False):
        pass
    
    def check_action_is_legal(self, game):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pid={self.pid}, move_id={self.move_id}, args={self.args}, kwargs={self.kwargs})"
    
class KillFromDeck(MoskaAction):
    """ Attempt to kill by picking the next card from the deck.
    This is a special action, that can only be played if:
    - The player is the target,
    - There are cards to kill,
    - There are no card_to_kill that have been played from the deck.
    - There is deck left,
    This move works as follows:
    - When this move is called a target_is_kopling flag is set in the game state,
    indicating that the target wants to play from the deck. The current_pid is not changed.
    - The environment then picks the next card from the deck and:
        - If it can't kill any card: add it to cards_to_kill (set kopled flag of the card to True),
        change current_pid, set target_is_kopling flag to False.
        - If it CAN kill a card, add the card to the players hand (set kopled flag of the card to True),
        keep the current_pid, and keep target_is_kopling flag to True.
    - If the card is in the player's hand, the player must kill a card from the table using that card.
    """
    def __init__(self, pid : int, move_id : str):
        super().__init__(pid, move_id)
        
    def is_move_id_legal(self, game : 'MoskaGame') -> bool:
        """ Check if the KillFromDeck move can be played.
        """
        if game.target_is_kopling:
            return False
        gs : 'MoskaGameState' = game.get_current_state()
        if not self._is_target(game):
            return False
        if len(gs.cards_to_kill) == 0:
            return False
        if not gs.target_can_play_from_deck():
            return False
        return True
    
    def check_action_is_legal(self, game : 'MoskaGame') -> Tuple[bool, str]:
        msg = ""
        gs : 'MoskaGameState' = game.get_current_state()
        if not self._is_target(game):
            msg += "The player is not the target."
        if not self._is_initiated(game):
            msg += "The game has not been initiated."
        if not gs.target_can_play_from_deck():
            msg += "The target can not play from the deck, since there are kopled cards on the table."
        is_legal = True if not msg else False
        return is_legal, msg
    
    def modify_game(self, game : 'MoskaGame', inplace=False):
        """ Modify the game state, if the player is playing from the deck.
        """
        game.target_is_kopling = True
        return game.game_state_class.from_game(game, copy = False)
    
class EndBout(MoskaAction):
    """ A class for the EndBout action.
    The EndBout action can be made when
    the game has been initiated, and the target is the player,
    and all other cards have been killed.
    """
    def __init__(self, pid : int, move_id : str, cards_to_lift : List[Card] = []):
        super().__init__(pid, move_id, cards_to_lift)
        self.cards_to_lift = cards_to_lift

    def _lifting_only_cards_to_kill(self, game : 'MoskaGame') -> bool:
        """ Check if the player is lifting only cards to kill.
        """
        return all(card in game.cards_to_kill for card in self.cards_to_lift)
    
    def is_move_id_legal(self, game : 'MoskaGame') -> bool:
        """ Check if the EndBout move can be played.
        """
        if game.target_is_kopling:
            return False
        if self._all_others_ready(game) and self._is_initiated(game) and self._is_target(game):
            return True
        return False
    
    def check_action_is_legal(self, game : 'MoskaGame') -> Tuple[bool, str]:
        msg = ""
        if not self._all_others_ready(game):
            msg += "Not all other players are ready."
        if not self._is_initiated(game):
            msg += "The game has not been initiated."
        if not self._is_target(game):
            msg += "The player is not the target."
        is_legal = True if not msg else False
        return is_legal, msg
    
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
        # Set all cards kopled to False
        for card in game.cards_to_kill:
            card.kopled = False
        for card in self.cards_to_lift:
            card.kopled = False
        if self._lifting_only_cards_to_kill(game):
            gs = self._modify_game_only_lifted_cards_to_kill(game)
        else:
            gs = self._modify_game_lifted_all_cards(game)
        return gs

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
        if game.target_is_kopling:
            return False
        gs : MoskaGameState = game.get_current_state()
        if self._is_target(game) and self._all_others_ready(game):
            return False
        if not self._is_initiated(game) and gs.player_is_initiating(self.pid):
            return False
        return True
    
    def check_action_is_legal(self, game : 'MoskaGame') -> bool:
        return True, ""
    
    def modify_game(self, game : 'MoskaGame', inplace=False):
        return game.game_state_class.from_game(game, copy = False)


class AttackMove(MoskaAction):
    """ A superclass for attacking type moves
    """
    __slots__ = ["pid", "move_id", "target_pid", "cards"]
    def __init__(self, pid : int, move_id : str, target_pid : int = None, cards : List[Card] = []):
        super().__init__(pid, move_id, target_pid, cards)
        self.target_pid = target_pid
        self.cards = cards
        
    def is_move_id_legal(self, game: 'MoskaGame') -> bool:
        if game.target_is_kopling:
            return False
        return
    
    def _target_pid_is_target(self, game : 'MoskaGame') -> bool:
        """ Check if the target_pid is the target player.
        """
        return self.target_pid == game.target_pid
        
    def _attack_cards_are_available(self, game : 'MoskaGame') -> bool:
        """ Check if the cards are available in the player's hand.
        """
        return all(card in game.players[self.pid].hand for card in self.cards)
    
    def _pids_are_valid(self, game : 'MoskaGame') -> bool:
        """ Check if the player ids are valid and both are
        in the game.
        """
        return self.pid in range(len(game.players)) and self.target_pid in range(len(game.players))
        
    def _target_pid_is_target(self, game : 'MoskaGame') -> bool:
        """ Check if the target_pid is the target player.
        """
        return self.target_pid == game.target_pid
    
    def _ranks_are_in_table(self, game : 'MoskaGame') -> bool:
        """ Check if the card ranks are in
        cards_to_kill or killed_cards.
        """
        card_ranks = game.get_ranks_on_table()
        played_ranks = set([card.rank for card in self.cards])
        return all(rank in card_ranks for rank in played_ranks)
    
    def _attacker_and_target_are_different(self, game : 'MoskaGame'):
        """ Check if the attacker and target are the same player.
        """
        return self.pid != self.target_pid
    
    def _attack_cards_fit(self, game : 'MoskaGame'):
        """ Check if the cards fit the cards to kill.
        """
        return len(game.cards_to_kill) + len(self.cards) <= len(game.player_full_cards[self.target_pid])
        
    
    def check_action_is_legal(self, game) -> Tuple[bool, str]:
        msg = ""
        if not self._pids_are_valid(game):
            msg += "The player ids are not valid."
        if not self._attack_cards_are_available(game):
            msg += "The attack cards are not available in the player's hand."
        if not self._target_pid_is_target(game):
            msg += "The target_pid is not the target player."
        if not self._ranks_are_in_table(game):
            msg += "The card ranks can not be played, as they are not in cards_to_kill or killed_cards."
        if not self._attacker_and_target_are_different(game):
            msg += "The attacker and target are the same player."
        if not self._attack_cards_fit(game):
            msg += "The attack cards do not fit the cards to kill."
        is_legal = True if not msg else False
        return is_legal, msg
    
    
    def modify_game(self, game : 'MoskaGame', inplace=False) -> 'MoskaGameState':
        """ Modify the game instance according to the action.
        """
        #gs : MoskaGameState = game.game_state_class.from_game(game, copy = inplace)
        # Remove the cards from the player's hand
        for card in self.cards:
            game.player_full_cards[self.pid].remove(card)
            if card in game.player_public_cards[self.pid]:
                game.player_public_cards[self.pid].remove(card)
        # Add the cards to the cards_to_kill
        game.cards_to_kill += self.cards
        return game.game_state_class.from_game(game, copy = False)
    
class AttackInitial(AttackMove):
    
    def is_move_id_legal(self, game: 'MoskaGame') -> bool:
        """ The initial attack move can be played if the board is empty.
        """
        if game.target_is_kopling:
            return False
        gs : MoskaGameState = game.get_current_state()
        is_initiating = gs.player_is_initiating(self.pid)
        return is_initiating and self._board_is_empty(game)
    
    def _attacker_and_target_are_different(self, game : 'MoskaGame'):
        """ Check if the attacker and target are the same player.
        """
        return self.pid != self.target_pid
    
    def _cards_are_single_or_multiple(self, game : 'MoskaGame'):
        """ In an initial attack, the cards can be either single cards,
        or multiple cards of the same rank.
        """
        c = Counter([c.rank for c in self.cards])
        return len(self.cards) == 1 or all((count >= 2 for count in c.values()))
    
    def check_action_is_legal(self, game : 'MoskaGame') -> Tuple[bool, str]:
        msg = ""
        if not self._board_is_empty(game):
            msg += "The game is already initiated."
        if not self._cards_are_single_or_multiple(game):
            msg += "The cards are not single cards, or multiple cards of the same rank."
        if not self._pids_are_valid(game):
            msg += "The player ids are not valid."
        if not self._attack_cards_are_available(game):
            msg += "The attack cards are not available in the player's hand."
        if not self._target_pid_is_target(game):
            msg += "The target_pid is not the target player."
        is_legal = True if not msg else False
        return is_legal, msg
            

class AttackSelf(AttackMove):
    
    def is_move_id_legal(self, game: 'MoskaGame') -> bool:
        """ Target can attack self, at all times, if the board is not empty.
        """
        if game.target_is_kopling:
            return False
        #gs : MoskaGameState = game.get_current_state()
        return self.pid == game.target_pid and not self._board_is_empty(game) and len(game.deck) > 0
    
    def _attacker_and_target_are_same(self, game):
        """ Check if the attacker and target are the same player.
        """
        return self.pid == self.target_pid
    
    def check_action_is_legal(self, game):
        msg = ""
        if not self._pids_are_valid(game):
            msg += "The player ids are not valid."
        if not self._attack_cards_are_available(game):
            msg += "The attack cards are not available in the player's hand."
        if not self._target_pid_is_target(game):
            msg += "The target_pid is not the target player."
        if not self._ranks_are_in_table(game):
            msg += "The card ranks can not be played, as they are not in cards_to_kill or killed_cards."
        if not self._attacker_and_target_are_same(game):
            msg += "The attacker and target are not the same player."
        is_legal = True if not msg else False
        return is_legal, msg
        
        
class AttackOther(AttackMove):
    def is_move_id_legal(self, game: 'MoskaGame') -> bool:
        """ The player can attack the target, if the board is not empty, and fits more than 0.
        """
        if game.target_is_kopling:
            return False
        gs : MoskaGameState = game.get_current_state()
        if not gs.bout_is_initiated():
            return False
        if self._board_is_empty(game):
            return False
        if self.pid == game.target_pid:
            return False
        if game.num_fits_to_table == 0:
            return False
        return True
    
    def check_action_is_legal(self, game : 'MoskaGame') -> Tuple[bool, str]:
        return super().check_action_is_legal(game)
    
class KillFromHand(MoskaAction):
    __slots__ = ["pid", "move_id", "kill_mapping"]
    def __init__(self, pid : int, move_id : str, kill_mapping : dict = {}):
        super().__init__(pid, move_id, kill_mapping)
        self.kill_mapping = kill_mapping
        
    def is_move_id_legal(self, game: 'MoskaGame') -> bool:
        """ The KillFromHand move can be played if there are cards to kill, and the player is the target.
        """
        return self._is_target(game) and len(game.cards_to_kill) > 0
    
    def _killing_cards_are_in_hand(self, game: 'MoskaGame'):
        """ Check if the killing cards are in the player's hand.
        """
        return all(card in game.player_full_cards[self.pid] for card in self.kill_mapping.keys())
    
    def _kill_mapping_is_valid(self, game: 'MoskaGame'):
        """ Check if the kill mapping is valid.
        """
        if any((not check_can_kill_card(c, v, game.trump_suit) for c, v in self.kill_mapping.items())):
            return False
        return True
    
    def check_action_is_legal(self, game: 'MoskaGame'):
        msg = ""
        if not self._is_target(game):
            msg += "The player is not the target."
        if not self._killing_cards_are_in_hand(game):
            msg += "The killing cards are not in the player's hand."
        if not self._kill_mapping_is_valid(game):
            msg += f"The kill mapping ({self.kill_mapping}) is not valid."
        if game.target_is_kopling:
            if len(self.kill_mapping.keys()) > 1:
                msg += "The target is kopling, and can only kill one card."
            if len(self.kill_mapping.keys()) == 0:
                msg += "The target is kopling, and must kill a card."
            k = list(self.kill_mapping.keys())[0]
            #print(f"Kopled card is: {k}, kopled: {k.kopled}")
            if len(self.kill_mapping.keys()) > 0 and not list(self.kill_mapping.keys())[0].kopled:
                msg += "The target is kopling, and must use the kopled card to kill."
        is_legal = True if not msg else False
        return is_legal, msg
    
    def modify_game(self, game : 'MoskaGame', inplace=False) -> 'MoskaGameState':
        """ Modify the game instance according to the action.
        """
        cards_from_hand = list(self.kill_mapping.keys())
        cards_on_table = list(self.kill_mapping.values())
        # Remove the cards from the player's hand
        #game.player_full_cards[self.pid] = [card for card in game.player_full_cards[self.pid] if card not in cards_from_hand]
        for card in cards_from_hand:
            game.player_full_cards[self.pid].remove(card)
            if card in game.player_public_cards[self.pid]:
                game.player_public_cards[self.pid].remove(card)
        # Remove the killed cards from the table
        for card in cards_on_table:
            game.cards_to_kill.remove(card)
        # Add the cards from hand, and the killed cards to the killed cards
        cards_to_killed_cards = []
        for hand_card, table_card in self.kill_mapping.items():
            cards_to_killed_cards.append(table_card)
            cards_to_killed_cards.append(hand_card)
        game.killed_cards += cards_to_killed_cards
        if game.target_is_kopling and inplace:
            for card in game.killed_cards:
                card.kopled = False
            for card in game.cards_to_kill:
                card.kopled = False
        game.target_is_kopling = False
        game_state = game.game_state_class.from_game(game, copy = False)
        return game_state
        
        
    
        