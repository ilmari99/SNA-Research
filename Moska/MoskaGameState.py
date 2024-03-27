import copy
from typing import List, SupportsFloat
import pickle
import warnings
# named tuple class
from collections import namedtuple

import numpy as np
from MoskaGame import MoskaGame
from MoskaPlayer import MoskaPlayer
from RLFramework.GameState import GameState
from Card import REFERENCE_DECK, Card

class MoskaGameState(GameState):
    """ A class representing the state of the game Moska.
    """
    def __init__(self, state_json : dict):
        """ Initialize the game state.
        """
        self.deck : List[Card] = state_json["deck"]
        self.trump_card : Card = state_json["trump_card"]
        self.cards_to_kill : List[Card] = state_json["cards_to_kill"]
        self.killed_cards : List[Card] = state_json["killed_cards"]
        self.discarded_cards : List[Card] = state_json["discarded_cards"]
        self.current_pid : int = state_json["current_pid"]
        self.target_pid : int = state_json["target_pid"]
        self.players_ready : List[bool] = state_json["players_ready"]
        self.player_full_cards : List[List[Card]] = state_json["player_full_cards"]
        self.player_public_cards : List[List[Card]] = state_json["player_public_cards"]
        
        super().__init__(state_json)
        
    def bout_is_initiated(self) -> bool:
        """ Check if the bout is initiated.
        """
        return len(self.cards_to_kill + self.killed_cards) > 0
    
    def player_is_initiating(self, pid) -> bool:
        """ Check if the current player is initiating the bout.
        This is true, if the table is empty,
        and the player with the next pid is the target.
        Return True, if the current_pid is equal to the pid before
        the target.
        """
        remaining_player_pids = [i for i in range(len(self.players_ready)) if not self.check_is_player_finished(i)]
        num_remaining_players = len(remaining_player_pids)
        target_pid = self.target_pid
        target_pid_fake = remaining_player_pids.index(target_pid)
        # Return true, if the current player's
        # pid is the one before the target, accounting
        # for already finished players
        return pid == remaining_player_pids[(target_pid_fake - 1) % num_remaining_players]
    
    def target_can_play_from_deck(self) -> bool:
        if (any(card.kopled for card in self.cards_to_kill) or
            any(card.kopled for card in self.player_full_cards[self.current_pid])):
            return False
        if len(self.deck) == 0:
            return False
        return True
        
    
    def get_finished_players(self) -> List[int]:
        """ Get player pids who have finished in the state.
        Finishing means no cards, and no deck left.
        For the target, additionally, no cards to kill.
        """
        if len(self.deck) > 0:
            return []
        
        finished_players = []
        for pid in range(len(self.players_ready)):
            if self.check_is_player_finished(pid):
                finished_players.append(pid)
        return finished_players
    
    def check_is_player_finished(self, pid : int) -> bool:
        """ Check if the player is finished.
        """
        if len(self.player_full_cards[pid]) == 0 and len(self.deck) == 0:
            if pid == self.target_pid:
                return len(self.cards_to_kill) == 0
            return True
        return False
    
    
    def game_to_state_json(cls, game : 'MoskaGame', player : 'MoskaPlayer'):
        """ Create a json describing the game state.
        The json describes the game state with perfect information,
        so the 'player' argument is ignored.
        """
        if player is not None:
            warnings.warn("The player argument is ignored in MoskaGameState.game_to_state_json")
        state_json = {
            "deck" : game.deck,
            "trump_card" : game.trump_card,
            "cards_to_kill" : game.cards_to_kill,
            "killed_cards" : game.killed_cards,
            "discarded_cards" : game.discarded_cards,
            "current_pid" : game.current_pid,
            "target_pid" : game.target_pid,
            "players_ready" : game.ready_players,
            "player_full_cards" : [player_full_cards for player_full_cards in game.player_full_cards],
            "player_public_cards" : [public_cards for public_cards in game.player_public_cards],
        }
        return state_json
    
    def deepcopy(self):
        """ Custom copy the game state by deep copying the lists.
        """
        for k,v in self.state_json.items():
            if isinstance(v, list):
                self.state_json[k] = copy.deepcopy(v)
        return MoskaGameState(self.state_json)
    
    def cards_to_vector(self, cards : List[Card]) -> List[SupportsFloat]:
        """ Convert a list of cards to a vector.
        """
        card_vector = np.zeros(len(REFERENCE_DECK))
        for card in cards:
            ind = REFERENCE_DECK.index(card)
            card_vector[ind] = 1
        return card_vector.tolist()
    
    def to_vector(self, perspective_pid = None) -> List[SupportsFloat]:
        """ Convert the state to a vector.
        In the state, we first include all the meta data,
        such as num cards in deck, trump suit, which player is the target, etc.
        Then, we include the card data, where
        each set of cards is represented as a one-hot vector, where
        1 means the card is in the set, and 0 means it is not.
        """
        if perspective_pid is None:
            perspective_pid = self.perspective_pid
        
        deck_len = len(self.deck)
        trump_suit_map = {"C" : 0, "D" : 1, "H" : 2, "S" : 3}
        trump_suit = trump_suit_map[self.trump_card.suit]
        target_pid = self.target_pid
        current_pid = self.current_pid
        is_kopled_card_on_table = 1 if any((c.kopled for c in self.cards_to_kill)) else 0
        player_hand_lens = [len(player.hand) for player in self.players]
        player_is_ready = [player.ready for player in self.players]
        # Now we know the meta data, we can create the card data
        meta_data = [perspective_pid, deck_len, trump_suit, target_pid, current_pid, is_kopled_card_on_table]
        meta_data += player_hand_lens + player_is_ready
        card_data = []
        # As card data we have:
        # - The players own cards (full info),
        # - The åublic cards of each player (including self),
        # - The cards to kill (full info),
        # - The killed cards (full info),
        # - The discarded cards (full info)
        my_cards = self.cards_to_vector(self.players[perspective_pid].hand)
        cards_to_kill = self.cards_to_vector(self.cards_to_kill)
        killed_cards = self.cards_to_vector(self.killed_cards)
        discarded_cards = self.cards_to_vector(self.discarded_cards)
        public_cards = [self.cards_to_vector(player.public_cards) for player in self.players]
    
        card_data = my_cards + cards_to_kill + killed_cards + discarded_cards
        card_data += np.concatenate(public_cards).tolist()
        
        return meta_data + card_data