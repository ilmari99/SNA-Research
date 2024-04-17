import itertools
import os
import sys
from RLFramework import GameState
from RLFramework.Action import Action
from RLFramework.Player import Player
from RLFramework.Result import Result
import numpy as np
from RLFramework import Game
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
from RLFramework.utils import TFLiteModel

from MoskaResult import MoskaResult
from MoskaGameState import MoskaGameState
from MoskaPlayer import MoskaPlayer
from MoskaAction import MoskaAction, VALID_MOVE_IDS, get_moska_action
from Card import Card, REFERENCE_DECK

from utils import get_initial_attacks, get_all_matchings, check_can_kill_card


class MoskaGame(Game):
    
    def __init__(self, logger_args : dict = None, timeout : int = 10, model_paths = [], *args, **kwargs):
        super().__init__(MoskaGameState,custom_result_class=MoskaResult, logger_args=logger_args, timeout=timeout, *args, **kwargs)
        self.deck : List[Card] = []
        self.trump_card : Card = None
        self.players : List[MoskaPlayer] = []
        self.cards_to_kill : List[Card] = []
        self.killed_cards : List[Card] = []
        self.discarded_cards : List[Card] = []
        self.player_full_cards : List[List[Card]] = []
        self.player_public_cards : List[List[Card]] = []
        self.target_pid : int = 0
        self.current_pid : int = 0
        self.models : Dict[str, TFLiteModel] = {}
        self.ready_players : List[bool] = []
        self.target_is_kopling : bool = False
        self.set_models(model_paths)


    def get_all_possible_actions(self) -> List['MoskaAction']:
        """ Return a list with all possible actions, that the current player can make.
        """
        if self.players[self.current_pid] in self.get_finished_players():
            return []
        actions = []
        gs : MoskaGameState = self.get_current_state()
        current_player = self.players[self.current_pid]
        max_moves_to_consider = current_player.max_moves_to_consider
        for move_id in VALID_MOVE_IDS:
            
            base_action = get_moska_action(self.current_pid, move_id)
            if not base_action.is_move_id_legal(self):
                current_player.logger.debug(f"Move {move_id} is not legal.")
                continue
            
            if move_id == "AttackInitial":
                possible_plays = get_initial_attacks(self.player_full_cards[self.current_pid], self.num_fits_to_table, max_moves=max_moves_to_consider - len(actions))
                actions += [get_moska_action(self.current_pid, move_id, target_pid = self.target_pid, cards = list(play)) for play in possible_plays]
                current_player.logger.debug(f"Player {self.current_pid} has {len(possible_plays)} possible initial attacks.")
                
            elif move_id == "EndBout":
                # In an EndBout move we can either pick all cards from the table, or only cards_to_kill
                actions.append(get_moska_action(self.current_pid, move_id, self.cards_to_kill.copy()))
                if len(self.killed_cards) > 0:
                    actions.append(get_moska_action(self.current_pid, move_id, self.killed_cards.copy() + self.cards_to_kill.copy()))
                    
            elif move_id == "Skip":
                actions.append(get_moska_action(self.current_pid, move_id))
                
            elif move_id == "AttackOther":
                playable_values = self.players[self.current_pid].get_playable_ranks_from_hand()
                playable_cards = [card for card in self.player_full_cards[self.current_pid] if card.rank in playable_values]
                play_iterables = []
                for i in range(1, min(len(playable_cards) + 1, self.num_fits_to_table + 1)):
                    play_iterables.append(itertools.combinations(playable_cards, i))
                plays = itertools.chain.from_iterable(play_iterables)
                num_attacks = 0
                for play in plays:
                    actions.append(get_moska_action(self.current_pid, move_id, target_pid = self.target_pid, cards = list(play)))
                    num_attacks += 1
                    if len(actions) >= max_moves_to_consider:
                        self.logger.debug(f"Maximum number of moves to consider reached.")
                        break
                current_player.logger.debug(f"Player {self.current_pid} had {num_attacks} possible attacks to target.")
                
            elif move_id == "AttackSelf":
                playable_values = self.players[self.current_pid].get_playable_ranks_from_hand()
                playable_cards = [card for card in self.player_full_cards[self.current_pid] if card.rank in playable_values]
                play_iterables = []
                for i in range(1, len(playable_cards) + 1):
                    play_iterables.append(itertools.combinations(playable_cards, i))
                plays = itertools.chain.from_iterable(play_iterables)
                num_attacks = 0
                for play in plays:
                    actions.append(get_moska_action(self.current_pid, move_id, target_pid = self.current_pid, cards = list(play)))
                    num_attacks += 1
                    if len(actions) >= max_moves_to_consider:
                        self.logger.debug(f"Maximum number of moves to consider reached.")
                        break
                current_player.logger.debug(f"Player {self.current_pid} has {num_attacks} possible attacks to self.")
                
            elif move_id == "KillFromHand":
                if self.target_is_kopling:
                    # If the target is kopling, then we must kill with the kopled card
                    possible_killings = get_all_matchings([c for c in self.player_full_cards[self.current_pid] if c.kopled == True],
                                                          self.cards_to_kill,
                                                          trump=self.trump_card.suit,
                                                          max_moves=max_moves_to_consider - len(actions)
                    )
                    # Since the assignments are based on the indices of from and to,
                    # we must convert them to indices in our hand. The index is always the last card.
                    last_idx = len(self.player_full_cards[self.current_pid]) - 1
                    for possible_killing in possible_killings:
                        possible_killing.inds = (last_idx, possible_killing.inds[1])
                        possible_killing._hand_inds = (last_idx,)
                else:
                    possible_killings = get_all_matchings(self.player_full_cards[self.current_pid],
                                                        self.cards_to_kill,
                                                        trump=self.trump_card.suit,
                                                        max_moves=max_moves_to_consider - len(actions)
                    )
                # Elements of possble_killings
                # are Assignments (at utils.py)
                # They have two attrs: _hand_inds, _table_inds
                # lets convert them to cards
                possible_killings_as_cards = []
                for assignment in possible_killings:
                    hand_inds = assignment._hand_inds
                    table_inds = assignment._table_inds
                    cards_from_hand = [self.player_full_cards[self.current_pid][i] for i in hand_inds]
                    cards_from_table = [self.cards_to_kill[i] for i in table_inds]
                    possible_killings_as_cards.append({hc : tc for hc, tc in zip(cards_from_hand, cards_from_table)})
                actions += [get_moska_action(self.current_pid, move_id, kill_mapping = kill_mapping) for kill_mapping in possible_killings_as_cards]
                current_player.logger.debug(f"Player {self.current_pid} has {len(possible_killings_as_cards)} possible killings from hand.")
                
            elif move_id == "KillFromDeck":
                action = get_moska_action(self.current_pid, move_id)
                actions.append(action)
                current_player.logger.debug(f"Player {self.current_pid} can kill from deck.")
            else:
                raise ValueError(f"Invalid move_id: {move_id}")
            
            if len(actions) >= max_moves_to_consider:
                break
            
        #print(f"Player {self.current_pid} has {actions} possible actions.", flush=True)
        return actions
    
    def calculate_reward(self, pid : int, new_state : 'GameState'):
        """ Calculate the reward for the player that made the move.
        """
        finished_players = [i for i in range(len(self.players)) if self.check_is_player_finished(i, new_state)]
        player = self.players[pid]
        if pid in finished_players and not player.has_received_reward and len(finished_players) < len(self.players):
            player.has_received_reward = True
            return 1
        return 0
    
    def __repr__(self) -> str:
        """ Print the game state.
        Number of cards in deck: {len(self.deck)}
        Trump card: {self.trump_card}
        Player1 (4)*: self.public_cards[0]
        Player2 (6): self.public_cards[1]
        player3 (6): self.public_cards[2]...
        etc.
        Cards to kill: {self.cards_to_kill}
        Killed cards: {self.killed_cards}

        """
        s = f"Number of cards in deck: {len(self.deck)}\n"
        s += f"Trump card: {self.trump_card}\n"
        for i, player in enumerate(self.players):
            s += f"Player{i} ({len(self.player_full_cards[i])})"
            if i == self.target_pid:
                s += "*"
            if "human" in player.__class__.__name__.lower():
                s += f": {self.player_full_cards[i]}\n"
            else:
                s += f": {self.player_public_cards[i]}\n"
        s += f"Cards to kill: {self.cards_to_kill}\n"
        s += f"Killed cards: {self.killed_cards}\n"
        return s

    def get_players_condition(self, condition = None) -> List[MoskaPlayer]:
        """ Get the players that satisfy the condition.
        """
        if condition is None:
            return self.players
        return [player for player in self.players if condition(player)]
    
    def environment_action(self, game_state : 'MoskaGameState') -> 'MoskaGameState':
        """ In Moska,
        The environment_action is called after each move.
        The environment action fills the players hands, if they have less than 6 cards.
        It also decides who is the next player, if it is not decided in the game_state.
        
        In a koplaus situation, the game_state should have 'is_kopling' set to True.
        The environment then picks a card from the deck, and sets it to game_state.kopled_card IF the
        card can kill any card on the table. If not, the kopled card is added to cards_to_kill.
        """
        if game_state.target_is_kopling:
            # If the target is kopling, then we must pick a card from the deck
            kopled_card = game_state.deck.pop(0)
            kopled_card.kopled = True
            can_kill_card = any(check_can_kill_card(kopled_card, card, game_state.trump_card.suit) for card in game_state.cards_to_kill)
            # If the kopled card can kill any card on the table
            # Then we set it's kopled attribute to True, and add it to the target's hand. The current_pid remains.
            if can_kill_card:
                game_state.player_full_cards[game_state.target_pid].append(kopled_card)
            # Else, we add the kopled card to the cards_to_kill, set the current_pid to -1, and set the target_is_kopling to False
            else:
                game_state.cards_to_kill.append(kopled_card)
                game_state.current_pid = -1
                game_state.target_is_kopling = False
                
        self.fill_hands(game_state)
        if game_state.current_pid == -1:
            #print(self.players)
            game_state.current_pid = self.select_turn(self.players, game_state.previous_turns)
            game_state.previous_turns.append(game_state.current_pid)
        return game_state
        
    def get_finished_players(self) -> List[int]:
        """ Get the finished players.
        """
        gs : MoskaGameState = self.get_current_state()
        return gs.get_finished_players()
    
    def get_ranks_on_table(self) -> Set[int]:
        """ Get the ranks of the cards on the table.
        """
        return set([card.rank for card in self.cards_to_kill + self.killed_cards])
    
    def check_target_can_end_bout(self) -> bool:
        """ The target can end the bout,
        if there is atleast 1 card in the table, and
        all other players are ready.
        """
        return (len(self.cards_to_kill + self.killed_cards) > 0 and
                all(player.ready for player in self.players if player.pid != self.target_pid))
        
    def check_target_can_kill_from_deck(self) -> bool:
        return len(self.deck) > 0 and not any(card.kopled for card in self.cards_to_kill)
    

    def check_target_must_end_bout(self) -> bool:
        """ The target player must end the bout, if:
        - There are cards on the table, and
        - They have no playable cards, and 
        - All players are ready, and
        - The player can't play from deck (there is no deck left or there is a kopled card already)
        """
        if not self.check_target_can_end_bout():
            return False
        target = self.players[self.target_pid]
        kill_mapping = target.get_possible_kill_mapping()
        can_kill = any(kill_mapping.values())
        # If the target can kill
        if can_kill:
            return False
        has_playble_ranks = bool(target.get_playable_ranks_from_hand())
        # If the target has playable ranks (to play to self)
        if has_playble_ranks:
            return False
        can_kill_from_deck = self.check_target_can_kill_from_deck()
        # If the target can kill from deck
        if can_kill_from_deck:
            return False
        return True
    
    def step(self, action: Action, real_move=True) -> MoskaGameState:
        # Super step automatically sets the next player,
        # and appends them to the previous turns
        self.logger.debug(f"Stepping (Real={real_move}) with action {action}")
        curr_board_len = len(self.cards_to_kill + self.killed_cards)
        current_pid = self.current_pid
        # After the step, the current_pid is -1, since we do not know the next player
        state : MoskaGameState = super().step(action, real_move)
        state.ready_players[current_pid] = True
        if len(self.cards_to_kill + self.killed_cards) != curr_board_len:
            # If the board state changes, then set all players (who are not finished) to not ready
            finished_players = state.get_finished_players()
            for i in range(len(self.players)):
                if i not in finished_players:
                    state.ready_players[i] = False
        return state
    
    def fill_hands(self, game_state : MoskaGameState = None, inplace = True) -> None:
        """ Fill the hands of the players.
        """
        if game_state is None:
            game_state = self.get_current_state()
        if len(game_state.deck) == 0:
            return
        # Find a player with less than 6 cards
        player_with_missing_cards = None
        for i, player_full_cards in enumerate(game_state.player_full_cards):
            if len(player_full_cards) < 6:
                player_with_missing_cards = i
                break
        # The else block is executed if the loop completes without breaking
        else:
            return
        player = self.players[player_with_missing_cards]
        if player.pid == game_state.target_pid:
            return
            
        if player_with_missing_cards is None:
            return
        # Fill the hand of the player
        pick_n_cards = min(6 - len(self.players[player_with_missing_cards].hand), len(self.deck))
        self.players[player_with_missing_cards].logger.debug(f"Player {player_with_missing_cards} lifted {pick_n_cards} cards from deck.")
        game_state.player_full_cards[player_with_missing_cards] += [self.deck.pop(0) for _ in range(pick_n_cards)]
        if inplace:
            self.restore_game(game_state)
        return
    
    
    def restore_game(self, game_state: MoskaGameState) -> None:
        """ Given a gamestate, restore the game to that state.
        In Moska, this means restoring:
        - Deck,
        - Trump card,
        - Players' cards,
        - Known cards in each hand,
        - Cards to kill,
        - Killed cards,
        - Discarded cards,
        - Current player,
        - Target player,
        - Which players are ready (have played Skip once after the board last changed),        
        """
        self.deck = game_state.deck
        self.trump_card = game_state.trump_card
        self.player_full_cards = game_state.player_full_cards
        self.player_public_cards = game_state.player_public_cards
        self.ready_players = game_state.ready_players
        self.previous_turns = game_state.previous_turns
        self.finishing_order = game_state.finishing_order
        self.player_scores = game_state.player_scores
        
        self.cards_to_kill = game_state.cards_to_kill
        self.killed_cards = game_state.killed_cards
        self.discarded_cards = game_state.discarded_cards
        self.current_pid = game_state.current_pid
        self.target_pid = game_state.target_pid
        self.target_is_kopling = game_state.target_is_kopling
        #self.logger.debug(f"Game restored to state:\n{game_state}")
        
    def reset(self) -> None:
        """ Reset the game to the initial state.
        """
        self.deck = []
        self.trump_card = None
        self.players = []
        self.cards_to_kill = []
        self.killed_cards = []
        self.discarded_cards = []
        self.player_full_cards = []
        self.player_public_cards = []
        self.target_pid = 0
        self.ready_players = []
        self.game_states = []
        self.previous_turns = []
        self.current_pid = 0
        self.finishing_order = []
        self.player_scores = []
        self.total_num_played_turns = 0
        self.current_player_name = ""
        self.target_is_kopling : bool = False
    
    def initialize_game(self, players: List[Player]) -> None:
        """ Initialize the MoskaGame instance.
        Here, we create the deck, shuffle it, and deal the cards to the players.
        We also select the trump card/suit.
        """
        assert len(players) >= 2 and len(players) <= 8, "Moska is a game for 2-8 players."
        self.players = players
        self.deck = REFERENCE_DECK.copy()
        np.random.shuffle(self.deck)
        player_full_cards = []
        for i, player in enumerate(players):
            # Remove 6 cards from the top of the deck
            player_full_cards.append([self.deck.pop(0) for _ in range(6)])
        self.player_full_cards = player_full_cards
        # The next card is the trump card
        self.trump_card = self.deck.pop(0).copy()
        # Place the trump card at the bottom of the deck
        self.deck.append(self.trump_card)
        self.ready_players = [False for _ in players]
        self.player_public_cards = [[] for _ in players]
        
    def check_is_player_finished(self, pid : int, game_state : MoskaGameState) -> bool:
        """ A player is finished, if they have no cards left in their hand,
        and the deck is empty.
        Also, if the player is the target, then additionally,
        all the cards on the table have to have been killed (cards_to_kill is empty)
        """
        player_hand = game_state.player_full_cards[pid]

        # If the player is the last player, then they are finished
        if len(game_state.get_finished_players()) == len(self.players) - 1:
            return True
        
        if pid == game_state.target_pid:
            return (len(player_hand) == 0 and
                    len(self.deck) == 0 and
                    len(game_state.cards_to_kill) == 0)
        elif (len(player_hand) == 0 and len(self.deck) == 0):
            return True
        return False
        
    def select_turn(self, players: List[Player], previous_turns: List[int]) -> int:
        """ In Moska, the turns are mostly based on speed.
        But in this implementation, we will select the next player randomly.
        """
        if self.current_pid == -1:
            # If all players are ready, return the target player
            ready_players = [i for i in range(len(players)) if self.ready_players[i]]
            if len(ready_players) == len(players):
                return self.target_pid
            return np.random.choice([i for i in range(len(players)) if i not in ready_players])
        return previous_turns[-1]
    
    def get_model(self, model_name : str) -> TFLiteModel:
        """ Get the model with the given name.
        """
        model_name = os.path.abspath(model_name)
        try:
            return self.models[model_name]
        except KeyError:
            raise ValueError(f"Model with name {model_name} not found. Available models: {list(self.models.keys())}")
        
    def set_models(self, model_paths : List[str]) -> None:
        """ Set the models to the given paths.
        """
        #print(f"Setting models: {model_paths}")
        self.model_paths = model_paths
        self.models = {path: TFLiteModel(path) for path in model_paths}
        #print(f"Models set: {self.models}")
        
    @property
    def trump_suit(self) -> str:
        return self.trump_card.suit
    
    @property
    def num_fits_to_table(self) -> int:
        """ Return the number of cards that fit the cards to kill.
        """
        return len(self.players[self.target_pid].hand) - len(self.cards_to_kill)