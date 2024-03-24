
from collections import Counter
import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
from Card import Card


class Assignment:
    """ An assignment is a mapping from cards in the hand to cards on the table.
    Two assignments are equal if the same cards are played to the same cards, regardless of order.
    """
    def __init__(self, inds : Tuple[int]):
        self.inds = inds
        self._hand_inds = self.inds[::2]
        self._table_inds = self.inds[1::2]
    
    def __eq__(self, other):
        """ Two assignments are equal if the same cards are played to the same cards, regardless of order."""
        return set(self._hand_inds) == set(other._hand_inds) and set(self._table_inds) == set(other._table_inds)
    
    def __repr__(self):
        return f"Assignment({self.inds})"
    
    def __hash__(self):
        """ Two assignments are equal if the same cards are played to the same cards, regardless of order."""
        #return hash(frozenset(self._hand_inds)) + hash(frozenset(self._table_inds))
        return hash(tuple(sorted(list(self._hand_inds)) + sorted(list(self._table_inds))))

def get_initial_attacks(cards : List[Card], fits : int):
    single_solutions = itertools.combinations(cards,1)
    og_counter = Counter([c.rank for c in cards])
    cards = [c for c in cards if og_counter[c.rank] >= 2]


    card_sets = [[c for c in cards if c.rank == val] for val, count in og_counter.items() if count >= 2]

    legal_plays = list(single_solutions)
    # Take all combinations of atleast two cards from each set
    # Store in a dictionary, where the first key is the cards value, and the second key is the length of the play
    cards_set_combinations = {}
    for i in range(1,len(card_sets) + 1):
        value = card_sets[i - 1][0].rank
        cards_set_combinations[value] = {}
        for len_play in range(2, min(fits, len(card_sets[i - 1]))+1):
            plays = list(itertools.combinations(card_sets[i - 1],len_play))
            if len(plays) > 0:
                cards_set_combinations[value][len_play] = plays

    # Now we have a dictionary of each card value, and all the pure plays of that value
    # cards_set_combinations = {value(int) : {len_play(int) : plays(List)}}

    # Now we want to find all the combinations of those plays
    # We do this, by sort of tree searching, where we start with a value, and add all of its pure plays to the list of plays
    # Then for each of those plays, we go to all other values, and combine all of their pure plays with the current play and them to the list of plays
    # Then for each of those plays, we go to all other values, and combine all of their pure plays with the current play and them to the list of plays
    # And so on, until we have gone through all values
    def get_play_combinations(play,visited = set(), started_with = set()):
        """ Return a list of combinations from the cards_set_combinations dictionary. The input is a tuple play,
        and this function returns all plays, that can be combined with the input play, that do not share values with the input play.
        """
        play = list(play)
        if len(play) >= fits:
            return [play]
        if not visited:
            visited = set((c.rank for c in play))
        combined_plays = [play]
        for val, plays in cards_set_combinations.items():
            if val not in visited:
                for len_play, plays in plays.items():
                    if len_play + len(play) > fits:
                        continue
                    for p in plays:
                        if p in started_with:
                            continue
                        visited.add(val)
                        old_visited = visited.copy()
                        combs = get_play_combinations(tuple(list(play) + list(p)),visited,started_with)
                        visited = old_visited
                        combined_plays += combs
        return combined_plays
    
    started_with = set()
    # Now we have a function that can return all combinations of plays, that do not share values, from some starting play
    for val, plays in cards_set_combinations.items():
        for len_play, plays in plays.items():
            for play in plays:
                started_with.add(tuple(play))
                combs = get_play_combinations(tuple(play),started_with=started_with)
                legal_plays += [tuple(c) for c in combs]
    legal_plays = list(set(legal_plays))
    legal_plays = [list(play) for play in legal_plays]
    return legal_plays



def make_kill_matrix(from_ : List[Card], to : List[Card], trump : str) -> np.ndarray:
    """ Make a matrix (len(from_) x len(to)), where index (i,j) is 1
    if the card in hand at index i can kill the card on the table at index j, 0 otherwise.
    """
    matrix = np.zeros((len(from_),len(to)))
    for i, card in enumerate(from_):
        for j, killed_card in enumerate(to):
            if check_can_kill_card(card,killed_card,trump):
                matrix[i,j] = 1
    return matrix

def get_single_kills(matrix : np.ndarray) -> List[List[int]]:
    """ Return all single card assignments from the assignment matrix as a list of (row, col) tuples.
    """
    nz = np.nonzero(matrix)
    return list(zip(nz[0],nz[1]))

def _get_assignments(from_ : List[Card], to : List[Card] = [], trump : str = "", start=[], found_assignments = None, max_num : int = 1000) -> Set[Assignment]:
    """ Return a set of found Assignments, containing all possible assignments of cards from the hand to the cards to fall.
    Symmetrical assignments are considered the same when the same cards are played to the same cards, regardless of order.
    
    The assignments (partial matchings) are searched for recursively, in a depth-first manner:
    - Find all single card assignments (row-col pairs where the intersection == 1)
    - Add the assignment to found_assignments
    - Mark the vertices (row and column of the cost_matrix) as visited (0) (played_card = column, hand_card = row)
    - Repeat
    """
    matrix = None
    if not to:
        if not isinstance(from_, np.ndarray):
            raise TypeError("from_ must be a numpy array if to_ is not given")
        matrix = from_
    # Create a set of found assignments, if none is given (first call)
    if not found_assignments:
        found_assignments = set()
    # If no matrix is given, create the matrix, where 1 means that the card can fall, and 0 means it can't
    if matrix is None:
        if not trump:
            raise ValueError("trump must be given if from_ is not a cost matrix")
        matrix = make_kill_matrix(from_ = from_, to = to, trump = trump)
        
    # Find all single card assignments (row-col pairs where the intersection == 1)
    new_assignments = get_single_kills(matrix)
    
    # If there are no more assignments, or the max number of states is reached, return the found assignments
    for row, col in new_assignments:
        if len(found_assignments) >= max_num:
            return found_assignments
        og_len = len(found_assignments)
        # Named tuple with a custom __eq__ and __hash__ method
        assignment = Assignment(tuple(start + [row,col]))
        found_assignments.add(assignment)
        # If the assignment was already found, there is no need to recurse deeper, since there could only be more symmetrical assignments
        if len(found_assignments) == og_len:
            continue
        # Get the visited cards
        # The cards in hand are even (0,2...), the cards on the table are odd (1,3...)
        #hand_cards = [c for i,c in enumerate(start + [row,col]) if i % 2 == 0]
        #table_cards = [c for i,c in enumerate(start + [row,col]) if i % 2 == 1]
        hand_cards = assignment._hand_inds
        table_cards = assignment._table_inds
        # Store the values, to restore them later
        row_vals = matrix[hand_cards,:]
        col_vals = matrix[:,table_cards]
        # Mark the cards as visited (0)
        matrix[hand_cards,:] = 0
        matrix[:,table_cards] = 0
        # Find the next assignments, which adds the new assignments to the found_assignments
        _get_assignments(from_ = matrix, start = start +[row,col], found_assignments = found_assignments, max_num = max_num)
        # Restore matrix
        matrix[hand_cards,:] = row_vals
        matrix[:,table_cards] = col_vals
    return found_assignments


def get_killable_in(card : Card, in_ : List[Card], trump : str) -> List[Card]:
    """ Return a list of cards, that the input card can fall from `to` list of cards.
    """
    return [c for c in in_ if check_can_kill_card(card,c,trump)]

def get_killable_mapping(from_ : List[Card] , to : List[Card], trump : str) -> Dict[Card,List[Card]]:
    """Map each card in hand, to cards on the table, that can be fallen. Returns a dictionary of from_c : List_t pairs.
    """
    # Make a dictionary of 'card-in-hand' : List[card-on-table] pairs, to know what which cards can be fallen with which cards
    can_fall = {}
    for card in from_:
        can_fall[card] = get_killable_in(card,to,trump)
    return can_fall

def check_can_kill_card(kill_card : Card,
                        killed_card : Card,
                        trump : str) -> bool:
    """Returns true, if the kill_card, can kill the killed_card.
    The kill card can kill killed_card, if:
    - The kill card has the same suit and is greater than killed_card
    - If the kill_card is trump suit, and the killed_card is not.

    Args:
        kill_card (Card): The card played from hand
        killed_card (Card): The card on the table
        trump (str): The trump suit of the current game

    Returns:
        bool: True if kill_card can kill killed_card, false otherwise
    """
    success = False
    # Jos kortit ovat samaa maata ja pelattu kortti on suurempi
    if kill_card.suit == killed_card.suit and kill_card.rank > killed_card.rank:
        success = True
    # Jos pelattu kortti on valttia, ja kaadettava kortti ei ole valttia
    elif kill_card.suit == trump and killed_card.suit != trump:
            success = True
    return success

def check_signature(sig : Sequence, inp : Sequence) -> bool:
    """ Check whether the input sequences types match the expected sequence.
    """
    for s, i in zip(sig,inp):
        if not issubclass(type(i),s):
            print("Expected type",s,"got",type(i))
            return False
    return True