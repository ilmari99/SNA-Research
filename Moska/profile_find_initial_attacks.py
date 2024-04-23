from collections import Counter
import itertools
import random
from typing import List
import tracemalloc


class Card:
    """ A class representing a card.
    """
    __slots__ = ['suit', 'rank', 'kopled']
    
    def __init__(self, suit : str, rank : int, kopled = False):
        self.suit = suit
        self.rank = rank
        self.kopled = kopled
        
    def __eq__(self, other) -> bool:
        return self.suit == other.suit and self.rank == other.rank
    
    def __repr__(self) -> str:
        s = f"{self.suit}{self.rank}"
        s += "\'" if self.kopled else ""
        return s
    
    def __hash__(self) -> int:
        return hash((self.suit, self.rank))
    
    def __lt__(self, other) -> bool:
        return self.rank < other.rank
    
    def __eq__(self, other) -> bool:
        return self.rank == other.rank and self.suit == other.suit
    
    def __copy__(self):
        return Card(self.suit, self.rank, self.kopled)
    
    def copy(self):
        return self.__copy__()
    
    def __deepcopy__(self, memo=None):
        return Card(self.suit, self.rank, self.kopled)
    
    def deepcopy(self):
        return self.__deepcopy__()


def get_initial_attacks(cards : List[Card], fits : int, max_moves : int = 1000) -> List[List[Card]]:
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
    def get_play_combinations(play,visited = set(), started_with = set(), max_moves = 1000):
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
            if val in visited:
                continue
            for len_play, plays in plays.items():
                if len_play + len(play) > fits:
                    continue
                for p in plays:
                    if p in started_with:
                        continue
                    visited.add(val)
                    old_visited = visited.copy()
                    combs = get_play_combinations(tuple(list(play) + list(p)),visited,started_with, max_moves - len(combined_plays))
                    visited = old_visited
                    combined_plays += combs
                    if len(combined_plays) > max_moves:
                        break
        return combined_plays
    
    started_with = set()
    # Now we have a function that can return all combinations of plays, that do not share values, from some starting play
    for val, plays in cards_set_combinations.items():
        for len_play, plays in plays.items():
            for play in plays:
                started_with.add(tuple(play))
                combs = get_play_combinations(tuple(play),started_with=started_with,max_moves=max_moves - len(legal_plays))
                legal_plays += [tuple(c) for c in combs]
        if len(legal_plays) > max_moves:
            break
    legal_plays = list(set(legal_plays))
    legal_plays = [list(play) for play in legal_plays]
    return legal_plays

if __name__ == "__main__":
    tracemalloc.start()
    deck = [Card(suit, value)for suit, value in itertools.product(["C","D","H","S"], range(2,15))]
    random.shuffle(deck)
    # Profile the memory use if get_initial_attacks
    # Take 20 random cards from the deck
    cards = deck[:20]
    fits = 10
    max_moves = 1000
    snapshot1 = tracemalloc.take_snapshot()
    attacks = get_initial_attacks(cards, fits, max_moves)
    snapshot2 = tracemalloc.take_snapshot()
    print(f"Found {len(attacks)} attacks")
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("[ Top 10 differences ]")
    for stat in top_stats[:10]:
        print(stat)
        
    