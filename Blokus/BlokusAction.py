from typing import Tuple, TYPE_CHECKING
from RLFramework import Action
from RLFramework.Game import Game
from RLFramework.GameState import GameState
import numpy as np

from BlokusPieces import BLOKUS_PIECE_MAP
if TYPE_CHECKING:
    from BlokusGame import BlokusGame

class BlokusAction(Action):
    """ A class representing an action in the game Blokus.
    """
    def __init__(self, piece_id : int, x : int, y : int, rotation : int, flip : bool):
        self.piece_id = piece_id
        self.x = x
        self.y = y
        self.rotation = rotation
        self.flip = flip
    
    def __eq__(self, other: 'BlokusAction') -> bool:
        """ Two actions are equal, if they take the same coordinates
        """
        if not isinstance(other, BlokusAction):
            return False
        other_coords = other.get_piece_coordinates()
        self_coords = self.get_piece_coordinates()
        # Check that the coordinates are the same, regardless of order
        return set(other_coords) == set(self_coords)
    
    def modify_game(self, game: 'BlokusGame', inplace: bool = False) -> GameState:
        """ Place the piece on the board.
        We set 0 values to -1, and 1 values to the player id.
        We then flip and rotate the piece according to the action.
        Then we place the piece on the board, s.t. the left upper corner of the piece is at the given x, y coordinates.
        """
        if self.piece_id == -1:
            game.current_pid = (game.current_pid + 1) % len(game.players)
            return game.game_state_class.from_game(game, copy = False)
        # Get the piece
        piece = BLOKUS_PIECE_MAP[self.piece_id]
        piece = np.rot90(piece, k=self.rotation)
        if self.flip:
            piece = np.flip(piece, axis=0)
        # Modify the values of the piece
        piece = np.where(piece == 0, -1, game.current_pid)
        # Place the piece on the board, so that only the self.pid values are changed on the board
        board = np.array(game.board)
        piece_mask = piece != -1
        # Where piece_mask is True, we set the values of the board to the values of the piece
        board[self.x:self.x+piece.shape[0], self.y:self.y+piece.shape[1]][piece_mask] = piece[piece_mask]
        # Update the board
        game.board = board.tolist()
        # Remove the piece from the player's remaining pieces
        game.player_remaining_pieces[game.current_pid].remove(self.piece_id)
        # Update the current player
        #print(game.players)
        #print(game.get_current_state())
        game.current_pid = (game.current_pid + 1) % len(game.players)
        # Return the new game state
        return game.game_state_class.from_game(game, copy = False)
    
    def is_grid_inside_board(self, grid, board_size):
        """ Check if the grid is inside the board.
        """
        return grid[0] >= 0 and grid[0] < board_size[0] and grid[1] >= 0 and grid[1] < board_size[1]
    
    def __repr__(self) -> str:
        if self.piece_id == -1:
            return "BlokusAction(None)"
        piece_mat = BLOKUS_PIECE_MAP[self.piece_id]
        return f"BlokusAction({piece_mat}, x: {self.x}, y: {self.y})"
    
    def get_piece_coordinates(self):
        """ Return a list of coordinates of the piece.
        """
        piece = BLOKUS_PIECE_MAP[self.piece_id]
        piece = np.rot90(piece, k=self.rotation)
        if self.flip:
            piece = np.flip(piece, axis=0)
        piece_grids = np.where(piece != 0)
        piece_grids = [(self.x + piece_grids[0][i], self.y + piece_grids[1][i]) for i in range(len(piece_grids[0]))]
        return piece_grids
    
    def find_num_connected_pieces(self, board, pid, x, y):
        """ Find the number of connected pieces, starting from the grid (x, y).
        """
        stack = [(x, y)]
        visited = []
        num_connected_pieces = 0
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.append((x, y))
            num_connected_pieces += 1
            # Now we need to check the neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                # Check if the new coordinates are inside the board
                if not self.is_grid_inside_board((new_x, new_y), board.shape):
                    continue
                # Check if the new coordinates are the same player id
                if board[new_x, new_y] == pid:
                    stack.append((new_x, new_y))
        return num_connected_pieces
    
    def check_action_is_legal(self, game: Game):
        return self._check_action_is_legal(game)
        #return True, ""
    
    def _check_action_is_legal(self, game: 'BlokusGame') -> Tuple[bool, str]:
        """ Check if the action is legal in the given game state.
        """
        if self.piece_id == -1:
            return True, ""
        # Check if the piece is in the player's remaining pieces
        if self.piece_id not in game.player_remaining_pieces[game.current_pid]:
            return False, f"The piece with id {self.piece_id} is not in the player's remaining pieces."
        # Check if the piece is placed on the board
        piece = BLOKUS_PIECE_MAP[self.piece_id]
        piece = np.rot90(piece, k=self.rotation)
        if self.flip:
            piece = np.flip(piece, axis=0)
        piece = np.where(piece == 0, -1, game.current_pid)
        
        if ((not self.is_grid_inside_board((self.x, self.y),
                                           game.board_size)) or 
            (not self.is_grid_inside_board((self.x + piece.shape[0] - 1,
                                            self.y + piece.shape[1] - 1),
                                           game.board_size))):
            return False, f"The piece with id {self.piece_id} is placed outside the board."
        # Check that all the values in board are -1, where the piece is != -1
        board_area = np.array(game.board)[self.x:self.x+piece.shape[0], self.y:self.y+piece.shape[1]]
        if np.any((board_area != -1) & (piece != -1)):
            return False, f"The piece with id {self.piece_id} overlaps with another piece."
        
        piece_grids = np.where(piece != -1)
        piece_grids = [(self.x + piece_grids[0][i], self.y + piece_grids[1][i]) for i in range(len(piece_grids[0]))]
        
        # If the piece is the player's first piece, then one of it's grids must be in the player's corner 0:lu, 1:ru, 2:rd, 3:ld
        if np.sum(np.array(game.board) == game.current_pid) == 0:
            corner_grids = [(0, 0),
                            (0, game.board_size[1]-1),
                            (game.board_size[0]-1, game.board_size[1]-1),
                            (game.board_size[0]-1, 0)]
            for corner_grid in corner_grids:
                if corner_grid in piece_grids:
                    return True, ""
            return False, f"The selected piece is not connected to the player's corner."

        # Calculate the size of the piece
        piece_size = np.sum(piece != -1)
        # Find a coordinate, where the piece is true
        piece_begin = np.where(piece != -1)
        piece_begin = (piece_begin[0][0], piece_begin[1][0])
        piece_begin = (self.x + piece_begin[0], self.y + piece_begin[1])
        board_copy = np.copy(game.board)
        # Set the piece on the board
        piece_mask = piece != -1
        board_copy[self.x:self.x+piece.shape[0], self.y:self.y+piece.shape[1]][piece_mask] = piece[piece_mask]
        
        # Calculate the number self.pid pieces, that are connected from the piece_begin
        connected_pieces = self.find_num_connected_pieces(board_copy,
                                                          game.current_pid,
                                                          piece_begin[0],
                                                          piece_begin[1])
        board_copy = np.array(game.board)
        #print(f"Connected pieces: {connected_pieces}, piece size: {piece_size}")
        if connected_pieces != piece_size:
            return False, f"The selected piece is connected to another piece with a side."

        # Finally, check that there is atleast one own piece,
        # connected via a corner to the new piece
        # To find this, we check all the grids in the piece.
        # If any of the new pieces is connected via a corner to another one of our piece,
        # then they are connected via a corner.
        all_adjacent_grid_changes = [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1)]
        has_connected_corner = False
        for x, y in piece_grids:
            # A corner is free, if three consecutive adjacent grids are free (excluding all three on the same row/col)
            has_free_corner = False
            for start_idx in range(8 - 2):
                for end_idx in range(start_idx, start_idx + 3):
                    changes = all_adjacent_grid_changes[start_idx:end_idx+1]
                    if len(set([c[0] for c in changes])) == 1 or len(set([c[1] for c in changes])) == 1:
                        continue
                    adjacent_grids = [(x + dx, y + dy) for dx, dy in changes]
                    # Check that all the grids are inside the board
                    if not all([self.is_grid_inside_board(grid, game.board_size) for grid in adjacent_grids]):
                        continue
                    #print(adjacent_grids)
                    # Check that all the grids are not ours
                    if all([board_copy[grid] == game.current_pid for grid in adjacent_grids]):
                        continue
                    has_free_corner = True
                    break
            if not has_free_corner:
                continue
            
            # Check if the piece is connected to another piece with a corner
            corner_changes = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
            for dx, dy in corner_changes:
                new_x, new_y = x + dx, y + dy
                if new_x < 0 or new_x >= game.board_size[0] or new_y < 0 or new_y >= game.board_size[1]:
                    continue
                if board_copy[new_x, new_y] == game.current_pid and (new_x, new_y) not in piece_grids:
                    has_connected_corner = True
                    break
            if has_connected_corner:
                break
        if not has_connected_corner:
            return False, f"The selected piece is not connected to another piece with a corner."
        return True, ""
    
    def __hash__(self):
        return hash((self.piece_id, self.x, self.y, self.rotation, self.flip))

        





