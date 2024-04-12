from typing import Tuple, TYPE_CHECKING
from RLFramework import Action
from RLFramework.Game import Game
from RLFramework.GameState import GameState
import numpy as np

from BlockusPieces import BLOCKUS_PIECE_MAP
if TYPE_CHECKING:
    from BlockusGame import BlockusGame

class BlockusAction(Action):
    """ A class representing an action in the game Blockus.
    """
    def __init__(self, piece_id : int, x : int, y : int, rotation : int, flip : bool):
        self.piece_id = piece_id
        self.x = x
        self.y = y
        self.rotation = rotation
        self.flip = flip

    def modify_game(self, game: 'BlockusGame', inplace: bool = False) -> GameState:
        """ Place the piece on the board.
        We set 0 values to -1, and 1 values to the player id.
        We then flip and rotate the piece according to the action.
        Then we place the piece on the board, s.t. the left upper corner of the piece is at the given x, y coordinates.
        """
        # Get the piece
        piece = BLOCKUS_PIECE_MAP[self.piece_id]
        piece = np.rot90(piece, k=self.rotation)
        if self.flip:
            piece = np.flip(piece, axis=0)
        # Modify the values of the piece
        piece = np.where(piece == 0, -1, game.current_pid)
        # Place the piece on the board
        board = np.array(game.board)
        board[self.x:self.x+piece.shape[0], self.y:self.y+piece.shape[1]] = piece
        # Update the board
        game.board = board.tolist()
        # Remove the piece from the player's remaining pieces
        game.player_remaining_pieces[game.current_pid].remove(self.piece_id)
        # Update the current player
        game.current_pid = (game.current_pid + 1) % len(game.players)
        # Return the new game state
        return game.game_state_class.from_game(game, copy = False)
        
    
    def check_action_is_legal(self, game: 'BlockusGame') -> Tuple[bool, str]:
        """ Check if the action is legal in the given game state.
        """
        # Check if the piece is in the player's remaining pieces
        if self.piece_id not in game.player_remaining_pieces[game.current_pid]:
            return False, f"The piece with id {self.piece_id} is not in the player's remaining pieces."
        # Check if the piece is placed on the board
        piece = BLOCKUS_PIECE_MAP[self.piece_id]
        piece = np.rot90(piece, k=self.rotation)
        if self.flip:
            piece = np.flip(piece, axis=0)
        piece = np.where(piece == 0, -1, game.current_pid)
        if self.x + piece.shape[0] > game.board_size[0] or self.y + piece.shape[1] > game.board_size[1]:
            return False, f"The piece with id {self.piece_id} is placed outside the board."
        # Check that all the values in board are -1, where the piece is != -1
        board_area = np.array(game.board)[self.x:self.x+piece.shape[0], self.y:self.y+piece.shape[1]]
        if np.any((board_area != -1) & (piece != -1)):
            return False, f"The piece with id {self.piece_id} overlaps with another piece."
        # Calculate the size of the piece
        piece_size = np.sum(piece != -1)
        # Find a coordinate, where the piece is true
        piece_begin = np.where(piece != -1)
        piece_begin = (piece_begin[0][0], piece_begin[1][0])
        piece_begin = (self.x + piece_begin[0], self.y + piece_begin[1])
        board_copy = np.copy(game.board)
        board_copy[self.x:self.x+piece.shape[0], self.y:self.y+piece.shape[1]] = piece
        # Calculate the number self.pid pieces, that are connected from the piece_begin
        stack = [piece_begin]
        visited = []
        connected_pieces = 0
        piece_grids = [piece_begin]
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.append((x, y))
            connected_pieces += 1
            # Now we need to check the neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                # Check if the new coordinates are inside the board
                if new_x < 0 or new_x >= game.board_size[0] or new_y < 0 or new_y >= game.board_size[1]:
                    continue
                # Check if the new coordinates are the same player id
                if board_copy[new_x, new_y] == game.current_pid:
                    stack.append((new_x, new_y))
                    piece_grids.append((new_x, new_y))
        # Check if the piece is connected
        if connected_pieces != piece_size:
            return False, f"The selected piece is connected to another piece with a side."
        # Finally, check that there is atleast one own piece, connected via a corner to the new piece
        # Corner of the piece is any square of the piece, that has atleast 3 empty squares around it.
        has_connected_corner = False
        piece_corners = []
        for x, y in piece_grids:
            dxdys = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
            num_free_squares = 0
            for dx, dy in dxdys:
                new_x, new_y = x + dx, y + dy
                # Check if the new coordinates are inside the board
                if new_x < 0 or new_x >= game.board_size[0] or new_y < 0 or new_y >= game.board_size[1]:
                    continue
                if board_copy[new_x, new_y] == game.current_pid:
                    break
            else:
                piece_corners.append((x, y))

        





