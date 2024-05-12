from typing import List, SupportsFloat, Tuple, TYPE_CHECKING

import numpy as np
from RLFramework.Game import Game
from RLFramework.GameState import GameState
from BlockusAction import BlockusAction
from BlockusPieces import BLOCKUS_PIECE_MAP
from BlockusPlayer import BlockusPlayer
from BlockusGame import BlockusGame

class BlockusGameState(GameState):
    """ A class representing the state of the game TicTacToe.
    """
    def __init__(self, state_json):
        super().__init__(state_json)
        self.board = state_json["board"]
        self.player_remaining_pieces : List[List[int]] = state_json["player_remaining_pieces"]
        self.finished_players : List[int] = state_json["finished_players"]
        
    @property
    def game(self) -> 'BlockusGame':
        game = BlockusGame(board_size=(len(self.board), len(self.board[0])))
        game.initialize_game_wrap([BlockusPlayer(name=f"Player{i}", logger_args=None) for i in range(4)])
        game.restore_game(self)
        return game
    
    def calculate_reward(self, pid : int) -> float:
        """ Calculate the reward for the player.
        If the player wins, the reward is 1.0.
        If the game is a draw, the reward is 0.5
        """
        #print("Called calculate_reward")
        all_players_finished = len(self.finished_players) == len(self.player_scores)
        # Number of squares occupied by player
        if not all_players_finished:
            player_board = np.array(self.board) == pid
            player_area = np.sum(player_board)
            # Return how much new area the player has occupied
            score_boost = player_area - self.player_scores[pid]
            return score_boost
        
        # If the game is finished: +50 for win, +25 for draw, 0 for loss, and +15 if all pieces placed
        all_pieces_placed = len(self.player_remaining_pieces[pid]) == 0
        score_boost = 0
        if all_pieces_placed:
            score_boost += 15
        return score_boost
        
    def _check_action_is_legal(self, action) -> Tuple[bool, str]:
        """ Check if the action is legal in the given game state.
        """
        if action.piece_id == -1:
            return True, ""
        # Check if the piece is in the player's remaining pieces
        if action.piece_id not in self.player_remaining_pieces[self.current_pid]:
            return False, f"The piece with id {action.piece_id} is not in the player's remaining pieces."
        # Check if the piece is placed on the board
        piece = BLOCKUS_PIECE_MAP[action.piece_id]
        piece = np.rot90(piece, k=action.rotation)
        if action.flip:
            piece = np.flip(piece, axis=0)
        piece = np.where(piece == 0, -1, self.current_pid)
        
        if ((not self.is_grid_inside_board((action.x, action.y))) or 
            (not self.is_grid_inside_board((action.x + piece.shape[0] - 1,
                                            action.y + piece.shape[1] - 1)))):
            return False, f"The piece with id {self.piece_id} is placed outside the board."
        # Check that all the values in board are -1, where the piece is != -1
        board_area = np.array(self.board)[action.x:action.x+piece.shape[0], action.y:action.y+piece.shape[1]]
        if np.any((board_area != -1) & (piece != -1)):
            return False, f"The piece with id {action.piece_id} overlaps with another piece."
        
        piece_grids = np.where(piece != -1)
        piece_grids = [(action.x + piece_grids[0][i], action.y + piece_grids[1][i]) for i in range(len(piece_grids[0]))]
        
        board_size = (len(self.board), len(self.board[0]))
        # If the piece is the player's first piece, then one of it's grids must be in the player's corner 0:lu, 1:ru, 2:rd, 3:ld
        if np.sum(np.array(self.board) == self.current_pid) == 0:
            corner_grids = [(0, 0),
                            (0, board_size[1]-1),
                            (board_size[0]-1, board_size[1]-1),
                            (board_size[0]-1, 0)]
            for corner_grid in corner_grids:
                if corner_grid in piece_grids:
                    return True, ""
            return False, f"The selected piece is not connected to the player's corner."

        # Calculate the size of the piece
        piece_size = np.sum(piece != -1)
        # Find a coordinate, where the piece is true
        piece_begin = np.where(piece != -1)
        piece_begin = (piece_begin[0][0], piece_begin[1][0])
        piece_begin = (action.x + piece_begin[0], action.y + piece_begin[1])
        board_copy = np.copy(self.board)
        # Set the piece on the board
        piece_mask = piece != -1
        board_copy[action.x:action.x+piece.shape[0], action.y:action.y+piece.shape[1]][piece_mask] = piece[piece_mask]
        
        # Calculate the number self.pid pieces, that are connected from the piece_begin
        connected_pieces = self.find_num_connected_pieces(board_copy,
                                                          self.current_pid,
                                                          piece_begin[0],
                                                          piece_begin[1])
        board_copy = np.array(self.board)
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
                    if not all([self.is_grid_inside_board(grid) for grid in adjacent_grids]):
                        continue
                    #print(adjacent_grids)
                    # Check that all the grids are not ours
                    if all([board_copy[grid] == self.current_pid for grid in adjacent_grids]):
                        continue
                    has_free_corner = True
                    break
            if not has_free_corner:
                continue
            
            # Check if the piece is connected to another piece with a corner
            corner_changes = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
            for dx, dy in corner_changes:
                new_x, new_y = x + dx, y + dy
                if new_x < 0 or new_x >= board_size[0] or new_y < 0 or new_y >= board_size[1]:
                    continue
                if board_copy[new_x, new_y] == self.current_pid and (new_x, new_y) not in piece_grids:
                    has_connected_corner = True
                    break
            if has_connected_corner:
                break
        if not has_connected_corner:
            return False, f"The selected piece is not connected to another piece with a corner."
        return True, ""
        
    def get_all_possible_actions(self) -> List[BlockusAction]:
        """ Return all possible actions.
        """
        available_pieces = self.player_remaining_pieces[self.current_pid]
        actions = []
        corner_grids, grids_sharing_corner = self.get_corner_positions(self.current_pid)
        board_size = (len(self.board), len(self.board[0]))
        if np.sum(np.array(self.board) == self.current_pid) == 0:
            # If the board is empty, the only valid places are the corners of the board.
            grids_sharing_corner = [(0, 0), (0, board_size[1] - 1), (board_size[0] - 1, 0), (board_size[0] - 1, board_size[1] - 1)]
        # All the possible actions are all the ways to place a piece on the board,
        # s.t. atleast of the piece's grids is in a 'grids_sharing_corner' grid.
        for valid_shared_corner in grids_sharing_corner:
            for piece_id in available_pieces:
                piece = BLOCKUS_PIECE_MAP[piece_id]
                piece_grids = np.where(piece != 0)
                max_block_size = self.find_num_connected_pieces(np.array(self.board),
                                                                -1,
                                                                valid_shared_corner[0],
                                                                valid_shared_corner[1]
                                                                )
                if max_block_size < len(piece_grids[0]):
                    #print(f"Largest possible block size {max_block_size}. Can not place piece with {len(piece_grids[0])} blocks")
                    continue
                # Now, we test all the possible placements of the piece
                # to the grid valid_shared_corner
                # The piece can be rotated and flipped, so we need to test all the possible combinations
                num_rotations = 4
                # If the piece is full (only has one value), and square, we only have 1 rotation
                if len(set(piece.flatten())) == 1 and piece.shape[0] == piece.shape[1]:
                    num_rotations = 1
                # If the piece is full, but not square, we have 2
                if len(set(piece.flatten())) == 1 and piece.shape[0] != piece.shape[1]:
                    num_rotations = 2
                    
                for rot in range(num_rotations):
                    for flip in [True, False] if num_rotations == 4 else [False]:
                        transformed_piece = np.rot90(piece, k=rot)
                        transformed_piece = np.flip(transformed_piece, axis=0) if flip else transformed_piece
                        transformed_piece_grids = np.where(transformed_piece != 0)
                        # Let 'piece_grid' be the grid in the piece, that is placed on the grid valid_shared_corner
                        for piece_grid in zip(transformed_piece_grids[0], transformed_piece_grids[1]):
                            # The lu corner of the piece
                            lu_corner = (valid_shared_corner[0] - piece_grid[0], valid_shared_corner[1] - piece_grid[1])
                            # Check that all corners of the piece are inside the board
                            all_corners = [lu_corner, (lu_corner[0] + transformed_piece.shape[0] - 1, lu_corner[1]),
                                             (lu_corner[0], lu_corner[1] + transformed_piece.shape[1] - 1),
                                             (lu_corner[0] + transformed_piece.shape[0] - 1, lu_corner[1] + transformed_piece.shape[1] - 1)]
                            if not all([self.is_grid_inside_board(corner) for corner in all_corners]):
                                #print(f"Piece {piece_id} not inside board")
                                continue
                            # Check that the move is valid through the Action
                            action = BlockusAction(piece_id, lu_corner[0], lu_corner[1], rot, flip)
                            # Check if an equivalent action is already in the list
                            #if action in actions:
                            #    continue
                            is_legal, msg = self._check_action_is_legal(action)
                            if is_legal:
                                actions.append(action)
        #print(f"Number of possible actions: {len(actions)}") 
        if len(actions) == 0:
            # Add null action
            actions.append(BlockusAction(-1, -1, -1, -1, False))
            self.finished_players.append(self.current_pid)
        return actions
        
        
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
                if not self.is_grid_inside_board((new_x, new_y)):
                    continue
                # Check if the new coordinates are the same player id
                if board[new_x, new_y] == pid:
                    stack.append((new_x, new_y))
        return num_connected_pieces
    
    def get_corner_positions(self, pid : int) -> Tuple[List[Tuple[int, int]],
                                                       List[Tuple[int, int]]]:
        """ Return two lists:
        - The first list contains the grids, which have a player's piece, and that
        are a corner of the occupied area.
        - The second list contains the grids, that are free, that share a corner
        with atleast one of the grids in the first list, and that do not have any
        common sides with the player's pieces.
        """
        player_board = np.array(self.board) == pid
        corner_grids = set()
        grids_sharing_corner = set()
        board_size = (len(self.board), len(self.board[0]))
        # We add the first vector to the end, so that we can check the last corner
        to_surrounding_grids = [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
        for i in range(board_size[0]):
            for j in range(board_size[1]):
                # Check that the grid is occupied by the player
                if not player_board[i,j]:
                    continue
                # The grid is a corner grid, if three consecutive surrounding
                # grids are not occupied by the player, s.t. the middle of the three is abs(1,1)
                # from the current grid.
                for start_idx in range(9 - 2):
                    start_vec = to_surrounding_grids[start_idx]
                    # Check that both x and y in start vec are not |1|
                    if abs(start_vec[0]) == 1 and abs(start_vec[1]) == 1:
                        #print(f"Start vec {start_vec} not valid")
                        continue
                    middle_vec = to_surrounding_grids[start_idx + 1]
                    end_vec = to_surrounding_grids[start_idx + 2]
                    # Get the positions of the three surrounding grids
                    three_surrounding_grids = [(i + vec[0], j + vec[1]) for vec in [start_vec, middle_vec, end_vec]]
                    # Check that all the grids are inside the board
                    if not all([self.is_grid_inside_board(grid) for grid in three_surrounding_grids]):
                        #print(f"Surrounding grids {three_surrounding_grids} not inside board")
                        continue
                    # Check that none of the grids are occupied by the player
                    if any([player_board[grid] for grid in three_surrounding_grids]):
                        #print(f"Surrounding grids {three_surrounding_grids} occupied")
                        continue
                    # The middle grid must also be free
                    if self.board[i + middle_vec[0]][j + middle_vec[1]] != -1:
                        #print(f"Middle grid {i + middle_vec[0], j + middle_vec[1]} not free")
                        continue
                    # Finally, check that none of our own pieces share a side with the middle grid
                    side_grids = to_surrounding_grids[:7:2]
                    is_valid = True
                    for side_vec in side_grids:
                        side_grid = (i + middle_vec[0] + side_vec[0], j + middle_vec[1] + side_vec[1])
                        if not self.is_grid_inside_board(side_grid):
                            #print(f"Side grid {side_grid} not inside board")
                            continue
                        if player_board[side_grid]:
                            is_valid = False
                            break
                    if is_valid:
                        corner_grids.add((i, j))
                        grids_sharing_corner.add((i + middle_vec[0], j + middle_vec[1]))
        #print(f"Corner grids: {corner_grids}")
        #print(f"Grids sharing corner: {grids_sharing_corner}")
        return list(corner_grids), list(grids_sharing_corner)
    
    def is_grid_inside_board(self, grid):
        """ Check if the grid is inside the board.
        """
        board_size = (len(self.board), len(self.board[0]))
        return grid[0] >= 0 and grid[0] < board_size[0] and grid[1] >= 0 and grid[1] < board_size[1]
    
    def is_terminal(self) -> bool:
        """ Return True if the game is over.
        """
        return self.perspective_pid in self.finished_players
    
    def get_possible_actions(self) -> List[int]:
        """ Get the possible actions.
        """
        return self.get_all_possible_actions()
    
    def game_to_state_json(cls, game : 'BlockusGame', player):
        """ Convert a Game to a state_json.
        """
        state_json = {
            "board" : game.board,
            "player_remaining_pieces" : game.player_remaining_pieces,
            "finished_players" : game.finished_players,
        }
        return state_json
    
    def to_vector(self, perspective_pid = None) -> List[SupportsFloat]:
        """ Convert the state to a vector.
        """
        if perspective_pid is None:
            perspective_pid = self.perspective_pid
        board_arr = np.array(self.board)
        return [perspective_pid] + [self.current_pid] + board_arr.flatten().tolist()