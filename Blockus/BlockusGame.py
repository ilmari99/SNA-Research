import os
import sys
from RLFramework import GameState
from RLFramework.Player import Player
from RLFramework.Result import Result
import numpy as np
from RLFramework import Game
from typing import Dict, List, Tuple
from BlockusAction import BlockusAction
from BlockusGameState import BlockusGameState
from BlockusPlayer import BlockusPlayer
from BlockusResult import BlockusResult
import matplotlib.pyplot as plt
from RLFramework.utils import TFLiteModel

from BlockusPieces import BLOCKUS_PIECE_MAP


class BlockusGame(Game):
    """ A class representing the game TicTacToe.
    """
    def __init__(self, board_size : Tuple[int, int] = (20, 20), model_paths=[], **kwargs):
        super().__init__(BlockusGameState, custom_result_class=BlockusResult, **kwargs)
        self.board_size = board_size
        self.model_paths = []
        self.board = None
        self.set_models(model_paths)
        
    def play_game(self, players: List[Player]) -> Result:
        # Load the models before playing the game
        current_models = set(self.model_paths)
        model_paths = set([p.model_path for p in players if (hasattr(p, "model_path") and p.model_path is not None)])
        # If there are any new models, load them
        if model_paths - current_models:
            self.set_models(list(model_paths))
        out = super().play_game(players)
        if self.render_mode == "human":
            plt.savefig("blockus.png")
        return out
    
    def __repr__(self) -> str:
        s = ""
        for i, row in enumerate(self.board):
            s += " ".join([str(x).ljust(2) for x in row]) + "\n"
        return s

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
        self.model_paths = model_paths
        self.models = {path: TFLiteModel(path) for path in model_paths}
    
    def initialize_game(self, players: List[BlockusPlayer]) -> None:
        """ When the game is started, we need to set the board.
        """
        self.board = [[-1 for _ in range(self.board_size[1])] for _ in range(self.board_size[0])]
        #self.board = np.array(self.board)
        self.current_pid = 0
        self.player_remaining_pieces = [list(range(21)) for _ in players]
        self.finished_players = []
    
    def init_render_human(self) -> None:
        plt.cla()
        plt.clf()
        plt.close()
        self.fig, self.ax = plt.subplots(1, 2)
        plt.ion()
        plt.show()
        
    def render_human(self, ax: plt.Axes = None) -> None:
        """ Render the game.
        """
        board_ax : plt.Axes = self.ax[0]
        pieces_ax : plt.Axes = self.ax[1]
        board_ax.clear()
        color_map = {-1 : "black", 0 : "red", 1 : "blue", 2 : "green", 3 : "yellow"}
        board_ax.imshow(self.board, cmap="tab20", vmin=-1, vmax=3)
        board_ax.set_xticks(np.arange(-0.5, self.board_size[1], 1), minor=True)
        board_ax.set_yticks(np.arange(-0.5, self.board_size[0], 1), minor=True)
        board_ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        board_ax.set_title(f"Scores: {self.player_scores}")
        board_ax.set_xticks([])
        board_ax.set_yticks([])

        # Add section to show remaining pieces for current player
        remaining_pieces = self.player_remaining_pieces[self.current_pid]
        remaining_pieces = [BLOCKUS_PIECE_MAP[piece] for piece in remaining_pieces]
        
        pieces_ax.clear()
        # Place all the pieces in the pieces_ax
        # We place the pieces on 20x20 grid, where each piece is placed in a 4x4 grid
                
        remaining_pieces_board = np.zeros_like(self.board) - 1
        row = 0
        col = 0
        for i, piece in enumerate(remaining_pieces):
            piece = np.where(piece == 0, -1, self.current_pid)
            remaining_pieces_board[row:row+piece.shape[0], col:col+piece.shape[1]] = piece
            col += 4
            if col >= 20:
                col = 0
                row += 4
        
        pieces_ax.imshow(remaining_pieces_board, cmap="tab20", vmin=-1, vmax=3)
        pieces_ax.set_xticks(np.arange(-0.5, 20, 1), minor=True)
        pieces_ax.set_yticks(np.arange(-0.5, 20, 1), minor=True)
        pieces_ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        pieces_ax.set_title(f"Player {self.current_pid} remaining pieces")
        pieces_ax.set_xticks([])
        pieces_ax.set_yticks([])
        # Add a grid dividing the pieces in to 4x4 grids by addding black lines
        for i in range(5):
            pieces_ax.axhline(i * 4 - 0.5, color='black', linewidth=2)
            pieces_ax.axvline(i * 4 - 0.5, color='black', linewidth=2)
            
        
        plt.pause(0.01)

    def restore_game(self, game_state: BlockusGameState) -> None:
        """ Restore the game to the state described by the game_state.
        We don't need to worry about the players states or their scores, as they are automatically restored.
        """
        self.board = game_state.board
        self.player_remaining_pieces = game_state.player_remaining_pieces
        self.current_pid = game_state.current_pid
        self.previous_turns = game_state.previous_turns
        self.finished_players = game_state.finished_players
    
    def calculate_reward(self, pid : int, game_state: BlockusGameState) -> float:
        """ Calculate the reward for the player.
        If the player wins, the reward is 1.0.
        If the game is a draw, the reward is 0.5
        """
        #print("Called calculate_reward")
        all_players_finished = len(game_state.finished_players) == len(self.players)
        # Number of squares occupied by player
        if not all_players_finished:
            player_board = np.array(game_state.board) == pid
            player_area = np.sum(player_board)
            # Return how much new area the player has occupied
            score_boost = player_area - game_state.player_scores[pid]
            self.players[pid].logger.info(f"Player {pid} score: {game_state.player_scores[pid]}, new reward: {score_boost}")
            return score_boost
        
        # If the game is finished: +50 for win, +25 for draw, 0 for loss, and +15 if all pieces placed
        all_pieces_placed = len(game_state.player_remaining_pieces[pid]) == 0
        score_boost = 0
        if all_pieces_placed:
            self.players[pid].logger.info(f"Player {pid} placed all pieces")
            score_boost += 15
        self.players[pid].logger.info(f"Player {pid} score: {game_state.player_scores[pid]}, new reward: {score_boost}")
        return score_boost
    
    def environment_action(self, game_state : 'BlockusGameState') -> 'BlockusGameState':
        self.update_finished_players_in_gamestate(game_state)
        self.update_player_scores_in_gamestate(game_state)
        self.update_player_attributes()
        return super().environment_action(game_state)
    
    def check_is_player_finished(self, pid : int, game_state: GameState) -> bool:
        """ A player is finished if the game is finished.
        I.e. if the player has won, or if there are no more free spots, or if the other player is finished.
        """
        if pid in game_state.finished_players:
            return True
        return False
    
    def is_grid_inside_board(self, grid, board_size):
        """ Check if the grid is inside the board.
        """
        return grid[0] >= 0 and grid[0] < board_size[0] and grid[1] >= 0 and grid[1] < board_size[1]
    
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
        # We add the first vector to the end, so that we can check the last corner
        to_surrounding_grids = [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
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
                    if not all([self.is_grid_inside_board(grid, self.board_size) for grid in three_surrounding_grids]):
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
                        if not self.is_grid_inside_board(side_grid, self.board_size):
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
        
    
    def get_all_possible_actions(self) -> List[BlockusAction]:
        """ Return all possible actions.
        """
        available_pieces = self.player_remaining_pieces[self.current_pid]
        actions = []
        corner_grids, grids_sharing_corner = self.get_corner_positions(self.current_pid)
        if np.sum(np.array(self.board) == self.current_pid) == 0:
            # If the board is empty, the only valid places are the corners of the board.
            grids_sharing_corner = [(0, 0), (0, self.board_size[1] - 1), (self.board_size[0] - 1, 0), (self.board_size[0] - 1, self.board_size[1] - 1)]
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
                            if not all([self.is_grid_inside_board(corner, self.board_size) for corner in all_corners]):
                                #print(f"Piece {piece_id} not inside board")
                                continue
                            # Check that the move is valid through the Action
                            action = BlockusAction(piece_id, lu_corner[0], lu_corner[1], rot, flip)
                            is_legal, msg = action._check_action_is_legal(self)
                            if is_legal:
                                actions.append(action)
        #print(f"Number of possible actions: {len(actions)}") 
        if len(actions) == 0:
            # Add null action
            actions.append(BlockusAction(-1, -1, -1, -1, False))
            self.finished_players.append(self.current_pid)
        return actions
    