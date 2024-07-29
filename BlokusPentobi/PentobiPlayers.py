import os
import random
import warnings
import numpy as np

from PentobiGTP import PentobiGTP,random_playout

    
class PentobiInternalPlayer:
    def __init__(self, pid, pentobi_sess, get_move_pentobi_sess = None,move_selection_strategy="best", move_selection_kwargs={}, name="PentobiInternalPlayer"):
        self.pid = pid
        self.pentobi_sess : PentobiGTP = pentobi_sess
        # We can provide a separate PentobiGTP session to get moves from a session with different (level) settings
        self.has_separate_pentobi_sess = True if get_move_pentobi_sess is not None else False
        self.name = name
        self.get_move_pentobi_sess = get_move_pentobi_sess
        if get_move_pentobi_sess is None:
            self.get_move_pentobi_sess : PentobiGTP = pentobi_sess
            
        self.move_selection_strategy = move_selection_strategy
        self.move_selection_kwargs = move_selection_kwargs
        assert move_selection_strategy in ["best", "random", "epsilon_greedy"], f"Invalid move selection strategy: {move_selection_strategy}"
        if move_selection_strategy != "epsilon_greedy":
            assert not move_selection_kwargs, f"move_selection_kwargs should be empty for move_selection_strategy: {move_selection_strategy}"
        if move_selection_strategy == "epsilon_greedy" and "epsilon" not in move_selection_kwargs:
            warnings.warn("No epsilon provided for epsilon_greedy move selection strategy. Defaulting to epsilon=0.1")
            self.move_selection_kwargs["epsilon"] = 0.1
            
    def set_move_session_state(self):
        if not self.has_separate_pentobi_sess:
            return
        # We save the state of penobi_sess to a unique temporary file
        hash_str = str(hash(self.pentobi_sess.board)) + str(random.randint(0, 2**32-1))
        self.pentobi_sess.send_command(f"savesgf {hash_str}.blksgf")
        # We load the state of pentobi_sess to the get_move_pentobi_sess
        self.get_move_pentobi_sess.send_command(f"loadsgf {hash_str}.blksgf", lock=False)
        self.get_move_pentobi_sess.current_player = self.pid
        # Remove the temporary file
        os.remove(f"{hash_str}.blksgf")
        return
    
    def _make_move_with_pentobi_sess(self):
        with self.get_move_pentobi_sess.lock:
            self.set_move_session_state()
            #assert np.array_equal(self.pentobi_sess.board_np, self.get_move_pentobi_sess.board_np), "Boards are not equal"
            mv = self.get_move_pentobi_sess.bot_get_move(self.pid,lock=False)
        return mv
    
    def play_move(self):
        if self.move_selection_strategy == "random":
            all_moves = self.pentobi_sess.get_legal_moves(self.pid)
            selected_move = random.choice(all_moves)
        elif self.move_selection_strategy == "epsilon_greedy":
            if np.random.rand() < self.move_selection_kwargs["epsilon"]:
                all_moves = self.pentobi_sess.get_legal_moves(self.pid)
                selected_move = random.choice(all_moves)
            else:
                selected_move = self._make_move_with_pentobi_sess()
        elif self.move_selection_strategy == "best":
            selected_move = self._make_move_with_pentobi_sess()
        #print(f"Player {self.pid} chose move: {selected_move}", flush=True)
        if selected_move == "=":
            selected_move = "pass"
        self.pentobi_sess.play_move(self.pid, selected_move, mock_move=False)
        return
    
class MCTSNode:
    """ An MCTS node contains the following information:
    - The state of the game (blksgf file, pid)
    - The number of visits
    - The number of wins
    - The children of the node
    - The parent of the node
    """
    def __init__(self, state_file, pid, session : PentobiGTP, parent=None):
        self.state_file = state_file
        with open(state_file, "r") as f:
            self.state_file_content = f.read()
        self.pid = pid
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.session = session
    
    @property
    def hash(self):
        return hash(self.state_file_content) + self.pid
    
    def get_moves(self):
        return self.session.get_legal_moves(self.pid)
    
    def set_session_state(self):
        self.session.send_command(f"loadsgf {self.state_file}", lock=True)
        self.session.current_player = self.pid
        return
    
    def set_children(self):
        moves = self.get_moves()
        if not moves:
            return
        states = []
        for move in moves:
            self.set_session_state()
            self.session.play_move(self.pid, move, mock_move=True)
            state_file = f"{hash(self.session.board)}.blksgf"
            self.session.send_command(f"savesgf {state_file}")
            states.append((state_file, self.session.current_player))
        self.moves = moves
        self.children = [MCTSNode(state_file, pid, self.session, parent=self) for state_file, pid in states]
        return
    
    def get_ucb(self, c=1.0):
        if self.visits == 0:
            return np.inf
        return self.wins/self.visits + c*np.sqrt(np.log(self.parent.visits)/self.visits)
    
    def select_child(self, c=1.0):
        ucb_vals = [child.get_ucb(c) for child in self.children]
        return self.children[np.argmax(ucb_vals)]
    
    def playout(self):
        assert self.parent is not None, "Cannot playout from root node"
        self.set_session_state()
        while not self.session.is_game_finished():
            moves = self.session.get_legal_moves(self.session.current_player)
            move = random.choice(moves)
            self.session.play_move(self.session.current_player, move)
        scores = self.session.score
        max_score = max(scores)
        won = scores[self.pid-1] == max_score and scores.count(max_score) == 1
        return 1 if won else 0
    
    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)
        return
    
    def delete_tree(self):
        # remove all state files
        for child in self.children:
            child.delete_tree()
            os.remove(child.state_file)
        return
    
    def __repr__(self):
        return f"MCTSNode: Visits: {self.visits}, Wins: {self.wins}, Children: {len(self.children)}"


class MCTSPentobiPlayer:
    def __init__(self, pid, pentobi_sess, name="MCTSPentobiPlayer"):
        self.pid = pid
        self.pentobi_sess : PentobiGTP = pentobi_sess
        self.name = name
        
    def get_root(self):
        current_state = f"{hash(self.pentobi_sess.board)}.blksgf"
        if self.previous_root is None:
            self.pentobi_sess.send_command(f"savesgf {current_state}")
            root = MCTSNode(current_state, self.pid, self.pentobi_sess)
            return root
        # If the previous root is not None, and the current state is already in the tree, return the node
        nodes = [self.previous_root]
        while nodes:
            node = nodes.pop()
            if node.state_file == current_state:
                return node
            nodes.extend(node.children)
        # If the current state is not in the tree, return None
        self.pentobi_sess.send_command(f"savesgf {current_state}")
        root = MCTSNode(current_state, self.pid, self.pentobi_sess)
        return root
        
        
    def play_move(self):
        """ Run an MCTS search to find the best move and play it
        """
        current_state = f"{hash(self.pentobi_sess.board)}.blksgf"
        self.pentobi_sess.send_command(f"savesgf {current_state}")
        root = MCTSNode(current_state, self.pid, self.pentobi_sess)
        root.set_children()
        for _ in range(100):
            node = root
            # Find a leaf node
            while node.children:
                node = node.select_child()
            # Playout from the leaf node
            result = node.playout()
            node.backpropagate(result)
        best_child_idx = np.argmax([child.visits for child in root.children])
        best_child = root.children[best_child_idx]
        best_move = root.moves[best_child_idx]
        # Return to the current state
        self.pentobi_sess.send_command(f"loadsgf {current_state}")
        self.pentobi_sess.current_player = self.pid
        self.pentobi_sess.play_move(self.pid, best_move)
        
        # Save the root
        self.previous_root = root
        
        
        
            
                
    
class PentobiNNPlayer:
    def __init__(self, pid, pentobi_sess, model, move_selection_strategy="best", move_selection_kwargs={}, name="PentobiNNPlayer"):
        self.pid = pid
        self.pentobi_sess : PentobiGTP = pentobi_sess
        self.model = model
        self.name = name
        self.move_selection_strategy = move_selection_strategy
        self.move_selection_kwargs = move_selection_kwargs
        
    def play_move(self):
        self.make_action_with_nn(self.move_selection_strategy, self.move_selection_kwargs)
        return
        
    def get_next_states(self):
        all_moves = self.pentobi_sess.get_legal_moves(self.pid)
        next_states = []
        for move in all_moves:
            if move == "pass":
                continue
            self.pentobi_sess.play_move(self.pid, move, mock_move=True)
            board_np = self.pentobi_sess.board_np.flatten()
            misc = np.array([self.pid-1, self.pid-1])
            next_states.append(np.concatenate([misc, board_np]))
            
            resp = self.pentobi_sess.send_command("undo")
            if "?" in resp:
                raise Exception(f"'undo' command failed")
            self.pentobi_sess.current_player = self.pid
        return all_moves, np.array(next_states, dtype=np.float32)
        
    def make_action_with_nn(self, move_selection_strategy="best", move_selection_kwargs={}):
        moves, next_states = self.get_next_states()
        if move_selection_strategy == "random":
            selected_move = random.choice(moves)
        elif move_selection_strategy == "epsilon_greedy":
            epsilon = move_selection_kwargs.get("epsilon", 0.1)
            if np.random.rand() < epsilon:
                selected_move = random.choice(moves)
            else:
                move_selection_strategy = "best"
                move_selection_kwargs = {}
        if len(moves) == 1 and moves[0] == "pass":
            return self.pentobi_sess.play_move(self.pid, "pass", mock_move=False)
        predictions = self.model.predict(next_states)
        predictions = np.array(predictions)
        #print(predictions,flush=True)
        # If each prediction is not a single value, we'll assume the values are the probabilities
        # of ending up 1st, 2nd, 3rd, or 4th
        if len(predictions[0]) > 1 and predictions.shape[1] > 1:
            weights = np.array([4,3,2,1])
            # Dot product of predictions and weights
            predictions = np.dot(predictions, weights)
        #print(predictions,flush=True)
        if move_selection_strategy == "best":
            best_move = np.argmax(predictions)
            #print(f"Best move index: {best_move}")
            selected_move = moves[best_move]
        elif move_selection_strategy == "weighted":
            top_p = move_selection_kwargs.get("top_p", 1.0)
            psoftmax = np.exp(predictions) / np.sum(np.exp(predictions))
            psoftmax = psoftmax.flatten()
            #print(psoftmax,flush=True)
            # Sort from best to worst
            sort_idxs = np.argsort(psoftmax)[::-1]
            #print(sort_idxs,flush=True)
            psoftmax = psoftmax[sort_idxs]
            #print(moves,flush=True)
            moves = [moves[i] for i in sort_idxs]
            # Take the top p moves
            psoftmax_cumsum = np.cumsum(psoftmax)
            cutoff = psoftmax_cumsum >= top_p - 10**-6
            cutoff_idx = np.argmax(cutoff) + 1
            #print(f"cutoff idx: {cutoff_idx}", flush=True)
            psoftmax = psoftmax[:cutoff_idx]
            moves = moves[:cutoff_idx]
            # Renormalize
            psoftmax = psoftmax / np.sum(psoftmax)
            
            selected_move = np.random.choice(moves, p=psoftmax)
        #print(f"Selected move: {selected_move}")
        self.pentobi_sess.play_move(self.pid, selected_move)
        return