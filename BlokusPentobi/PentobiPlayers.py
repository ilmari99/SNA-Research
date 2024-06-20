import os
import random
import warnings
import numpy as np

from PentobiGTP import PentobiGTP

    
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
    
    def _make_move_with_proc(self):
        with self.get_move_pentobi_sess.lock:
            self.set_move_session_state()
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
                selected_move = self._make_move_with_proc()
        elif self.move_selection_strategy == "best":
            selected_move = self._make_move_with_proc()
        print(f"Player {self.pid} chose move: {selected_move}", flush=True)
        if selected_move == "=":
            selected_move = "pass"
        self.pentobi_sess.play_move(self.pid, selected_move, mock_move=False)
        return
    
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