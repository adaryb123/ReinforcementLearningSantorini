"""
Version of RL agent for casual play, or as an opponent while evaluating the main model
Author: Adam Rybansky (xryban00)
FIT VUT 2023
"""

from engine.Board import encode_board
import numpy as np
import torch as T
import os
try:
    from deep_q_network_2x8 import DeepQNetwork
except ImportError:
    from train.deep_q_network_2x8 import DeepQNetwork


class RLBot:
    def __init__(self, color, seed,
                 input_dims=(3,5,5), n_actions=128, checkpoint_frequency=1000, lr=0.0001,
                 adamw_optimizer=False, dropout=False, chkpt_dir='../train/models/', former_seed=None):
        self.color = color
        self.chkpt_dir = chkpt_dir
        self.seed = seed
        self.counter = 0
        self.checkpoint_frequency = checkpoint_frequency

        self.q_eval = DeepQNetwork(lr=lr, n_actions = n_actions,
                                          input_dims=input_dims,
                                          name=str(self.seed) + '_secondary_q_eval',
                                          chkpt_dir=self.chkpt_dir, adamw_optimizer=adamw_optimizer, dropout=dropout)

        if former_seed is not None:
            filename = self.chkpt_dir + former_seed + "_q_eval"
            if os.path.exists(filename):
                self.q_eval.load_checkpoint(filename)


    def check_model_file_exists(self):
        filename = self.chkpt_dir + self.seed + "_q_eval"
        return os.path.exists(filename)

    def reload_network(self):
        filename = self.chkpt_dir + self.seed + "_q_eval"
        if os.path.exists(filename):
            self.q_eval.load_checkpoint(self.chkpt_dir + self.seed + "_q_eval")

    def flip_tensor_values(self, states):
        states[:, 1, :, :] *= -1
        return states

    def make_turn(self, board, _=None):

        if self.counter % self.checkpoint_frequency == 0:
            self.reload_network()
        self.counter += 1

        observation = encode_board(board)
        state = np.array([observation], copy=False, dtype=np.float32)

        if self.color=="black":
            state = self.flip_tensor_values(state)

        state_tensor = T.tensor(state).to(self.q_eval.device)
        _, advantages = self.q_eval.forward(state_tensor)

        for action in range(len(advantages[0])):
            move = board.create_move_from_number(action, self.color)
            valid, msg = board.check_move_valid(move)
            if not valid:
                advantages[0, action] = -np.inf

        best_action = T.argmax(advantages).item()
        move = board.create_move_from_number(best_action, self.color)

        return move
