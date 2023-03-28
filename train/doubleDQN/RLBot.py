import numpy as np
import torch as T
from deep_q_network import DuelingDeepQNetwork
import os


class RLBot:
    def __init__(self, color, input_dims, n_actions, seed, checkpoint_frequency):
        self.color = color
        self.chkpt_dir = 'models/'
        self.seed = seed
        self.counter = 0
        self.checkpoint_frequency = checkpoint_frequency
        lr = 0.0001
        self.q_eval = DuelingDeepQNetwork(lr, n_actions,
                                          input_dims=input_dims,
                                          name=str(self.seed) + 'secondary_q_eval',
                                          chkpt_dir=self.chkpt_dir)

    def reload_network(self):
        filename = self.chkpt_dir + self.seed + "_q_eval"
        if os.path.exists(filename):
            self.q_eval.load_checkpoint(self.chkpt_dir + self.seed + "_q_eval")
    def flip_tensor_values(self, states):
        states[:, 1, :, :] *= -1
        return states
    def make_turn(self, board):

        # if self.counter % self.checkpoint_frequency == 0:
        #     self.reload_network()
        # self.counter += 1

        observation = board.encode()
        state = np.array([observation], copy=False, dtype=np.float32)  # torch
        state_tensor = T.tensor(self.flip_tensor_values(state)).to(self.q_eval.device)
        _, advantages = self.q_eval.forward(state_tensor)

        _, valid_actions = board.find_possible_moves(self.color)
        valid_advantages = T.full((1, 128), -np.inf)
        for i in valid_actions:
            valid_advantages[0,i] = advantages[0,i]
        best_action = T.argmax(valid_advantages).item()
        move = board.create_move_from_number(best_action, self.color)

        return move
