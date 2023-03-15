from engine.Board import Board
from engine.Move import Move
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
    def make_turn(self, environment):

        if self.counter % self.checkpoint_frequency == 0:
            self.reload_network()
        self.counter += 1

        observation = environment.encode_input(environment.board)
        state = np.array([observation], copy=False, dtype=np.float32)  # torch
        state_tensor = T.tensor(self.flip_tensor_values(state)).to(self.q_eval.device)
        _, advantages = self.q_eval.forward(state_tensor)

        for number in range(len(advantages[0])):
            move = environment.create_move(number, self.color)
            valid, msg = environment.check_move_valid(move, environment.board)
            if not valid:
                advantages[0, number] = -np.inf

        action = T.argmax(advantages).item()
        move = environment.create_move(action, self.color)

        return move
