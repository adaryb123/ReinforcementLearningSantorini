from engine.Board import Board
from engine.Move import Move
import numpy as np
import torch as T
# from train.doubleDQN.deep_q_network import DuelingDeepQNetwork


class RLBot:
    def __init__(self, color, input_dims, n_actions, seed):
        self.color = color
        self.q_eval = DuelingDeepQNetwork(0.0001, n_actions,
                                          input_dims=input_dims,
                                          name=str(seed) + 'secondary_q_eval',
                                          chkpt_dir='../train/doubleDQN/models/')

        self.q_eval.load_checkpoint('../train/doubleDQN/models/' + seed + "_q_eval")

    def make_turn(self, environment):

        observation = environment.encode_input(environment.board)
        state = np.array([observation], copy=False, dtype=np.float32)  # torch
        state_tensor = T.tensor(state).to(self.q_eval.device)
        _, advantages = self.q_eval.forward(state_tensor)

        for number in range(len(advantages[0])):
            move = environment.create_move(number, self.color)
            valid, msg = environment.check_move_valid(move, e.board)
            if not valid:
                advantages[0, number] = -np.inf

        action = T.argmax(advantages).item()
        move = environment.create_move(action,self.color)

        return move
