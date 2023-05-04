"""
Double deep Q learning agent
Author: Adam Rybansky (xryban00)
FIT VUT 2023
Based on: https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DuelingDQN
"""

import numpy as np
import torch as T
from replay_memory import ReplayMemory
from engine.Board import decode_board
import deep_q_network_2x8
import deep_q_network_2x32
import deep_q_network_4x8
import deep_q_network_1linear

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min, eps_dec,
                 replace, learn_amount, seed=None, checkpoint_dir=None,
                 invalid_moves_enabled=False, network="2X8",
                 epsilon_softmax=False, adamw_optimizer=False, dropout=False):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.seed = seed
        self.chkpt_dir = checkpoint_dir
        self.learn_amount = learn_amount
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.invalid_moves_enabled = invalid_moves_enabled
        self.epsilon_softmax = epsilon_softmax
        self.adamw_optimizer = adamw_optimizer

        self.memory = ReplayMemory(mem_size, input_dims, n_actions)

        if network=="2X8":
            self.q_eval = deep_q_network_2x8.DeepQNetwork(self.lr, self.n_actions,
                                                          input_dims=self.input_dims,
                                                          name=str(self.seed) + '_q_eval',
                                                          chkpt_dir=self.chkpt_dir,
                                                          adamw_optimizer=self.adamw_optimizer,
                                                          dropout=dropout)
            self.q_next = deep_q_network_2x8.DeepQNetwork(self.lr, self.n_actions,
                                                          input_dims=self.input_dims,
                                                          name=str(self.seed) + '_q_next',
                                                          chkpt_dir=self.chkpt_dir,
                                                          adamw_optimizer=self.adamw_optimizer,
                                                          dropout=dropout)
        elif network=="4X8":
            self.q_eval = deep_q_network_4x8.DeepQNetwork(self.lr, self.n_actions,
                                                          input_dims=self.input_dims,
                                                          name=str(self.seed) + '_q_eval',
                                                          chkpt_dir=self.chkpt_dir)
            self.q_next = deep_q_network_4x8.DeepQNetwork(self.lr, self.n_actions,
                                                          input_dims=self.input_dims,
                                                          name=str(self.seed) + '_q_next',
                                                          chkpt_dir=self.chkpt_dir)
        elif network=="2X32":
            self.q_eval = deep_q_network_2x32.DeepQNetwork(self.lr, self.n_actions,
                                                           input_dims=self.input_dims,
                                                           name=str(self.seed) + '_q_eval',
                                                           chkpt_dir=self.chkpt_dir)
            self.q_next = deep_q_network_2x32.DeepQNetwork(self.lr, self.n_actions,
                                                           input_dims=self.input_dims,
                                                           name=str(self.seed) + '_q_next',
                                                           chkpt_dir=self.chkpt_dir)
        elif network=="1LINEAR":
            self.q_eval = deep_q_network_1linear.DeepQNetwork(self.lr, self.n_actions,
                                                              input_dims=self.input_dims,
                                                              name=str(self.seed) + '_q_eval',
                                                              chkpt_dir=self.chkpt_dir)
            self.q_next = deep_q_network_1linear.DeepQNetwork(self.lr, self.n_actions,
                                                              input_dims=self.input_dims,
                                                              name=str(self.seed) + '_q_next',
                                                              chkpt_dir=self.chkpt_dir)
        else:
            print("invalid dqn network parameter")
            exit(0)

    def choose_action(self, observation, e):
        """ Choose an action based on the observation of the environment"""

        if self.invalid_moves_enabled:
            if np.random.random() > self.epsilon:
                """ Make best move"""
                state = np.array([observation], copy=False, dtype=np.float32)
                state_tensor = T.tensor(state).to(self.q_eval.device)
                _, advantages = self.q_eval.forward(state_tensor)

                best_action = T.argmax(advantages).item()

            else:
                if self.epsilon_softmax:
                    """ Make random move based on softmax probabilities"""
                    state = np.array([observation], copy=False, dtype=np.float32)
                    state_tensor = T.tensor(state).to(self.q_eval.device)
                    _, advantages = self.q_eval.forward(state_tensor)
                    probabilities = T.nn.functional.softmax(advantages[0], dim=0)
                    best_action = np.random.choice(self.action_space, p=probabilities.cpu().detach().numpy())
                else:
                    """ Make entirely random  move"""
                    best_action = np.random.choice(self.action_space)

        else:
            """ Only valid moves are enabled"""
            if np.random.random() > self.epsilon:
                """ Make best move"""
                state = np.array([observation], copy=False, dtype=np.float32)
                state_tensor = T.tensor(state).to(self.q_eval.device)
                _, advantages = self.q_eval.forward(state_tensor)

                for action in range(len(advantages[0])):        # set all invalid moves to -np.inf
                    move = e.board.create_move_from_number(action, e.primary_player_color)
                    valid, msg = e.board.check_move_valid(move)
                    if not valid:
                        advantages[0, action] = -np.inf

                best_action = T.argmax(advantages).item()
            else:
                """ Make random movee from valid moves"""
                valid_moves = []
                for action in range(len(self.action_space)):     # set all invalid moves to -np.inf
                    move = e.board.create_move_from_number(action, e.primary_player_color)
                    valid, msg = e.board.check_move_valid(move)
                    if valid:
                        valid_moves.append(action)

                best_action = np.random.choice(valid_moves)

        return best_action

    def learn(self):
        """Get moves from replay buffer, compute Q values, backpropagate loss"""
        if self.memory.mem_cntr < self.batch_size:
            return

        for i in range(self.learn_amount):
            self.q_eval.optimizer.zero_grad()

            self.replace_target_network()

            states, actions, rewards, states_, dones = self.sample_memory()

            V_s, A_s = self.q_eval.forward(states)

            states_ = self.flip_tensor_values(states_)

            V_s_, A_s_ = self.q_next.forward(states_)

            if not self.invalid_moves_enabled:
                for advantage in range(len(A_s_)):       # set all invalid moves to -np.inf
                    board = decode_board(states_[advantage])
                    for action in range(len(A_s_[0])):
                        move = board.create_move_from_number(action, "white")
                        valid, msg = board.check_move_valid(move)
                        if not valid:
                            A_s_[advantage, action] = -np.inf

            indices = T.arange(self.batch_size)

            q_pred = T.add(V_s,
                           (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
            q_next = T.add(V_s_,
                           (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]

            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next

            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
            loss.backward()
            self.q_eval.optimizer.step()
            self.learn_step_counter += 1

            self.decrement_epsilon()


    def store_transition(self, state, action, reward, state_, done):
        """ Store move to the replay memory"""
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        """ Get a batch of moves from the replay memory"""
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(
            self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        """Copy evaluation network to target network"""
        if self.replace_target_cnt is not None and \
                self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def flip_tensor_values(self, states_):
        states_[:, 1, :, :] *= -1
        return states_

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self, old_seed):
        self.q_eval.load_checkpoint(self.chkpt_dir + old_seed + "_q_eval")
        self.q_next.load_checkpoint(self.chkpt_dir + old_seed + "_q_next")
        print(self.chkpt_dir + old_seed + "_q_eval loaded")
        print(self.chkpt_dir + old_seed + "_q_next loaded")