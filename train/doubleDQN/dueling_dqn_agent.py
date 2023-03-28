import numpy as np
import torch as T
from deep_q_network import DuelingDeepQNetwork
from replay_memory import ReplayBuffer
from line_profiler_pycharm import profile
from engine.Board import Board
from myenv import MyEnv


class DuelingDQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min, eps_dec,
                 replace, learn_amount, seed=None, checkpoint_dir=None,
                 invalid_moves_enabled=False, color="white"):
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

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=str(self.seed) + '_q_eval',
                                          chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=str(self.seed) + '_q_next',
                                          chkpt_dir=self.chkpt_dir)

        self.color = color

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)  # staci volat tato funkcia asi

        states = T.tensor(state).to(self.q_eval.device)  # lepsie rovno do konstruktoru
        rewards = T.tensor(reward).to(
            self.q_eval.device)  # v momente ako bude replay buffer cely v torchi, tieto riadky uz nebude treba
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def choose_action(self, observation, e):
        if self.invalid_moves_enabled:
            if np.random.random() > self.epsilon:
                state = np.array([observation], copy=False, dtype=np.float32)  # torch
                state_tensor = T.tensor(state).to(self.q_eval.device)
                _, advantages = self.q_eval.forward(state_tensor)

                best_action = T.argmax(advantages).item()
            else:
                best_action = np.random.choice(self.action_space)  # torch

            return best_action

        else:
            if np.random.random() > self.epsilon:
                state = np.array([observation], copy=False, dtype=np.float32)  # torch
                state_tensor = T.tensor(state).to(self.q_eval.device)
                _, advantages = self.q_eval.forward(state_tensor)

                _, valid_actions = e.board.find_possible_moves(self.color)
                valid_advantages = T.full((1,128), -np.inf)
                for i in valid_actions:
                    valid_advantages[0,i] = advantages[0, i]
                best_action = T.argmax(valid_advantages).item()
            else:
                _, valid_actions = e.board.find_possible_moves(self.color)
                best_action = np.random.choice(valid_actions)

            return best_action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
                self.learn_step_counter % self.replace_target_cnt == 0:  # skusit experimentovat s inou hodnotou replace (menej casto)
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    # @profile
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:  # iba na zaciatku
            return

        for i in range(self.learn_amount):
            self.q_eval.optimizer.zero_grad()

            self.replace_target_network()

            states, actions, rewards, states_, dones = self.sample_memory()

            V_s, A_s = self.q_eval.forward(states)

            states_ = self.flip_tensor_values(states_)

            V_s_, A_s_ = self.q_next.forward(states_)

            if not self.invalid_moves_enabled:
                for advantage in range(len(A_s_)):
                    board = Board(states_[advantage])
                    _, valid_actions = board.find_possible_moves(self.color)
                    valid_advantages = T.full((1, 128), -np.inf)
                    for i in valid_actions:
                        valid_advantages[0, i] = A_s_[advantage, i]
                    A_s_[advantage] = valid_advantages

            indices = T.arange(self.batch_size)

            q_pred = T.add(V_s,
                           (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]  # ja mozem hrat nahodne
            q_next = T.add(V_s_,
                           (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]  # super predpokladame ze hra optimalne

            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next

            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
            loss.backward()
            self.q_eval.optimizer.step()
            self.learn_step_counter += 1

            self.decrement_epsilon()

    def flip_tensor_values(self, states_):
        states_[:, 1, :, :] *= -1
        return states_

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self, old_seed):
        self.q_eval.load_checkpoint(self.chkpt_dir + old_seed + "_q_eval")
        self.q_next.load_checkpoint(self.chkpt_dir + old_seed + "_q_next")
        print(self.chkpt_dir + old_seed + "_q_eval + loaded")
        print(self.chkpt_dir + old_seed + "_q_next + loaded")
