"""
Deep Q network, variant with 4 convolutional layers and 8 channels
Author: Adam Rybansky (xryban00)
FIT VUT 2023
Based on: https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DuelingDQN
"""

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv4 = nn.Conv2d(8, 8, 3, padding=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 200)
        self.fc2 = nn.Linear(200, 200)
        self.V = nn.Linear(200, 1)
        self.A = nn.Linear(200, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        print("device= " + str(self.device))


    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        dims = self.conv4(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        conv_state = conv2.view(conv4.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        flat2 = F.relu(self.fc2(flat1))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file, map_location=self.device))
        # print(checkpoint_file + " loaded")
