import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.agent_base import AgentBase
import numpy as np
from src.datasets import RLDataset


class SimpleAgent(AgentBase):
    def __init__(self, env_params):
        super().__init__(**env_params)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert replay buffer to PyTorch dataloader
        self.dynamic_buffer = RLDataset(buffer=self.buffer, sample_size=1)
        self.dataloader = torch.utils.data.DataLoader(self.dynamic_buffer, batch_size=1)

        self.fc1 = nn.Linear(430*10*2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024+1290, 430)

    def choose_action(self):
        state, action, reward, new_state = self.buffer[-1]
        liquid_stocks = self._get_liquid_stocks(new_state[0])
        random_indices = np.random.choice(liquid_stocks, 5,
                                          replace=False)
        stock_mat = np.zeros((self.num_stocks))
        stock_mat[random_indices] = 1

        return self.env.action_space_d.sample(), stock_mat

    def update(self):
        for index, data in enumerate(self.dataloader):
            order_book, f_arr = data.state[0], data.state[1]
            print(order_book.shape, f_arr.shape)

      
        # lob = lob.view(-1, self.flatten_features(lob))
        # f_arr = f_arr.view(-1, self.flatten_features(f_arr))
        #
        # x = F.relu(self.fc1(lob))
        # x = F.relu(self.fc2(x))
        #
        # print(x.shape)

    @staticmethod
    def flatten_features(x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features




