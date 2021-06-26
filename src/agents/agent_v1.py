"""
Double DQN Agent using a stack of FCNs
"""

import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.agent_base import AgentBase
import numpy as np
from src.datasets import RLDataset
import torch.optim as optim
from pfrl.policies import SoftmaxCategoricalHead


class LobNet(nn.Module):
    def __init__(self, num_stocks):
        super(LobNet, self).__init__()
        self.fc1 = nn.Linear(num_stocks * 10 * 2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        concated_flen = 1024+num_stocks*3+num_stocks*5
        self.fc3 = nn.Linear(concated_flen, num_stocks)
        self.fc4 = nn.Linear(num_stocks, 3)

    def forward(self, state):
        # Convert order book to another representation using FCN and concatenate with f_arr
        order_book, f_arr, portfolio = state[0], state[1], state[2]
        order_book = order_book.view(-1, self.flatten_features(order_book))
        f_arr = f_arr.view(-1, self.flatten_features(f_arr))
        portfolio = portfolio.view(-1, self.flatten_features(portfolio))
        x = F.relu(self.fc1(order_book))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, f_arr), dim=1)
        x = torch.cat((x, portfolio), dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return pfrl.action_value.DiscreteActionValue(x)

    def forward_2(self, state):
        # This function returns the stock matrix, while the forward func returns the q-values (
        # actions)
        with torch.no_grad():
            order_book, f_arr, portfolio = torch.from_numpy(state[0]).unsqueeze(0), \
                                           torch.from_numpy(state[1]).unsqueeze(0), \
                                           torch.from_numpy(state[2]).unsqueeze(0)
            order_book = order_book.view(-1, self.flatten_features(order_book)).to("cuda")
            f_arr = f_arr.view(-1, self.flatten_features(f_arr)).to("cuda")
            portfolio = portfolio.view(-1, self.flatten_features(portfolio)).to("cuda")

            x = F.relu(self.fc1(order_book))
            x = F.relu(self.fc2(x))
            x = torch.cat((x, f_arr), dim=1)
            x = torch.cat((x, portfolio), dim=1)
            x_stocks = torch.sigmoid(self.fc3(x))
            return x_stocks.detach().cpu().numpy().squeeze()

    @staticmethod
    def flatten_features(x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features


class LobNet2(nn.Module):
    def __init__(self, num_stocks):
        super(LobNet2, self).__init__()
        self.seq1 = nn.Sequential(nn.Linear(num_stocks * 10 * 2, 2048),
                             nn.ReLU(),
                             nn.Linear(2048, 1024))

        concated_flen = 1024+num_stocks*3+num_stocks*5

        self.seq2 = nn.Sequential(nn.Linear(concated_flen, num_stocks))

        self.seq3 = nn.Sequential(
            nn.ReLU(),
            pfrl.nn.Branched(
                nn.Linear(num_stocks, 3),
                SoftmaxCategoricalHead()
            ),
            nn.Linear(num_stocks, 1),
        )

    def forward(self, state):
        # Convert order book to another representation using FCN and concatenate with f_arr
        order_book, f_arr, portfolio = state[0], state[1], state[2]
        order_book = order_book.view(-1, self.flatten_features(order_book))
        f_arr = f_arr.view(-1, self.flatten_features(f_arr))
        portfolio = portfolio.view(-1, self.flatten_features(portfolio))

        x = self.seq1(order_book)
        x = torch.cat((x, f_arr), dim=1)
        x = torch.cat((x, portfolio), dim=1)

        x = self.seq2(x)

        return self.seq3(x)

    def forward_2(self, state):
        with torch.no_grad():
            order_book, f_arr, portfolio = state[0], state[1], state[2]
            order_book = order_book.view(-1, self.flatten_features(order_book))
            f_arr = f_arr.view(-1, self.flatten_features(f_arr))
            portfolio = portfolio.view(-1, self.flatten_features(portfolio))

            x = self.seq1(order_book)
            x = torch.cat((x, f_arr), dim=1)
            x = torch.cat((x, portfolio), dim=1)

            return self.seq2(x)

    @staticmethod
    def flatten_features(x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features

