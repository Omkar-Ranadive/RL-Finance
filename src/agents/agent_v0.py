import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.agent_base import AgentBase
import numpy as np
from src.datasets import RLDataset
import torch.optim as optim


class SimpleAgent(AgentBase):
    def __init__(self, env_params):
        super().__init__(**env_params)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert replay buffer to PyTorch dataloader
        self.dynamic_buffer = RLDataset(buffer=self.buffer, sample_size=1)
        self.dataloader = torch.utils.data.DataLoader(self.dynamic_buffer, batch_size=1)

        self.lobnet = LobNet()
        self.qnet = QNet()

        # Set the optimizer
        params = list(self.lobnet.parameters()) + list(self.qnet.parameters())
        self.optimizer = optim.Adam(params, lr=0.001)

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

            # Get valid stock indices
            valid_indices = self._get_liquid_stocks(order_book.cpu().numpy())


            x = self.lobnet(order_book, f_arr)
            x_stocks = F.softmax(x, dim=1)

            # Covert to 0/1 based on threshold
            x_stocks = (x_stocks > 0.7).int()

            # Mask the illiquid stocks
            stock_mat = x_stocks.cpu().numpy()
            all_indices = np.arange(start=0, stop=valid_indices.shape[0])
            invalid_indices = list(set(all_indices).difference(valid_indices))
            stock_mat[invalid_indices] = 0

            # Get the q-value
            q_values = self.qnet(x)
            q_value = torch.gather(q_values, dim=1, index=data.action.unsqueeze(0))

            with torch.no_grad():
                ob_next, f_arr_next = data.next_state[0], data.next_state[1]
                x_next = self.lobnet(ob_next, f_arr_next)
                q_value_next = self.qnet(x_next).max(1)[0].unsqueeze(-1)

            expected_q_value = data.reward + self.gamma * q_value_next

            loss = F.smooth_l1_loss(q_value, expected_q_value)

            # Optimize the model with backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("Worked!")


class LobNet(nn.Module):
    def __init__(self):
        super(LobNet, self).__init__()
        self.fc1 = nn.Linear(430 * 10 * 2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024 + 1290, 430)

    def forward(self, order_book, f_arr):
        # Convert order book to another representation using FCN and concatenate with f_arr
        order_book = order_book.view(-1, self.flatten_features(order_book))
        f_arr = f_arr.view(-1, self.flatten_features(f_arr))

        x = F.relu(self.fc1(order_book))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, f_arr), dim=1)
        x = self.fc3(x)
        return x

    @staticmethod
    def flatten_features(x):
        # Flatten all dimensions except the batch
        size = x.size()[1:]
        num_features = 1
        for dimension in size:
            num_features *= dimension

        return num_features


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        # Simple q-net
        self.fcq1 = nn.Linear(430, 3)

    def forward(self, x):
        q_values = F.leaky_relu(self.fcq1(x))
        q_values = F.softmax(q_values, dim=1)

        return q_values

