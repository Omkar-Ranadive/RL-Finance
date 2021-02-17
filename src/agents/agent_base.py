import sys
import numpy as np
import constants
import torch
from src.datasets import ReplayBuffer


class AgentBase:
    def __init__(self,
                 env,
                 buffer,
                 episodes=1,
                 max_steps=100,
                 total_money=10000,
                 writer=None,
                 epsilon=constants.MAX_EPSILON,
                 decay_rate=0.0005,
                 gamma=0.9,
                 ):

        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps
        self.writer = writer
        self.epsilon = epsilon
        self.total_money = total_money
        self.num_shares = 0
        self.num_stocks = 0
        self.buffer = buffer
        self.total_reward = 0
        self.decay_rate = decay_rate
        self.gamma = gamma

    def fit(self):
        for ep in range(self.episodes):
            # At step 0, get the pre-initialized state from environment
            state = self.env.get_state_info()
            self.num_stocks = state[0].shape[0]
            self.total_reward = 0
            for step in range(1, self.max_steps):
                if np.random.uniform(0, 1) < self.epsilon or step < 2:
                    print("Random action chosen")
                    # TODO Fractional shares are not included for now
                    # fraction = self.env.action_space_c.sample()
                    action = self.env.action_space_d.sample()
                    # Choose 5 stocks to buy at random
                    liquid_stocks = self._get_liquid_stocks(state[0])
                    random_indices = np.random.choice(liquid_stocks, 5,
                                                      replace=False)
                    stock_mat = np.zeros((self.num_stocks))
                    stock_mat[random_indices] = 1

                else:
                    action, stock_mat = self.choose_action()
                    print("Action by agent")

                print("Action chosen: {}", action)

                new_state, reward = self.env.step(action=action, stock_mat=stock_mat)
                self.total_reward += reward
                print("Total Reward: ", self.total_reward)

                # Append data to replay buffer
                self.buffer.append(state, np.array(action, dtype=np.int64), np.array(reward,
                                    dtype=np.float32), new_state)

                state = new_state

                self.update()

                # Decay epsilon
                self.epsilon = max(0.001, self.epsilon-self.decay_rate)
                print("Decayed epsilon value: {}".format(self.epsilon))

                # self._get_liquid_stocks(state[0])

                # print("State: ")
                # print(state[0][:10])
                # print("-------"*15)
                # print(state[1][:10])
                # print(state[0].shape, state[1].shape)

            # self.env.view_portfolio()

    def choose_action(self):
        """
        This function must be implemented by subclass.
        Returns:
            action (int): the action to take
        """
        raise NotImplementedError()

    def update(self):
        """
        This function must be implemented by subclass.
        """
        raise NotImplementedError()

    def _get_liquid_stocks(self, ob):
        """
        Return indices of those stocks which can be bought.
        Stocks where the top of the book has a -1 in ask price can't be bought

        Args:
            ob (nd.array): Order book containing top n entries

        Returns (nd.array): Stock

        """
        valid_indices = np.where(ob[:, 0, 1] != -1)[0]

        return valid_indices
