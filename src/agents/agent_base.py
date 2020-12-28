import numpy as np
import constants


class AgentBase:
    def __init__(self,
                 env,
                 episodes=1,
                 max_steps=100,
                 total_money=10000,
                 writer=None,
                 ):

        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps
        self.writer = writer
        self.epsilon = constants.MAX_EPSILON
        self.total_money = total_money  #
        self.num_shares = 0
        self.num_stocks = 0

    def fit(self):
        for ep in range(self.episodes):
            # At step 0, get the pre-initialized state from environment
            state = self.env.get_state_info()
            self.num_stocks = state[0].shape[0]

            for step in range(1, self.max_steps):
                if np.random.uniform(0, 1) < self.epsilon:
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
                    action = self.choose_action()

                print("Action chosen: {}", action)

                state, reward = self.env.step(action=0, stock_mat=stock_mat)
                self._get_liquid_stocks(state[0])

                # print("State: ")
                # print(state[0][:10])
                # print("-------"*15)
                print(state[1][:10])
                print(state[0].shape, state[1].shape)

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

        Returns (nd.array): Array containing valid stock indices

        """
        valid_indices = np.where(ob[:, 0, 1] != -1)[0]

        return valid_indices
