import numpy as np
import constants


class AgentBase:
    def __init__(self,
                 env,
                 episodes=1,
                 max_steps=100,
                 total_resource=1000,
                 writer=None,
                 ):

        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps
        self.writer = writer
        self.epsilon = constants.MAX_EPSILON
        self.total_resource = total_resource  # Example in this case would be initial money
        # which the agent starts with

    def fit(self):
        for ep in range(self.episodes):
            for step in range(self.max_steps):
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.env.action_space.sample()

                spent_res = action * self.total_resource
                self.total_resource -= spent_res
                print("Action taken {} Resource spent: {}, Resource left {}".format(action,
                                                                                 spent_res,
                                                                                    self.total_resource))
                stock_price, total_shares = self.env.step(action=spent_res)
                print("-"*15)

