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
        self.num_shares = 0

    def fit(self):
        for ep in range(self.episodes):
            # At step 0, give the opening price of the stock at time_step 0
            state = self.env.step(-1, -1)
            for step in range(1, self.max_steps):
                if np.random.uniform(0, 1) < self.epsilon:
                    fraction = self.env.action_space_c.sample()
                    action = self.env.action_space_d.sample()
                else:
                    fraction = self.env.action_space_c.sample()
                    action = self.env.action_space_d.sample()

                if action == 0:
                    amount = fraction * self.total_resource
                    self.total_resource -= amount

                    print("Action taken {} Fraction {} Resource spent: {}".format(action,
                                                                                  fraction,
                                                                                  amount))

                elif action == 1:
                    amount = fraction * self.num_shares
                    print("Action taken {} Fraction : {}, Shares sold {}".format(action,
                                                                                    fraction,
                                                                                    amount))
                else:
                    amount = 0
                    print("Do nothing")

                stock_price, total_shares, ret_amt = self.env.step(amount=amount, action=action)
                self.num_shares = total_shares
                self.total_resource += ret_amt
                print("Current resource: {}, Total shares: {}".format(self.total_resource,
                                                                      self.num_shares))
                print("-"*15)

