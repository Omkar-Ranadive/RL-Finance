import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from constants import META_PATH
from datetime import datetime
import pandas_datareader.data as web


class ResEnv(gym.Env):

    def __init__(self):
        # Continuous action space between 0 and 1
        self.action_space = spaces.Box(low=0, high=1.0, shape=(1, ))

        # # Load required meta info
        # av_key = open(META_PATH / 'alpha_vantage_key.txt', 'r').read()
        #
        # # Set up day wise stock info as state for now
        # self.df = web.DataReader("AAPL", "av-daily", start=datetime(2017, 2, 9), end=datetime(
        #     2017, 5, 24), api_key=av_key)



        # self.temporal_data = self.df['open'].to_numpy()
        self.temporal_data = np.random.normal(150.6, 3, 1000)
        self.cur_time_step = 0
        self.max_time_steps = self.temporal_data.shape[0]
        self.num_shares = 0

    def step(self, action):
        self.shares_bought = action / self.temporal_data[self.cur_time_step]
        self.num_shares += self.shares_bought
        print("Shares bought {}".format(self.shares_bought))
        print("Total shares {}".format(self.num_shares))
        self.cur_time_step = (self.cur_time_step + 1) % self.max_time_steps

        state = [self.temporal_data[self.cur_time_step], self.num_shares]

        return state

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    ResEnv()