import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from constants import META_PATH, DATA_PATH, HIST_PATH
from datetime import datetime
import pandas as pd
from heapq import heapify, heappush, heappop
from collections import defaultdict


class ResEnv(gym.Env):

    def __init__(self):
        # Continuous action space between 0 and 1
        self.action_space_c = spaces.Box(low=0, high=1.0, shape=(1, ))
        self.action_space_d = spaces.Discrete(3)
        """
        0 = Buy 
        1 = Sell 
        2 = Do nothing 
        """
        df = pd.read_csv(HIST_PATH / 'processed_DEEP_29_11_2020_02_05.csv')
        # Convert timestamp to date time object
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(df.head())
        print(df.dtypes)
        self.price_arr = df['price'].to_numpy()
        self.time_arr = df['timestamp'].to_numpy()
        self.cur_time_step = -1
        self.max_time_steps = self.price_arr.shape[0]
        #
        # df_temp = df[(df['symbol'] == 'BABA') & (df['side'] == 'B')]
        # print(df_temp[['price', 'side', 'size', 'timestamp']].head(500))
        # return
        # Initialize a order-book in the following form:
        # Key -> [Buyer Max Heap, Seller Min Heap]
        self.stocks = df['symbol'].to_list()
        self.sides = df['side'].to_list()
        self.num_shares = df['size'].to_list()

        order_book = {}
        # These two dict's will keep the current list of prices in the book
        buyers_dict = {}
        sellers_dict = {}
        init_steps = 100

        # Initialize the order book
        for stock in set(self.stocks):
            l1, l2 = [], []
            heapify(l1)
            heapify(l2)
            order_book[stock] = [l1, l2]

        # Fill the order-book for initial n steps
        while self.cur_time_step < init_steps:
            price = self.price_arr[self.cur_time_step]
            stock = self.stocks[self.cur_time_step]
            side = self.sides[self.cur_time_step]
            available_shares = self.num_shares[self.cur_time_step]
            # 0th index in list is the buyer's heap and 1st index is the seller's heap
            if side == 'B':
                heap_in = 0
                price = -price  # For max-heap invert the price
                if price not in buyers_dict:
                    # Push the price change to order book
                    # We only consider aggregated shares, so only one copy exists for each price
                    heappush(order_book[stock][heap_in], price)

                buyers_dict[price] = available_shares
            else:
                heap_in = 1
                if not price in sellers_dict:
                    heappush(order_book[stock][heap_in], price)
                sellers_dict[price] = available_shares

            # When available shares = 0, that entry is removed from order book
            # Note, here we assume that only the top entry will possibly become zero
            # TODO Add checks to ensure that shares = 0 doesn't happen due to other reasons
            if available_shares == 0:
                heappop(order_book[stock][heap_in])
                if side == 'B':
                    del buyers_dict[price]
                else:
                    del sellers_dict[price]

            self.cur_time_step += 1

        print(order_book)

        # self.num_shares = 0

    def step(self, amount, action):
        ret_amt = 0
        if action == 0:
            shares_bought = amount / self.price_arr[self.cur_time_step]
            self.num_shares += shares_bought
            print("Shares bought {}".format(shares_bought))
            print("Total shares {}".format(self.num_shares))
        elif action == 1:
            ret_amt = self.price_arr[self.cur_time_step]*amount
            self.num_shares -= amount
            print("Shares sold {}, Amount gained {}".format(amount, ret_amt))
            print("Total shares {}".format(self.num_shares))

        self.cur_time_step = (self.cur_time_step + 1) % self.max_time_steps

        state = [self.price_arr[self.cur_time_step], self.num_shares, ret_amt]

        return state

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    ResEnv()
