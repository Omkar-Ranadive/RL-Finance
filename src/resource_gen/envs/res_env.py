import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from constants import META_PATH, DATA_PATH, HIST_PATH
from datetime import datetime
import pandas as pd
from heapq import heapify, heappush, heappop
from collections import defaultdict
import bisect


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

        # Convert everything to lists/numpy for faster processing
        self.price_arr = df['price'].to_numpy()
        self.time_arr = df['timestamp'].to_numpy()
        self.cur_time_step = 0
        self.max_time_steps = self.price_arr.shape[0]
        self.stocks = df['symbol'].to_list()
        self.sides = df['side'].to_list()
        self.num_shares = df['size'].to_list()

        # Initialize a order-book in the following form:
        # Stock Symbol -> [Buyer List (Bid price), Seller List (Ask price)]
        # Buyer's list is sorted highest to lowest, Seller's list is from lowest to highest
        self.order_book = {}
        self.stock_to_index = {}
        self.bi, self.si, self.bd, self.sd = 0, 1, 2, 3   # For easier referencing
        for stock in self.stocks:
            # The last two entries are bid dict and ask dict
            # They will be of the form price -> available_volume
            self.order_book[stock] = [[], [], {}, {}]

        # Initialize feature matrices which will be used as state representation
        self.entries = 10   # Only top n entries of the order book will be passed as state
        num_stocks = len(set(self.stocks))
        additional_features = 3  # This num determines num of features like spread, volume etc
        # Multiply by -1.0 to denote empty values. The last dimension is 2 for buyer and seller
        self.ob_arr = -1.0*np.ones((num_stocks, self.entries, 2), dtype=np.float32)
        # TODO Add differential features like change in volume etc
        """
        Features: 
        0: Spread
        1: Total Volume per stock for buyer's side 
        2: Total volume per stock for seller's side 
        
        """
        self.f_arr = -1.0*np.ones((num_stocks, additional_features), dtype=np.float32)

        """
        Matrix to keep track of stocks the agent has bought 
        Features: 
        0: Num shares 
        1: Total spent on buying that share 
        """
        self.portfolio = np.zeros((num_stocks, 2))

        # Now, the order book keys need to be mapped to indices
        for index, k in enumerate(self.order_book.keys()):
            self.stock_to_index[k] = index

        # Pre-fill the order-book for first n steps
        self._update_order_book(time_steps=1000)
        # print(self.order_book)
        self._update_features()

    def _update_order_book(self, time_steps=5):
        """
        Update the limit order book by the number of time steps specified
        Args:
            time_steps (int): Number of time steps to move forward.
        """
        for i in range(time_steps):
            price = self.price_arr[self.cur_time_step]
            stock = self.stocks[self.cur_time_step]
            side = self.sides[self.cur_time_step]
            avail_vol = self.num_shares[self.cur_time_step]

            if side == 'B':
                if price not in self.order_book[stock][self.bd]:
                    bisect.insort(self.order_book[stock][self.bi], price)
                if avail_vol == 0:
                    # print("Stock {} Price {} Avail Vol {} Time Step {}".format(stock, price,
                    #                                                            avail_vol, self.cur_time_step))
                    del self.order_book[stock][self.bd][price]
                    # Pop the last entry (largest) in case of bid price
                    self.order_book[stock][self.bi].pop(-1)
                else:
                    self.order_book[stock][self.bd][price] = avail_vol

            else:
                if price not in self.order_book[stock][self.sd]:
                    bisect.insort(self.order_book[stock][self.si], price)
                if avail_vol == 0:
                    del self.order_book[stock][self.sd][price]
                    # Pop the first entry (smallest) in case of ask price
                    self.order_book[stock][self.si].pop(0)
                else:
                    self.order_book[stock][self.sd][price] = avail_vol

            self.cur_time_step += 1

    def _update_features(self):

        for stock, index in self.stock_to_index.items():
            # Remember buy prices should sorted in descending order; so reverse the list
            buyer_info = self.order_book[stock][self.bi][::-1]
            end_bi = min(len(buyer_info), self.entries)

            seller_info = self.order_book[stock][self.si]
            end_si = min(len(seller_info), self.entries)

            # Update the order book array using the top n entries of the order book for that stock
            self.ob_arr[index, :end_bi, 0] = buyer_info[:end_bi]
            self.ob_arr[index, :end_si, 1] = seller_info[:end_si]

            # Calculate the spread (highest buy (bid) price - lowest sell (ask) price)
            highest_bid = self.ob_arr[index, 0, 0]
            lowest_ask = self.ob_arr[index, 0, 1]
            if highest_bid != -1 and lowest_ask != -1:
                self.f_arr[index, 0] = highest_bid - lowest_ask

            # Store the total volume available per stock
            self.f_arr[index, 1] = sum(self.order_book[stock][self.bd].values())
            self.f_arr[index, 2] = sum(self.order_book[stock][self.sd].values())

    def step(self, action, stock_mat):
        """

        # TODO Doesn't include fractional shares yet -> Will make the problem too difficult
        # TODO May want to include it later
        Args:
            action (int): One of the three actions 0 = Buy, 1 = Sell, 2 = Hold
                          Note: Agent is only allowed to buy/sell using market orders.
                          So, if buy, order is bought using lowest ask (sell) price.
                          If sell, then order is sold using highest bid (buy) price.
            stock_mat (nd.array): Stock matrix denoting which stocks to purchase
        Returns:

        """
        step_count = 5
        reward = 0
        if action == 0:
            stocks_to_purchase = np.where(stock_mat == 1)[0]
            prices = self.ob_arr[stocks_to_purchase, 0, 1]  # Get the lowest ask prices
            # Only one share per stock can be bought for now, so just +=1 for every market order
            self.portfolio[stocks_to_purchase, 0] += 1
            self.portfolio[stocks_to_purchase, 1] += prices

        elif action == 1:
            stocks_to_sell = np.where(stock_mat == 1)[0]
            prices = self.ob_arr[stocks_to_sell, 0, 0]  # Get the highest bid price
            # print("Selling price: ", prices)
            total_shares = self.portfolio[stocks_to_sell, 0]
            reward_mat = total_shares*prices - self.portfolio[stocks_to_sell, 1]
            reward = np.mean(reward_mat)

            # Clear the portfolio for those stocks
            self.portfolio[stocks_to_sell, 0] = 0  # Assuming all shares of that stock are sold
            self.portfolio[stocks_to_sell, 1] = 0

        self._update_order_book(time_steps=step_count)

        state = [self.ob_arr, self.f_arr]

        return state, reward

    def get_state_info(self):
        state = [self.ob_arr, self.f_arr]
        return state

    def view_portfolio(self):
        for k, v in self.stock_to_index.items():
            if self.portfolio[v, 0] > 0:
                print("Index: {}, Stock name: {} Num Shares: {} Total Spent: {}".format(v, k,
                                                                       self.portfolio[v, 0],
                                                                       self.portfolio[v, 1]))

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    ResEnv()
