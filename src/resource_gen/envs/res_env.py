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
from iex_parser import Parser, DEEP_1_0, TOPS_1_6


class ResEnv(gym.Env):

    def __init__(self, reader_it, allowed_types, allowed_stocks, init_steps=500):

        """

        Args:
            reader_it (iterator obj): Reader iterator to iterate directly over LOB data
                                      from IEX platform
            allowed_types (list): List of strs specifying which type of data to process (Refer
            LOB manual from IEX) - This is used in function _deep_reader()
            allowed_stocks (list):  List of stock symbols to consider Used in _deep_reader() ->
            Allowed stocks can be generated using the function in data_utils.py
            init_steps (int): Initialize the order book by init_steps
        """

        # Continuous action space between 0 and 1
        self.action_space_c = spaces.Box(low=0, high=1.0, shape=(1, ))
        self.action_space_d = spaces.Discrete(3)

        """
        0 = Buy 
        1 = Sell 
        2 = Do nothing 
        """
        # df = pd.read_csv(DATA_PATH / 'processed_DEEP_29_11_2020_02_05.csv')
        # # Convert timestamp to date time object
        # df['timestamp'] = pd.to_datetime(df['timestamp'])
        #
        # # Convert everything to lists/numpy for faster processing
        # self.price_arr = df['price'].to_numpy()
        # # Round off price_arr to 2 decimal places
        # self.price_arr = np.round(self.price_arr, 2)
        # self.time_arr = df['timestamp'].to_numpy()
        self.cur_time_step = 0
        # self.max_time_steps = self.price_arr.shape[0]
        # self.stocks = df['symbol'].to_list()
        # self.sides = df['side'].to_list()
        # self.num_shares = df['size'].to_list()
        self.roi = 0  # Used for calculating Sharpe ratio

        # Initializations for the DEEP (LOB) Processing
        self.allowed_types = allowed_types
        self.allowed_stocks = allowed_stocks

        # For data processing
        self.reader_it = reader_it

        # Initialize a order-book in the following form:
        # Stock Symbol -> [Buyer List (Bid price), Seller List (Ask price)]
        # Buyer's list is sorted highest to lowest, Seller's list is from lowest to highest
        self.order_book = {}
        self.stock_to_index = {}
        self.unique_stock_number = 0
        self.bi, self.si, self.bd, self.sd = 0, 1, 2, 3   # For easier referencing

        # Buffer to store messages in lookahead -> used to avoid iterator problems
        self.mbuffer = []

        # Initialize feature matrices which will be used as state representation
        self.entries = 10   # Only top n entries of the order book will be passed as state
        num_stocks = len(allowed_stocks)

        additional_features = 5  # This num determines num of features like spread, volume etc
        # Multiply by -1.0 to denote empty values. The last dimension is 2 for buyer and seller
        self.ob_arr = -1.0*np.ones((num_stocks, self.entries, 2), dtype=np.float32)
        # TODO Add differential features like change in volume etc
        """
        Features: 
        0: Spread
        1: Total Volume per stock for buyer's side 
        2: Total volume per stock for seller's side 
        3: Mid price = (Best Bid + Best ask) / 2 
        4: Microprice = (Bid Vol * Ask price) + (Ask Vol * Bid Price) / (Bid Vol + Ask Vol) 
        """

        self.f_arr = -1.0*np.ones((num_stocks, additional_features), dtype=np.float32)

        """
        Matrix to keep track of stocks the agent has bought 
        Features: 
        0: Num shares 
        1: Total spent on buying that share 
        2. Value of the shares at time step t 
        """
        self.portfolio = np.zeros((num_stocks, 3), dtype=np.float32)

        # Agent specific variables
        self.hold_thres = 10
        self.hold_counter = 0

        # # Now, the order book keys need to be mapped to indices
        # for index, k in enumerate(self.order_book.keys()):
        #     self.stock_to_index[k] = index

        # Pre-fill the order-book for first n steps
        self._update_order_book(time_steps=init_steps)
        # print(self.order_book)
        self._update_features()
        print("Initialization done!")

    def _update_order_book(self, time_steps=5):
        """
        Update the limit order book by the number of time steps specified
        Args:
            time_steps (int): Number of time steps to move forward.
        """
        for i in range(time_steps):
            # price = self.price_arr[self.cur_time_step]
            # stock = self.stocks[self.cur_time_step]
            # side = self.sides[self.cur_time_step]
            # avail_vol = self.num_shares[self.cur_time_step]

            # First, exhaust the messages gathered by performing the lookahead
            if len(self.mbuffer) > 0:
                message = self.mbuffer.pop(0)
            else:
                message = self._deep_reader()
            price = message['price']
            stock = message['symbol']
            side = message['side']
            avail_vol = message['size']

            if stock not in self.order_book:
                self._initialize_entries(stock)

            if side == 'B':
                if str(price) not in self.order_book[stock][self.bd]:
                    # print("New price value: (Buyer's side)")
                    bisect.insort(self.order_book[stock][self.bi], price)
                    self.order_book[stock][self.bd][str(price)] = avail_vol
                    # print(self.order_book[stock][self.bi])
                    # print(self.order_book[stock][self.bd])
                else:
                    # print("Else statement: Buyer's side")
                    self.order_book[stock][self.bd][str(price)] = avail_vol
                    # print(self.order_book[stock][self.bi])
                    # print(self.order_book[stock][self.bd])

                if avail_vol == 0:
                    # Note: In few cases entries other than the final entry is getting popped
                    p_index = -1
                    # print("Buyer's side: Volume = 0")
                    if str(self.order_book[stock][self.bi][-1]) != str(price):
                        # print("THIS IS THE CASE!")
                        p_index = self.order_book[stock][self.bi].index(price)

                    del self.order_book[stock][self.bd][str(price)]
                    # Pop the last entry (largest) in case of bid price
                    self.order_book[stock][self.bi].pop(p_index)
                    # print("After deleting")
                    # print(self.order_book[stock][self.bi])
                    # print(self.order_book[stock][self.bd])

            else:
                if str(price) not in self.order_book[stock][self.sd]:
                    # print("New price (Seller's side)")
                    bisect.insort(self.order_book[stock][self.si], price)
                    self.order_book[stock][self.sd][str(price)] = avail_vol
                    # print(self.order_book[stock][self.si])
                    # print(self.order_book[stock][self.sd])
                else:
                    # print("Else statement (Seller's side)")
                    self.order_book[stock][self.sd][str(price)] = avail_vol
                    # print(self.order_book[stock][self.si])
                    # print(self.order_book[stock][self.sd])

                if avail_vol == 0:
                    # Note: In few cases entries other than the final entry is getting popped
                    p_index = 0
                    if str(self.order_book[stock][self.si][0]) != str(price):
                        # print("THIS IS THE CASE!")
                        p_index = self.order_book[stock][self.si].index(price)

                    del self.order_book[stock][self.sd][str(price)]

                    # Pop the first entry (smallest) in case of ask price (except rare cases)
                    self.order_book[stock][self.si].pop(p_index)

                    # print("After deleting")
                    # print(self.order_book[stock][self.si])
                    # print(self.order_book[stock][self.sd])

            self.cur_time_step += 1

    def _update_features(self):

        for stock, index in self.stock_to_index.items():
            # Remember buy prices should sorted in descending order; so reverse the list
            buyer_info = self.order_book[stock][self.bi][::-1]
            end_bi = min(len(buyer_info), self.entries)

            seller_info = self.order_book[stock][self.si]
            end_si = min(len(seller_info), self.entries)

            # If end_bi / end_si is 0, then make entries = -1
            if end_bi == 0:
                self.ob_arr[index, :, 0] = -1
            if end_si == 0:
                self.ob_arr[index, :, 1] = -1
            else:
                # Update the order book array using the top n entries of the order book for that stock
                self.ob_arr[index, :end_bi, 0] = buyer_info[:end_bi]
                self.ob_arr[index, :end_si, 1] = seller_info[:end_si]

            # Store the total volume available per stock
            # print(self.order_book[stock][self.bd].values(), sum(self.order_book[stock][
            # self.bd].values()))
            self.f_arr[index, 1] = sum(self.order_book[stock][self.bd].values())
            self.f_arr[index, 2] = sum(self.order_book[stock][self.sd].values())

            # Calculate the spread (highest buy (bid) price - lowest sell (ask) price)
            highest_bid = self.ob_arr[index, 0, 0]

            lowest_ask = self.ob_arr[index, 0, 1]

            if highest_bid != -1 and lowest_ask != -1:
                self.f_arr[index, 0] = highest_bid - lowest_ask
                # Calculate mid price
                self.f_arr[index, 3] = (highest_bid + lowest_ask) / 2

                # Calculate micro price
                bid_vol = self.order_book[stock][self.bd][str(highest_bid)]
                ask_vol = self.order_book[stock][self.sd][str(lowest_ask)]
                total_vol = bid_vol + ask_vol

                self.f_arr[index, 4] = (bid_vol*lowest_ask + ask_vol*highest_bid)/total_vol

            # Calculate current value of stock for the stocks in portfolio
            # Get shares > 0
            user_shares = np.where(self.portfolio[:, 0] > 0)
            # Get current market price for them
            self.portfolio[user_shares, 2] = self.ob_arr[user_shares, 0, 0]

    def _lookahead_future(self, time_steps, order_book, ob_arr):
        """
        Lookahead into the future by time_steps number of steps
        This function will be used to decide how good a present action is (Ex - buying a stock)
        """

        for i in range(time_steps):
            # price = self.price_arr[self.cur_time_step]
            # stock = self.stocks[self.cur_time_step]
            # side = self.sides[self.cur_time_step]
            # avail_vol = self.num_shares[self.cur_time_step]
            message = self._deep_reader()
            self.mbuffer.append(message)
            price = message['price']
            stock = message['symbol']
            side = message['side']
            avail_vol = message['size']

            if stock not in self.order_book:
                order_book[stock] = [[], [], {}, {}]
                self._initialize_entries(stock)

            if side == 'B':
                if str(price) not in order_book[stock][self.bd]:
                    bisect.insort(order_book[stock][self.bi], price)
                    order_book[stock][self.bd][str(price)] = avail_vol

                else:
                    order_book[stock][self.bd][str(price)] = avail_vol

                if avail_vol == 0:
                    # Note: In few cases entries other than the final entry is getting popped
                    p_index = -1
                    if str(order_book[stock][self.bi][-1]) != str(price):
                        p_index = order_book[stock][self.bi].index(price)

                    del order_book[stock][self.bd][str(price)]
                    # Pop the last entry (largest) in case of bid price
                    order_book[stock][self.bi].pop(p_index)

            else:
                if str(price) not in order_book[stock][self.sd]:
                    bisect.insort(order_book[stock][self.si], price)
                    order_book[stock][self.sd][str(price)] = avail_vol

                else:
                    order_book[stock][self.sd][str(price)] = avail_vol

                if avail_vol == 0:
                    # Note: In few cases entries other than the final entry is getting popped
                    p_index = 0
                    if str(order_book[stock][self.si][0]) != str(price):
                        p_index = order_book[stock][self.si].index(price)

                    del order_book[stock][self.sd][str(price)]

                    # Pop the first entry (smallest) in case of ask price (except rare cases)
                    order_book[stock][self.si].pop(p_index)

        for stock, index in self.stock_to_index.items():
            # Remember buy prices should sorted in descending order; so reverse the list
            buyer_info = order_book[stock][self.bi][::-1]
            end_bi = min(len(buyer_info), self.entries)

            seller_info = order_book[stock][self.si]
            end_si = min(len(seller_info), self.entries)

            # If end_bi / end_si is 0, then make entries = -1
            if end_bi == 0:
                ob_arr[index, :, 0] = -1
            if end_si == 0:
                ob_arr[index, :, 1] = -1
            else:
                # Update the order book array using the top n entries of the order book for that stock
                ob_arr[index, :end_bi, 0] = buyer_info[:end_bi]
                ob_arr[index, :end_si, 1] = seller_info[:end_si]

        return ob_arr

    def _initialize_entries(self, stock):
        self.order_book[stock] = [[], [], {}, {}]
        self.stock_to_index[stock] = self.unique_stock_number
        self.unique_stock_number += 1

    def step(self, action, stock_mat):
        """

        # TODO Doesn't include fractional shares yet -> Will make the problem too difficult
        # TODO May want to include it later
        Args:
            action (int): One of the three actions 0 = Buy, 1 = Sell, 2 = Hold
                          Note: Agent is only allowed to buy/sell using market orders.
                          So, if buy, order is bought using lowest ask (sell) price.
                          If sell, then order is sold using highest bid (buy) price.
            stock_mat (nd.array): Stock matrix denoting which stocks to purchase/sell
        Returns:

        """
        step_count = 5
        lookahead_count = 50
        reward = 0
        # print("Reward at start: ", reward)
        if action == 0:
            # Get the valid stock mat
            stock_mat = self._get_liquid_stocks(stock_mat, self.ob_arr, 1-action)

            stocks_to_purchase = np.where(stock_mat == 1)[0]
            # If all stocks are invalid, then don't buy anything, return reward of 0
            if stocks_to_purchase.size != 0:
                prices = self.ob_arr[stocks_to_purchase, 0, 1]  # Get the lowest ask prices
                # Only one share per stock can be bought for now, so just +=1 for every market order
                self.portfolio[stocks_to_purchase, 0] += 1
                self.portfolio[stocks_to_purchase, 1] += prices

                # Calculate reward
                # If the ask price has gone down in the future, then buying right now was a bad move
                # Else it was a good move
                mean_purchase = np.mean(prices)
                ob_arr = self._lookahead_future(time_steps=lookahead_count,
                                                order_book=self.order_book.copy(),
                                                ob_arr=self.ob_arr.copy())

                # Make sure the stock purchases are still valid after update
                stock_mat = self._get_liquid_stocks(stock_mat, ob_arr, 1-action)
                stocks_to_purchase = np.where(stock_mat == 1)[0]
                if stocks_to_purchase.size != 0:
                    future_mean_purchase = np.mean(ob_arr[stocks_to_purchase, 0, 1])
                    # TODO If the stocks become invalid, then was purchasing them before a good idea?
                    # TODO If yes, how to incorporate this into the reward?
                    reward = future_mean_purchase - mean_purchase
                else:
                    # If the stocks are unavailable, it means they were sold
                    # Assuming this is a good thing - reward positively
                    reward = 5

            self.hold_counter = 0

        elif action == 1:
            # Get the valid stock mat
            stock_mat = self._get_liquid_stocks(stock_mat, self.ob_arr, 1-action)

            stocks_to_sell = np.where(stock_mat == 1)[0]
            prices = self.ob_arr[stocks_to_sell, 0, 0]  # Get the highest bid price
            # print("Selling price: ", prices)
            total_shares = self.portfolio[stocks_to_sell, 0]
            # If total shares are 0, then the stock doesn't exist in the portfolio,
            # If size == 0, then non of the stocks were liquid
            if total_shares.size != 0 and np.sum(total_shares) > 0:
                reward_mat = total_shares*prices - self.portfolio[stocks_to_sell, 1]
                reward = np.mean(reward_mat)
                # Calculate roi
                self.roi = (reward / np.mean(self.portfolio[stocks_to_sell, 1])) * 100
                # TODO Incorporate what happens if stocks were sold in the future instead

            # Clear the portfolio for those stocks
            self.portfolio[stocks_to_sell, 0] = 0  # Assuming all shares of that stock are sold
            self.portfolio[stocks_to_sell, 1] = 0
            self.hold_counter = 0

        elif action == 2:
            self.hold_counter += 1
            if self.hold_counter > self.hold_thres:
                reward = -10  # Penalize for holding stocks for too long
                # Our assumption is that the agent is a high frequency trader

        # Update the states
        self._update_order_book(time_steps=step_count)
        self._update_features()

        state = self._normalize_state(self.ob_arr.copy(), self.f_arr.copy(), self.portfolio.copy())
        # print("Reward: ", reward)
        return state, reward

    def get_state_info(self):
        return self._normalize_state(self.ob_arr.copy(), self.f_arr.copy(), self.portfolio.copy())

    def get_sell_info(self):
        return self.roi

    @staticmethod
    def _normalize_state(ob_arr, f_arr, portfolio, min_range=-1, max_range=1):
        # Normalize between -1, 1 range
        ob_arr_max = np.max(ob_arr)
        ob_arr_min = np.min(ob_arr)
        ob_arr = min_range + (ob_arr - ob_arr_min)*(max_range-min_range) / (ob_arr_max - ob_arr_min)

        f_max = np.max(f_arr)
        f_min = np.min(f_arr)
        f_arr = min_range + (f_arr - f_min) * (max_range - min_range) / (f_max - f_min)

        port_max = np.max(portfolio)
        port_min = np.min(portfolio)
        # Initially, portfolio is all zeros, avoid that case
        if port_max != port_min:
            portfolio = (portfolio - port_min) / (port_max - port_min)

        return [ob_arr, f_arr, portfolio]

    @staticmethod
    def _get_liquid_stocks(stock_mat, ob, action):
        """
        Make sure the entries which are 1 in stock matrix are liquid.
        Return a stock mat which is liquid
        Args:
            stock_mat (nd.array):  Stock matrix denoting which stocks to purchase
            ob (nd.array): Order book containing top n entries
            action (int): 0 = buy, 1 = sell, depending on this, choose the correct entry from
            order book

        Returns (nd.array): Valid stock matrix

        """
        invalid_indices = np.where(ob[:, 0, action] == -1)[0]
        stock_mat[invalid_indices] = 0

        return stock_mat

    def view_portfolio(self):
        for k, v in self.stock_to_index.items():
            if self.portfolio[v, 0] > 0:
                print("Index: {}, Stock name: {} Num Shares: {} Total Spent: {}".format(v, k,
                                                                       self.portfolio[v, 0],
                                                                       self.portfolio[v, 1]))

    def _deep_reader(self):
        """
        Iterate through the data directly without storing it in dataframes
        """
        message = next(self.reader_it)

        while message['type'] not in self.allowed_types or message['symbol'].decode('utf-8') not \
                in self.allowed_stocks:
            message = next(self.reader_it)

        # Convert symbols and sides from byte-literals to strings
        message['symbol'] = message['symbol'].decode('utf-8')
        message['side'] = message['side'].decode('utf-8')
        # Round off price to 2 decimal places
        message['price'] = np.round(float(message['price']), 2)
        return message

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    file = str(HIST_PATH / 'data_feeds_20201124_20201124_IEXTP1_DEEP1.0.pcap.gz')
    allowed_types = ['price_level_update']

    # process_hist_data(file=file, allowed_types=allowed, hist_type="DEEP", max_count=300000)
    reader = Parser(file, DEEP_1_0).__enter__()
    reader_it = iter(reader)

    ResEnv(reader_it=reader_it, allowed_types=allowed_types)
