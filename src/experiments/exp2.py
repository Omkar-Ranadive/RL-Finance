"""
Double DQN Agent
Strategy:
Linear Epsilon Decay

"""


import gym
import resource_gen
from agents import agent_base_pfrl
import numpy as np
from datasets import ReplayBuffer
import pfrl
import torch
import torch.nn
from agents import agent_v1
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from constants import MODEL_PATH, HIST_PATH, DATA_PATH
from iex_parser import Parser, DEEP_1_0
from utils import load_file
import sys
import os

# Initializations for the DEEP (LOB) Processing
file = str(HIST_PATH / 'data_feeds_20201124_20201124_IEXTP1_DEEP1.0.pcap.gz')
allowed_types = ['price_level_update']
allowed_stocks_file = DATA_PATH / 'DEEP_2021_03_01-12_17_52_AM_T_100000_N_50'
allowed_stocks = load_file(filename=allowed_stocks_file)


# process_hist_data(file=file, allowed_types=allowed, hist_type="DEEP", max_count=300000)
reader = Parser(file, DEEP_1_0).__enter__()
reader_it = iter(reader)

# Env params
exp_id = "exp2_{}".format(datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))

env = gym.make("res-env-v0", reader_it=reader_it, allowed_types=allowed_types,
               allowed_stocks=allowed_stocks)


writer = SummaryWriter(MODEL_PATH / 'runs/{}'.format(exp_id))

# Model params
q_func = agent_v1.LobNet(num_stocks=len(allowed_stocks))
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-4)
gamma = 0.99
# Use epsilon-greedy for exploration
explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.1,
    decay_steps=50000, random_action_func=env.action_space_d.sample)
# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = 0

# Now create an agent that will interact with the environment.
agent_params = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=500,
    gpu=gpu,
)

agent = agent_base_pfrl.AgentBase(env=env,
                                  agent=agent_params,
                                  episodes=1,
                                  print_freq=1000,
                                  save_freq=10000,
                                  max_episode_len=100000,  # Put -1 for infinite steps
                                  exp_id=exp_id,
                                  writer=writer)

agent.fit()

