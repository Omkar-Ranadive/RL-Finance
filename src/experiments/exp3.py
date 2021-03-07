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

state = env.get_state_info()

print(state[0])
print(state[1])
print(state[2])