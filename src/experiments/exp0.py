import gym
import resource_gen
from agents import agent_base, agent_v0
import numpy as np
from datasets import ReplayBuffer

env = gym.make("res-env-v0")
#
# agent = agent_base.AgentBase(env=env, max_steps=3)
#
# agent.fit()
# #


buffer = ReplayBuffer(max_items=5)
env_params = {'env': env, 'max_steps': 15, 'epsilon': 0.1, 'buffer': buffer}

agent = agent_v0.SimpleAgent(env_params)
agent.fit()




# ob_arr, f_arr = env.get_state_info()
# num_stocks = ob_arr.shape[0]
# to_buy = np.zeros((num_stocks))
# to_buy[[5, 6, 7]] = 1
#
# env.step(action=0, stock_mat=to_buy)
# env.step(action=0, stock_mat=to_buy)
# env.step(action=1, stock_mat=to_buy)