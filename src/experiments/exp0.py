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


# buffer = ReplayBuffer(max_items=5)
# env_params = {'env': env, 'max_steps': 15, 'epsilon': 0.1, 'buffer': buffer}
#
# agent = agent_v0.SimpleAgent(env_params)
# agent.fit()




state = env.get_state_info()
ob_arr, f_arr, portfolio = state
print(ob_arr.shape, f_arr.shape, portfolio.shape)
num_stocks = ob_arr.shape[0]
to_buy = np.zeros((num_stocks))
to_buy[[5, 6, 7]] = 1

state, reward = env.step(action=0, stock_mat=to_buy)

print(state[0].shape, state[1].shape, state[2].shape)
state, reward = env.step(action=0, stock_mat=to_buy)
print(state[0].shape, state[1].shape, state[2].shape)
state, reward = env.step(action=1, stock_mat=to_buy)
print(state[0].shape, state[1].shape, state[2].shape)
state, reward = env.step(action=2, stock_mat=to_buy)
print(state[0].shape, state[1].shape, state[2].shape)
