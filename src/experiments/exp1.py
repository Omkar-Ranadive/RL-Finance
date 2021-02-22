"""
Basic test of PFRL framework
"""

import gym
import resource_gen
from agents import agent_base, agent_v0
import numpy as np
from datasets import ReplayBuffer
import pfrl
import torch
import torch.nn
from agents import agent_v1

env = gym.make("res-env-v0")
buffer = ReplayBuffer(max_items=5)
env_params = {'env': env, 'max_steps': 15, 'epsilon': 0.1, 'buffer': buffer}
q_func = agent_v1.LobNet()

optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-4)

gamma = 0.9

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.6, random_action_func=env.action_space_d.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# As PyTorch only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Set the device id to use GPU. To use CPU only, set it to -1.
gpu = 0

# Now create an agent that will interact with the environment.
agent = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    gpu=gpu,
)


n_episodes = 1
max_episode_len = 10000


for i in range(1, n_episodes + 1):

    R = 0  # return (sum of rewards)
    t = 0  # time step
    state = env.get_state_info()
    num_stocks = state[0].shape[0]

    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act(state)
        random_indices = np.random.choice(np.arange(430), 5,
                                          replace=False)
        stock_mat = np.zeros((num_stocks))
        stock_mat[random_indices] = 1

        obs, reward = env.step(action, stock_mat)
        # print("Action: {}, Reward: {}".format(action, reward))
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, False, reset)
        if reset:
            break
        if t % 500 == 0:
            print('episode:', i, 'R:', R)
        if t % 500 == 0:
            print('statistics:', agent.get_statistics())
print('Finished.')
