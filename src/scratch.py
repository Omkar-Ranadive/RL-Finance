import gym
import resource_gen
from agents import agent_base

env = gym.make("res-env-v0")

agent = agent_base.AgentBase(env=env, max_steps=5)

agent.fit()

