import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.agent_base import AgentBase


class SimpleAgent(AgentBase):
    def __init__(self, env):
        super(AgentBase).__init__(env=env)
        self.fc1 = nn.Linear(1290, 512)
        self.fc2 = nn.Linear(512, 430)



