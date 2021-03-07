"""
A2C Agent

"""

import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pfrl.agents import a2c
from pfrl.policies import SoftmaxCategoricalHead


