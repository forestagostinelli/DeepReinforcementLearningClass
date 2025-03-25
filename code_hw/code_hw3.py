from typing import List, Tuple, Dict, Optional
from environments.environment_abstract import Environment, State
import torch
from torch import nn
import numpy as np


def search(env: Environment, state_start: State, nnet: nn.Module) -> Optional[List[int]]:
    """ Find paths from start state to goal using trained DQN

    :param env: environment
    :param state_start: starting state
    :param nnet: trained DQN

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    """
    pass
