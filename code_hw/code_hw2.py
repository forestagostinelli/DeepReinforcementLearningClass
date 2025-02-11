from environments.environment_abstract import Environment, State
from torch import nn


def greedy_action(env: Environment, state: State, nnet: nn.Module) -> int:
    """

    @param env: Environment
    @param state: the state
    @param nnet: neural network

    @return: a greedy action
    """
    pass


def deep_rl(env: Environment) -> nn.Module:
    """
    @param env: environment
    @return: the trained neural network
    """
    pass
