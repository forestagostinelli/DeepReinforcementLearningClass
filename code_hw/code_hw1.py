from typing import List, Dict, Tuple
from environments.environment_abstract import Environment, State
from numpy.typing import NDArray


def mrp(state_trans_probs: NDArray, state_rewards: NDArray, gamma: float) -> NDArray:
    """ Compute the value of every state in the Markov reward process (MRP)
    |S| represents the size of the state space
    :param state_trans_probs: a |S|x|S| matrix of state transition probabilities
    :param state_rewards: a |S|x1 matrix of expected rewards for every state
    :param gamma: discount factor
    :return: a |S|x1 matrix of state values
    """
    pass


def dynamic_programming(env: Environment, states: List[State], state_values: Dict[State, float],
                        gamma: float) -> Tuple[Dict[State, float], Dict[State, List[float]]]:
    """ Perform tabular dynamic programming to exactly compute the optimal value function and obtain an optimal policy

    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param gamma: the discount factor

    @return: the state value function and policy found by value iteration
    """
    pass


def model_free(env: Environment, action_values: Dict[State, List[float]], gamma: float) -> Dict[State, List[float]]:
    """ Perform model free tabular reinforcement learning to compute an action-value function
    @param env: environment
    @param action_values: dictionary that maps states to their action values (list of floats)
    @param gamma: the discount factor

    @return: the action value function
    """
    pass
