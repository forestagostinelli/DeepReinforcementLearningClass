from typing import List
from environments.environment_abstract import State, Environment
from environments import env_utils
from torch import nn

from argparse import ArgumentParser

from code_hw.code_hw2 import deep_rl, greedy_action
import torch

import time


def follow_greedy_policy(env: Environment, state: State, nnet: nn.Module, num_steps: int) -> State:
    nnet.eval()
    for _ in range(num_steps):
        if env.is_terminal(state):
            return state

        action: int = greedy_action(env, state, nnet)
        state = env.sample_transition(state, action)[0]

    return state


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--rand', default=False, action='store_true', help="")

    args = parser.parse_args()
    torch.set_num_threads(1)

    # get environment
    env_name: str = "puzzle8"
    if args.rand:
        env_name = "puzzle8_rand"
    env: Environment = env_utils.get_environment(env_name)[0]

    start_time = time.time()

    # train nnet
    nnet: nn.Module = deep_rl(env)

    # run greedy
    print("---Testing---")
    num_test: int = 500
    states_test: List[State] = env.sample_start_states(num_test)
    num_solved: int = 0
    for state in states_test:
        state_end = follow_greedy_policy(env, state, nnet, 100)
        if env.is_terminal(state_end):
            num_solved += 1

    print(f"Solved: {num_solved}/{num_test} ({100.0 * num_solved/num_test:.2f}%) ")
    print("Time: %f (secs)" % (time.time() - start_time))
    # torch.save(value_net.state_dict(), "nnets/value_net.pt")

    print("DONE")


if __name__ == "__main__":
    main()
