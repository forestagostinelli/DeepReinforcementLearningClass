from typing import List, Optional, Tuple
from environments.environment_abstract import Environment, State
from environments import env_utils
import time
from argparse import ArgumentParser
import torch
from torch import nn

from code_hw.code_hw3 import search
import numpy as np
import pickle


class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim: int, layer_dims: List[int], layer_acts: List[str], batch_norms: List[bool]):
        super().__init__()
        self.layers: nn.ModuleList[nn.ModuleList] = nn.ModuleList()

        self.flatten_nn = nn.Flatten()

        # layers
        for layer_dim, act, batch_norm in zip(layer_dims, layer_acts, batch_norms):
            module_list = nn.ModuleList()

            # linear
            module_list.append(nn.Linear(input_dim, layer_dim))
            # module_list[-1].bias.data.zero_()

            if batch_norm:
                module_list.append(nn.BatchNorm1d(layer_dim))

            # activation
            if act.upper() == "RELU":
                module_list.append(nn.ReLU())
            elif act.upper() != "LINEAR":
                raise ValueError(f"Unknown activation function {act}")

            self.layers.append(module_list)

            input_dim = layer_dim

    def forward(self, x):
        module_list: nn.ModuleList
        for module_list in self.layers:
            for module in module_list:
                x = module(x)

        return x


def check_soln(env: Environment, state_start: State, actions: List[int]) -> Tuple[float, bool]:
    state: State = state_start
    path_cost: float = np.inf
    if actions is not None:
        # Get results
        path_cost: float = 0.0
        for action in actions:
            state, reward = env.sample_transition(state, action)
            path_cost += -reward

    if env.is_terminal(state):
        is_solved: bool = True
    else:
        is_solved: bool = False

    return path_cost, is_solved


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--grade', action='store_true', default=False, help="")

    args = parser.parse_args()
    env: Environment = env_utils.get_environment("puzzle8")[0]

    # get data
    dqn = FullyConnectedModel(81, [200, 100, 4], ["relu", "relu", "linear"], [False, False, False])
    dqn.load_state_dict(torch.load("data/puzzle8/nnet.pt"))
    dqn.eval()

    data = pickle.load(open("data/puzzle8/puzzle8_states.pkl", "rb"), encoding="latin1")
    states_start_all: List[State] = data['states']
    path_costs_gt_all: np.array = np.array(data['optimal_path_cost'])
    states_start: List[State] = []
    path_costs_gt: List[float] = []
    for path_cost in range(max(path_costs_gt_all) + 1):
        path_cost_idxs = np.where(path_costs_gt_all == path_cost)[0]
        path_cost_idxs_choose = np.random.choice(path_cost_idxs, size=min(5, path_cost_idxs.shape[0]))
        states_start.extend(states_start_all[idx] for idx in path_cost_idxs_choose)
        path_costs_gt.extend(path_costs_gt_all[idx] for idx in path_cost_idxs_choose)

    # run search
    is_optimal_l: List[bool] = []
    optimal_diffs: List[float] = []
    is_solved_l: List[bool] = []
    start_time_tot = time.time()
    for state_idx, state_start in enumerate(states_start):
        path_cost_gt = path_costs_gt[state_idx]

        start_time = time.time()
        actions: Optional[List[int]] = search(env, state_start, dqn)
        time_elapsed: float = time.time() - start_time

        path_cost, is_solved = check_soln(env, state_start, actions)
        time_elapsed_tot = time.time() - start_time_tot

        optimal_diffs.append(path_cost - path_cost_gt)
        is_solved_l.append(is_solved)
        is_optimal_l.append(path_cost == path_cost_gt)

        if not args.grade:
            print(f"State: {state_idx + 1}/{len(states_start)}, Path cost: {path_cost}, "
                  f"Optimal path cost: {path_cost_gt}, Solved: {is_solved}, "
                  f"Time state/total: %.5f secs / %.5f secs" % (time_elapsed, time_elapsed_tot))

    time_elapsed_tot = time.time() - start_time_tot
    print(f"Average difference with optimal path cost: {np.mean(optimal_diffs)}, "
          f"%%Solved: {100 * np.mean(is_solved_l)}%%, %%Optimal: {100 * np.mean(is_optimal_l)}%%, "
          f"Total time: %.5f secs" % time_elapsed_tot)


if __name__ == "__main__":
    main()
