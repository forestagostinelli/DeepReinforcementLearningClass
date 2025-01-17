from typing import List, Dict
from environments.environment_abstract import State
from environments import env_utils
from visualizer import viz_utils

from argparse import ArgumentParser

from code_hw_answers.code_hw1 import mrp, dynamic_programming, model_free
from numpy.typing import NDArray

import pickle
import numpy as np


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help="mrp, dp, mf")
    parser.add_argument('--env', type=str, default="aifarm", help="")
    parser.add_argument('--gamma', type=float, default=1.0, help="discount factor")
    parser.add_argument('--grade', default=False, action='store_true', help="")

    args = parser.parse_args()

    viz = None
    if args.task == "mrp":
        state_trans_probs = np.array([
            [0.7, 0.2, 0.1],
            [1 / 3, 1 / 3, 1 / 3],
            [0.1, 0.2, 0.7]
        ])
        state_rewards = np.array([[-10, -1, 10]]).transpose()
        state_names: List[str] = ["O", "A", "U"]
        gamma_answers: Dict[float, NDArray] = pickle.load(open("data_hw/mrp_example.pkl", "rb"))
        state_values: NDArray = mrp(state_trans_probs, state_rewards, args.gamma)
        assert state_values.shape[0] == state_rewards.shape[0]
        assert state_values.shape[1] == state_rewards.shape[1]
        printstr_l: List[str] = []
        for state_name, state_value in zip(state_names, state_values[:, 0]):
            printstr_l.append(f"{state_name}:{state_value}")
        print(f"State values: {', '.join(printstr_l)}")
        if args.gamma in gamma_answers.keys():
            mean_abs_error: float = np.mean(np.abs(gamma_answers[args.gamma] - state_values))
            print(f"Mean absolute error: {mean_abs_error}")
    elif args.task == "dp":
        env, viz, states = env_utils.get_environment(args.env)
        policy: Dict[State, List[float]] = {}
        state_values: Dict[State, float] = {}
        for state in states:
            num_actions: int = len(env.get_actions(state))
            policy[state] = (np.ones(num_actions)/num_actions).tolist()
            state_values[state] = 0.0
        state_vals, policy = dynamic_programming(env, states, state_values, args.gamma)
        viz_utils.update_dp(viz, state_vals, policy)

        if args.grade:
            ans_file_name: str = f"grading/code_hw1/policy_iteration_{args.env}.pkl"
            state_vals_ans, policy_ans = pickle.load(open(ans_file_name, "rb"))

            state_val_diffs: List[float] = []
            for state in states:
                state_val_diff: float = state_vals_ans[state] - state_vals[state]
                state_val_diffs.append(state_val_diff)

            print("State value diffs: Mean/Min/Max (Std): %.2f/%.2f/%.2f "
                  "(%.2f)" % (float(np.mean(state_val_diffs)), np.min(state_val_diffs), np.max(state_val_diffs),
                              float(np.std(state_val_diffs))))
    elif args.task == "mf":
        env, viz, states = env_utils.get_environment(args.env)
        action_values: Dict[State, List[float]] = {}
        for state in states:
            num_actions: int = len(env.get_actions(state))
            action_values[state] = np.zeros(num_actions).tolist()

        action_values: Dict[State, List[float]] = model_free(env, action_values, args.gamma)
        viz_utils.update_model_free(env, viz, states[0], action_values)
    else:
        raise ValueError("Unknown algorithm %s" % args.algorithm)

    print("DONE")

    if viz is not None:
        viz.mainloop()


if __name__ == "__main__":
    main()
