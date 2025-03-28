from typing import List
import re
from environments.environment_abstract import Environment
import numpy as np


def get_environment(env_name: str):
    env_name = env_name.lower()
    farm_regex = re.search("aifarm(_(\S+))?", env_name)
    env: Environment

    if farm_regex is not None:
        from environments.farm_grid_world import FarmGridWorld, FarmState
        from visualizer.farm_visualizer import InteractiveFarm

        grid = np.loadtxt("maps/map1.txt")
        grid = np.transpose(grid)

        assert np.sum(grid == 1) == 1, "Only one agent allowed"
        assert np.sum(grid == 2) == 1, "Only one goal allowed"

        rand_right: float
        if farm_regex.group(1) is None:
            rand_right = 0.0
        else:
            rand_right = float(farm_regex.group(2))

        env: FarmGridWorld = FarmGridWorld(env_name, grid.shape, rand_right, grid)
        # viz = InteractiveFarm(env, grid)
        viz = None

        # get states
        states: List[FarmState] = []

        for pos_i in range(grid.shape[0]):
            for pos_j in range(grid.shape[1]):
                state: FarmState = FarmState((pos_i, pos_j), env.goal_idx, env.plant_idxs, env.rocks_idxs)
                states.append(state)
    elif env_name == "puzzle8":
        from environments.n_puzzle import NPuzzle
        env = NPuzzle(env_name, 3, False)

        states = []

        viz = None
    elif env_name == "puzzle8_rand":
        from environments.n_puzzle import NPuzzle
        env = NPuzzle(env_name, 3, True)

        states = []

        viz = None
    else:
        raise ValueError('No known environment %s' % env_name)

    return env, viz, states
