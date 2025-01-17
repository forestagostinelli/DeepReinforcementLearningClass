from typing import Dict, List
from visualizer.farm_visualizer import InteractiveFarm
import numpy as np


def update_dp(viz: InteractiveFarm, state_values, policy):
    viz.set_state_values(state_values)
    viz.set_policy(policy)
    viz.window.update()


def update_model_free(env, viz: InteractiveFarm, state, action_values):
    viz.set_action_values(action_values)
    viz.board.delete(viz.agent_img)
    viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]

    policy = {}
    for state in action_values.keys():
        policy[state] = np.zeros(len(action_values[state])).tolist()
        policy[state][int(np.argmax(action_values[state]))] = 1.0

    viz.set_policy(policy, append=True)
    viz.window.update()
