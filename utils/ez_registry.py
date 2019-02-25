"""Modified Utility method for registering environments with OpenAI gym."""
import os, sys, inspect
cmd_folder = os.path.realpath(os.path.abspath(    os.path.join(os.path.split(os.path.split(inspect.getfile( inspect.currentframe() ))[0])[0],"envs")       ))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_folder = os.path.realpath(os.path.abspath(    os.path.split(os.path.split(inspect.getfile( inspect.currentframe() ))[0])[0]        ))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import gym
from gym.envs.registration import register

from copy import deepcopy

# import flow.envs
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import envs
import ferocious_grid


def make_create_env():
    """Create a parametrized flow environment compatible with OpenAI gym.

    Returns
    -------
    function
        method that calls OpenAI gym's register method and make method
    str
        name of the created gym environment
    """
    env_name = 'ferocious-grid-v1'

    def create_env(*_):
        try:
            register(
                id=env_name,
                entry_point='ferocious_grid.envs:FerociousGrid',)
        except Exception:
            pass
        return gym.envs.make(env_name)

    return create_env, env_name
