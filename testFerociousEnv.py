"""Entry point to trainig RL for Ferocious Grid environment"""
import os, sys, inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"config")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"utils")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import json

import gym
from gym.envs.registration import register

from copy import deepcopy

# import flow.envs
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import envs

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
import config.configurations as CONFIG
from utils.registry import make_create_env








def make_create_env1(params, version=0, render=None):

    exp_tag = params["exp_tag"]

    env_name = params["env_name"] + '-v{}'.format(version)

    module = __import__("flow.scenarios", fromlist=[params["scenario"]])
    scenario_class = getattr(module, params["scenario"])

    env_params = params['env']
    net_params = params['net']
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLightParams())

    def create_env(*_):
        sim_params = deepcopy(params['sim'])
        vehicles = deepcopy(params['veh'])

        scenario = scenario_class(
            name=exp_tag,
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=traffic_lights,
        )

        if render is not None:
            sim_params.render = render

        env_loc = 'envs'

        try:
            register(
                id=env_name,
                entry_point=env_loc + ':{}'.format(params["env_name"]),
                kwargs={
                    "env_params": env_params,
                    "sim_params": sim_params,
                    "scenario": scenario,
                    "simulator": params['simulator']
                })
        except Exception:
            pass
        return gym.envs.make(env_name)

    return create_env, env_name












TOTAL_CARS = (CONFIG.NUM_CARS_LEFT + CONFIG.NUM_CARS_RIGHT) * CONFIG.N_COLUMNS + (CONFIG.NUM_CARS_BOT + CONFIG.NUM_CARS_TOP) * CONFIG.N_ROWS

def get_non_flow_params(enter_speed, additional_net_params):
    additional_init_params = {'enter_speed': enter_speed}
    initial_config = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net_params = NetParams( no_internal_links=False, additional_params=additional_net_params)
    return initial_config, net_params

grid_array = {
    "short_length": CONFIG.SHORT_LENGTH,
    "inner_length": CONFIG.INNER_LENGTH,
    "long_length": CONFIG.LONG_LENGTH,
    "row_num": CONFIG.N_ROWS,
    "col_num": CONFIG.N_COLUMNS,
    "cars_left": CONFIG.NUM_CARS_LEFT,
    "cars_right": CONFIG.NUM_CARS_RIGHT,
    "cars_top": CONFIG.NUM_CARS_TOP,
    "cars_bot": CONFIG.NUM_CARS_BOT
}

# Ferocious can add different reward and observation parameters here to be passed to Ferocious's environment!
additional_env_params = {
        'target_velocity': CONFIG.TARGET_VELOCITY,
        'switch_time': CONFIG.SWITCH_TIME,
        'num_observed': CONFIG.NUM_OBSERVED,
        'discrete': CONFIG.DISCRETE,
        'tl_type': CONFIG.TL_TYPE
    }

additional_net_params = {
    'speed_limit': CONFIG.SPEED_LIMIT,
    'grid_array': grid_array,
    'horizontal_lanes': CONFIG.HORIZONTAL_LANES,
    'vertical_lanes': CONFIG.VERTICAL_LANES
}

vehicles = VehicleParams()
vehicles.add(
    veh_id='idm',
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        minGap=CONFIG.MINGAP,
        max_speed=CONFIG.V_ENTER,
        speed_mode="all_checks",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=TOTAL_CARS)

initial_config, net_params = get_non_flow_params(CONFIG.V_ENTER, additional_net_params)

flow_params = dict(
    exp_tag='ferocious',
    env_name='PO_FerociousEnv',
    scenario='SimpleGridScenario',
    simulator='traci',
    sim=SumoParams(
        sim_step=1,
        render=False,
    ),
    env=EnvParams(
        horizon=CONFIG.HORIZON,
        additional_params=additional_env_params,
    ),
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

if __name__ == '__main__':
    print('redid')
    create_env, gym_name = make_create_env(params=flow_params, version=0)
    register_env(gym_name, create_env)



"""
def __init__(self, env_params, sim_params, scenario, simulator='traci'):
"""