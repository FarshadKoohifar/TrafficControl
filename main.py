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
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
import config.configurations as CONFIG
from utils.registry import make_create_env

TOTAL_CARS = (CONFIG.NUM_CARS_LEFT + CONFIG.NUM_CARS_RIGHT) * CONFIG.N_COLUMNS + (CONFIG.NUM_CARS_BOT + CONFIG.NUM_CARS_TOP) * CONFIG.N_ROWS

def get_non_flow_params(enter_speed, additional_net_params):
    additional_init_params = {'enter_speed': enter_speed}
    initial_config = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

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

def setup_exps():
    agent_cls = get_agent_class(CONFIG.ALG_RUN)
    config = agent_cls._default_config.copy()
    config['num_workers'] = CONFIG.N_CPUS
    config['train_batch_size'] = CONFIG.HORIZON * CONFIG.N_ROLLOUTS
    config['gamma'] = CONFIG.GAMMA  # discount rate
    config['model'].update({'fcnet_hiddens': CONFIG.HIDDEN_LAYERS})
    config['use_gae'] = CONFIG.USE_GAE
    config['lambda'] = CONFIG.LAMBDA
    config['kl_target'] = CONFIG.KL_TARGET
    config['num_sgd_iter'] = CONFIG.NUM_SGD_ITER
    config['clip_actions'] = CONFIG.CLIP_ACTIONS  # FIXME(ev) temporary ray bug
    config['horizon'] = CONFIG.HORIZON

    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = CONFIG.ALG_RUN

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return gym_name, config

if __name__ == '__main__':
    gym_name, config = setup_exps()
    ray.init(num_cpus=CONFIG.N_CPUS + 1, redirect_output=False)
    trials = run_experiments({
        flow_params['exp_tag']: {
            'run': CONFIG.ALG_RUN,
            'env': gym_name,
            'config': {
                **config
            },
            'checkpoint_freq': CONFIG.CHECKPOINT_FREQ,
            'max_failures': CONFIG.MAX_FAILURES,
            'stop': {
                'training_iteration': CONFIG.TRAINING_ITERATION,
            },
        }
    })


"""
run_experiments
    gym_name
        flow_params
    config
        PPO config
        flow_json

"""