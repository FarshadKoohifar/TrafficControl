import json
import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.controllers import SumoCarFollowingController, GridRouter
from flow.envs.ferocious_green_wave_env import ADDITIONAL_ENV_PARAMS

# time horizon of a single rollout
HORIZON = 200
# number of rollouts per training iteration
N_ROLLOUTS = 4
# number of parallel workers
N_CPUS = 4
# enter speed
v_enter = 30
# grid dimentions
inner_length = 800
long_length = 200
short_length = 800
n = 3
m = 3
num_cars_left = 30
num_cars_right = 30
num_cars_top = 30
num_cars_bot = 30
rl_veh = 0
tot_cars = (num_cars_left + num_cars_right) * m + (num_cars_bot + num_cars_top) * n

grid_array = {    'short_length': short_length,    'inner_length': inner_length,    'long_length': long_length,    'row_num': n,    'col_num': m,    'cars_left': num_cars_left,    'cars_right': num_cars_right,    'cars_top': num_cars_top,    'cars_bot': num_cars_bot,    'rl_veh': rl_veh}

additional_env_params = ADDITIONAL_ENV_PARAMS#{        'target_velocity': 50,        'switch_time': 3.0,        'num_observed': 2,        'discrete': False,        'tl_type': 'controlled'    }

additional_net_params = {    'speed_limit': 35,    'grid_array': grid_array,    'horizontal_lanes': 1,    'vertical_lanes': 1}

vehicles = Vehicles()
vehicles.add(    veh_id='idm',    acceleration_controller=(SumoCarFollowingController, {}),    sumo_car_following_params=SumoCarFollowingParams(        minGap=2.5,        max_speed=v_enter,    ),    routing_controller=(GridRouter, {}),    num_vehicles=tot_cars,    speed_mode='all_checks')

additional_init_params = {'enter_speed': v_enter}
initial_config = InitialConfig(additional_params=additional_init_params)
net_params = NetParams(no_internal_links=False, additional_params=additional_net_params)

flow_params = dict(    exp_tag='green_wave',    env_name='FerociousTrafficLightGridEnv',    scenario='SimpleGridScenario',    sumo=SumoParams(    sim_step=1,    render=True,    ),    env=EnvParams(    horizon=HORIZON,    additional_params=additional_env_params,),    net=net_params,    veh=vehicles,    initial=initial_config,)

def setup_exps():

    alg_run = 'PPO'

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]})
    config['use_gae'] = True
    config['lambda'] = 0.97
    config['kl_target'] = 0.02
    config['num_sgd_iter'] = 10
    config['horizon'] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == '__main__':
    alg_run, gym_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS + 1, redirect_output=False)
    trials = run_experiments({
        flow_params['exp_tag']: {   'run': alg_run,     'env': gym_name,    'config': {    **config    },           'checkpoint_freq': 20,          'max_failures': 999,    'stop': {    'training_iteration': 1000,},  }
    })

"""
def gen_edges(row_num, col_num):
    edges = []
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges


def get_flow_params(col_num, row_num, additional_net_params):
    initial_config = InitialConfig(
        spacing='uniform', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='idm',
            edge=outer_edges[i],
            probability=0.25,
            departLane='free',
            departSpeed=20)

    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    return initial_config, net_params
"""