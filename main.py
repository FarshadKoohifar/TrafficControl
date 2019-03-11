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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
import config.rl_configurations as CONFIG
from utils.ez_registry import make_create_env
import gym
import ferocious_grid

if __name__ == '__main__':
    agent_cls = get_agent_class(CONFIG.ALG_RUN)
    config = agent_cls._default_config.copy()
    config['num_workers'] = CONFIG.N_CPUS
    config['num_gpus'] = CONFIG.N_GPUS
    config['train_batch_size'] = CONFIG.HORIZON * CONFIG.N_ROLLOUTS
    config['horizon'] = CONFIG.HORIZON
    config['env_config']['run'] = CONFIG.ALG_RUN
    if CONFIG.ALG_RUN == 'PPO':
        config['gamma'] = CONFIG.GAMMA  # discount rate
        config['model'].update({'fcnet_hiddens': CONFIG.HIDDEN_LAYERS})
        config['use_gae'] = CONFIG.USE_GAE
        config['lambda'] = CONFIG.LAMBDA
        config['kl_target'] = CONFIG.KL_TARGET
        config['num_sgd_iter'] = CONFIG.NUM_SGD_ITER
        config['clip_actions'] = CONFIG.CLIP_ACTIONS  # FIXME(ev) temporary ray bug
        config['observation_filter']= CONFIG.OBSERVATION_FILTER
    if CONFIG.ALG_RUN == 'APEX':
        pass

    create_env, gym_name = make_create_env()

    # Register as rllib env
    register_env(gym_name, create_env)

    ray.init(num_cpus=CONFIG.N_CPUS + 1, num_gpus = CONFIG.N_GPUS, redirect_output=False)
    trials = run_experiments({
        CONFIG.ALG_RUN: {
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

