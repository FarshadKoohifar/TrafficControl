import os, sys, inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"config")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"envs")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from envs.ferocious_env import PO_FerociousEnv

import gym
import random
import numpy as np 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter



LR  = 1e-3
GOAL_STEPS = 500
SCORE_REQUIREMENT = 50
INITIAL_GAMES = 10000
FEROCIOUSENV = False


if FEROCIOUSENV:
    env = gym.make('DirectFerociousEnv-v0')
else:
    env = gym.make('CartPole-v0')
env.reset()
def some_random_games_first():
    for episode in range (5):
        env.reset()
        for _ in range(GOAL_STEPS):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

some_random_games_first()

