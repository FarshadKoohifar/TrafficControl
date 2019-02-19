import gym
import universe
import random
import numpy as np 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

import ferocious_grid
'''
from gym.envs.registration import register

register(
    id='gym-ferocious-v1',
    entry_point='gym_ferocious.envs:GymFerocious',
)
'''

env = gym.make('ferocious-grid-v0')
env.reset()
def some_random_games_first():
    for episode in range (2):
        env.reset()
        for t in range(10):
            #env.render()
            action = t
            observation, reward, done, info = env.step(action)
            if done:
                break

some_random_games_first()