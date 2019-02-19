import gym
import universe
import random
import numpy as np 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter





env = gym.make('gym-ferocious-v0')
env.reset()
def some_random_games_first():
    for episode in range (5):
        env.reset()
        for _ in range(10):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

some_random_games_first()