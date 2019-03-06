import gym
import ferocious_grid
import numpy as np
env = gym.make('ferocious-grid-v0')
nof_iterations = 1
nof_steps = 2000
fixed_time = 20


def show_action_space_samples(nof_iterations):
    for i_episode in range(nof_iterations):
        env.reset()
        print("=============================== random_action <{}>".format(i_episode))
        for t in range(50):
            action = env.action_space.sample()
            env.step(action)
            print ("action: {}\t".format(action))

def random_action(nof_iterations, nof_steps):
    for i_episode in range(nof_iterations):
        observation = env.reset()
        sum_reward = 0
        for t in range(nof_steps):
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            sum_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        #print("=============================== random_action <{}>".format(i_episode))
        #print ("last action: {}\t".format(action))
        #print("last observation: {}\t".format(observation ))
        print("random_action(nof_iterations={}, nof_steps={}) \t\t sum_reward: {}".format(nof_iterations, nof_steps,sum_reward))

def fixed_time_action(nof_iterations, nof_steps, fixed_time):
    for i_episode in range(nof_iterations):
        observation = env.reset()
        sum_reward = 0
        aciton_one = 2**4-1
        action_neg_one = 0
        for t in range(nof_steps):
            if (t % (2*fixed_time)) < fixed_time:
                action = aciton_one
            else:
                action = action_neg_one
            observation, reward, done, _ = env.step(action)
            sum_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        #print("=============================== fixed_time_action <{}>".format(i_episode))
        #print ("last action: {}\t".format(action))
        #print("last observation: {}\t".format(observation ))
        print("fixed_time_action(nof_iterations={}, nof_steps={}, fixed_time={}) \t\t sum_reward: {}".format(nof_iterations, nof_steps, fixed_time,sum_reward))

def constant_action(nof_iterations,nof_steps):
    fixed_time_action(nof_iterations, nof_steps, nof_steps)

show_action_space_samples(nof_iterations)
random_action(nof_iterations, nof_steps)
fixed_time_action(nof_iterations, nof_steps, fixed_time)
constant_action(nof_iterations,nof_steps)
