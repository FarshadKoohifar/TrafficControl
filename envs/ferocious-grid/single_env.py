import gym
import ferocious_grid
import numpy as np
env = gym.make('ferocious-grid-v0')
nof_iterations = 1
nof_steps = 2000
fixed_time = 20

do_random_aciton = True
do_constant_action = True
do_fixed_action = True

def show_action_space_samples():
    action_list = []
    env.reset()
    for t in range(50):
        action = env.action_space.sample()
        action_list.append( action )
        env.step(action)
    return action_list

def random_action(nof_iterations, nof_steps):
    sum_reward_list = []
    for _ in range(nof_iterations):
        env.reset()
        sum_reward = 0
        for t in range(nof_steps):
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            sum_reward += reward
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                break
        sum_reward_list.append( sum_reward )
    return sum_reward_list

def fixed_time_action(nof_iterations, nof_steps, fixed_time):
    sum_reward_list = []
    for _ in range(nof_iterations):
        env.reset()
        sum_reward = 0
        aciton_one = 2**4-1
        action_neg_one = 0
        for t in range(nof_steps):
            if (t % (2*fixed_time)) < fixed_time:
                action = aciton_one
            else:
                action = action_neg_one
            _, reward, done, _ = env.step(action)
            sum_reward += reward
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                break
        sum_reward_list.append( sum_reward )
    return sum_reward_list

def constant_action(nof_iterations,nof_steps):
    return fixed_time_action(nof_iterations, nof_steps, nof_steps)

action = show_action_space_samples()
print ("action: {}\t".format(action))
if do_random_aciton:
    print("===============================")
    sum_reward =  random_action(nof_iterations, nof_steps)
    print("random_action(nof_iterations={}, nof_steps={}) \t\t sum_reward: {}".format(nof_iterations, nof_steps,sum_reward))

if do_fixed_action:
    print("===============================")
    sum_reward = fixed_time_action(nof_iterations, nof_steps, fixed_time)
    print("fixed_time_action(nof_iterations={}, nof_steps={}, fixed_time={}) \t\t sum_reward: {}".format(nof_iterations, nof_steps, fixed_time,sum_reward))

if do_constant_action:
    print("===============================")
    sum_reward = constant_action(nof_iterations,nof_steps)
    print("constant_action(nof_iterations={}, nof_steps={}) \t\t sum_reward: {}".format(nof_iterations, nof_steps,sum_reward))