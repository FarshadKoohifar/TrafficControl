import gym
import ferocious_grid
env = gym.make('ferocious-grid-v0')

for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        print(env.observation_space)
        print('single_env: len(observation) {}'.format(len(observation)) )
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print(observation )