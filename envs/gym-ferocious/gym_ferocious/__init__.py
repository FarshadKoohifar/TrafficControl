from gym.envs.registration import register

register(
    id='gym-ferocious-v0',
    entry_point='gym_ferocious.envs:GymFerocious',
)