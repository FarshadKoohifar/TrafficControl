from gym.envs.registration import register

register(
    id='ferocious-grid-v0',
    entry_point='ferocious_grid.envs:FerociousGrid',
)