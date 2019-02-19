from gym.envs.registration import register

from envs.ferocious_env import FerociousEnv, \
    PO_FerociousEnv, FerociousTestEnv

'''
TODO would be very nice if I had a default environment in which I could just do:
register( id='DirectFerociousEnv-v0', entry_point='envs.ferocious_env:Default_FerociousEnv',)
and be done
'''

__all__ = [
    'FerociousEnv','PO_FerociousEnv','FerociousTestEnv',
]
