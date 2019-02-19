import numpy as np
import re
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
from ferocious_grid.envs.ferocious_env import FerociousEnv
from ferocious_grid.envs.configurations import CONFIG

class FerociousGrid(FerociousEnv):
    """
    def __init__(self):
        pass
    def step(self, action):
        return CONFIG.GAMMA, 0,0,0
        pass
    def reset(self):
        pass
    def render(self, mode='human', close=False):
        pass
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__(env_params, sim_params, scenario)

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(2 ** self.num_traffic_lights)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(self.num_traffic_lights,),
                dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        speed = Box(
            low=0,
            high=1,
            shape=(self.scenario.vehicles.num_vehicles,),
            dtype=np.float32)
        dist_to_intersec = Box(
            low=0.,
            high=np.inf,
            shape=(self.scenario.vehicles.num_vehicles,),
            dtype=np.float32)
        edge_num = Box(
            low=0.,
            high=1,
            shape=(self.scenario.vehicles.num_vehicles,),
            dtype=np.float32)
        traffic_lights = Box(
            low=0.,
            high=1,
            shape=(3 * self.rows * self.cols,),
            dtype=np.float32)
        return Tuple((speed, dist_to_intersec, edge_num, traffic_lights))

    def get_state(self):
        """See class definition."""
        # compute the normalizers
        max_dist = max(self.k.scenario.network.short_length,
                       self.k.scenario.network.long_length,
                       self.k.scenario.network.inner_length)

        # get the state arrays
        speeds = [
            self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
            for veh_id in self.k.vehicle.get_ids()
        ]
        dist_to_intersec = [
            self.get_distance_to_intersection(veh_id) / max_dist
            for veh_id in self.k.vehicle.get_ids()
        ]
        edges = [
            self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
            (self.k.scenario.network.num_edges - 1)
            for veh_id in self.k.vehicle.get_ids()
        ]

        state = [
            speeds, dist_to_intersec, edges,
            self.last_change.flatten().tolist()
        ]
        return np.array(state)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        if self.discrete:
            # convert single value to list of 0's and 1's
            rl_mask = [int(x) for x in list('{0:0b}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        else:
            # convert values less than 0.5 to zero and above to 1. 0's indicate
            # that should not switch the direction
            rl_mask = rl_actions > 0.0

        for i, action in enumerate(rl_mask):
            # check if our timer has exceeded the yellow phase, meaning it
            # should switch to red
            if self.last_change[i, 2] == 0:  # currently yellow
                self.last_change[i, 0] += self.sim_step
                if self.last_change[i, 0] >= self.min_switch_time:
                    if self.last_change[i, 1] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state="GrGr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='rGrG')
                    self.last_change[i, 2] = 1
            else:
                if action:
                    if self.last_change[i, 1] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='yryr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='ryry')
                    self.last_change[i, 0] = 0.0
                    self.last_change[i, 1] = not self.last_change[i, 1]
                    self.last_change[i, 2] = 0

    def compute_reward(self, rl_actions, **kwargs):
        raise NotImplementedError