import numpy as np
import re
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
from ferocious_grid.envs.ferocious_env import FerociousEnv
from ferocious_grid.envs.configurations import CONFIG

from flow.scenarios.grid import SimpleGridScenario
from flow.core.params import EnvParams, SumoParams, VehicleParams ,InitialConfig, NetParams, SumoCarFollowingParams
from flow.controllers import SimCarFollowingController, GridRouter

class FerociousGrid(FerociousEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        env_params = EnvParams( additional_params={"switch_time": CONFIG.SWITCH_TIME,"tl_type": CONFIG.TL_TYPE, "discrete": CONFIG.DISCRETE, "observation_distance"= CONFIG.OBSERVATION_DISTANCE}, horizon=CONFIG.HORIZON)
        sim_params = SumoParams(sim_step=CONFIG.SIM_STEP, render=CONFIG.RENDER,)
        """
        for scenario, we need to do a lot of work! It needs vehicles. Vehicles need router etc:
            scenario
                vehicles
                    SimCarFollowingController
                    SumoCarFollowingParams
                    GridRouter
                    num_vehicles
                net_params
                    grid_array
                    additional_net_params
                initial_config
                traffic_lights
        """
        TOTAL_CARS = (CONFIG.NUM_CARS_LEFT + CONFIG.NUM_CARS_RIGHT) * CONFIG.N_COLUMNS + (CONFIG.NUM_CARS_BOT + CONFIG.NUM_CARS_TOP) * CONFIG.N_ROWS
        vehicles = VehicleParams()
        vehicles.add(
            veh_id='idm',
            acceleration_controller=(SimCarFollowingController, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=CONFIG.MINGAP,
                max_speed=CONFIG.V_ENTER,
                speed_mode="all_checks",
            ),
            routing_controller=(GridRouter, {}),
            num_vehicles=TOTAL_CARS)

        grid_array = {
            "short_length": CONFIG.SHORT_LENGTH,
            "inner_length": CONFIG.INNER_LENGTH,
            "long_length": CONFIG.LONG_LENGTH,
            "row_num": CONFIG.N_ROWS,
            "col_num": CONFIG.N_COLUMNS,
            "cars_left": CONFIG.NUM_CARS_LEFT,
            "cars_right": CONFIG.NUM_CARS_RIGHT,
            "cars_top": CONFIG.NUM_CARS_TOP,
            "cars_bot": CONFIG.NUM_CARS_BOT
        }
        additional_net_params = {
            'speed_limit': CONFIG.SPEED_LIMIT,
            'grid_array': grid_array,
            'horizontal_lanes': CONFIG.HORIZONTAL_LANES,
            'vertical_lanes': CONFIG.VERTICAL_LANES
        }
        net_params= NetParams(no_internal_links=False, additional_params=additional_net_params)

        initial_config = InitialConfig(spacing='custom', additional_params={'enter_speed': CONFIG.V_ENTER})

        scenario = SimpleGridScenario(
            name="thisGetsDiscarded",
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config
        )
        super().__init__(env_params, sim_params, scenario )
        # since there is no reference to kernel_api.vehicle.getLength, I guestimate it. Should be used just for normalization.
        self.max_vehicle_density = 1.0/(2*CONFIG.MINGAP)
        self.edge_names = self.k.vehicle.get_edge_names()
        print (self.edge_names)
        print (len(edge_names))
        self.observed_queue_max = 0
        for i in range(0, self.observation_distance, 1/self.max_vehicle_density):
            self.observed_queue_max += self.observation_distance - i
        self.observed_queue = np.zeros( len(edge_names) )
        self.traffic_lights = np.zeros( 4 * self.num_traffic_lights )
        for i in range (num_traffic_lights):
            self.traffic_lights[4*i:4*i+4] = [1, 0, 0, 0]
        self.lastchange = np.zeros( self.num_traffic_lights )

    @property
    def action_space(self):
        """This is exacly the same as parrent, but in the future, we are going to be able to add phases and make it more general"""
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
        """ Use normalized weighted average of cars in the queue as observation.
            Normalization is with regards to a full queue of vehicles.
            Traffic lights are observed as a vector signifying which lights are on.
            The lapsed time from the last change is normalized to the max phase lenght."""
        observed_queue = Box(
            low=0,
            high=1,
            shape=(self.num_edges,),   # we observe all edges
            dtype=np.float32)
        traffic_lights = Box(
            low=0.,
            high=1,
            shape=(4 * self.num_traffic_lights,),   # each traffic light can be observed as in up-down, left-rigth, or in yellow. [1 0 0 0] means grgr, [0 1 0 0] means yryr, [0 0 1 0] means rgrg, and [0 0 0 1] means ryry
            dtype=np.float32) # TODO after validation that it works with float, type should be turned into boolean.
        last_change = Box(
            low=0.,
            high=1,
            shape=(self.num_traffic_lights,),   # normalized time laps since the last change.
            dtype=np.float32)
        return Tuple((observed_queue, traffic_lights, last_change))

    def get_state(self):
        for index , edge in enumerate(self.edge_names)
            vehicles = self.k.vehicle.get_ids_by_edge (edge)
            self.observed_queue[index]=0
            for veh_id in vehicles:
                dist_to_intersec = self.edge_end_dist(veh_id, edge_id)
                if dist_to_intersec == -1 or dist_to_intersec > self.observation_distance: # this vehicle is inside the intersection or outside observation distance
                    self.k.vehicle.set_color(veh_id, (255, 255, 255)):
                    continue
                #if green_lighted and speed is high:continue
                if dist_to_intersec <= self.observation_distance:
                    self.observed_queue[index] += (self.observation_distance - dist_to_intersec)
                    self.k.vehicle.set_color(veh_id, (0, 255, 0)):
            self.observed_queue[index] /= self.observed_queue_max

        return np.array([self.observed_queue, self.traffic_lights, self.lastchange])

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

        """
            This needs to update traffic_lights and lastchange:
            for each intersection:
            (1) if enough time has not passed since last action:
                    just update lastchage
                    continue
            (2) else:
            (3)     if intersection is in yellow:
                        intersection go red or green
                        lastchange = 0
                        continue
            (4)     if rl_action == current state
                        just update lastchange
                        continue
            (5)     else:
                        intersection go yellow
                        last change = 0
                        continue
        """
        for intersection, action in enumerate(rl_mask):
            assert (sum(self.traffic_lights[4*intersection:4*intersection+4]==1))
            # (1)
            if self.last_change [ intersection ] + self.sim_step < self.min_switch_time:  # cant act
                self.last_change [ intersection ] += self.sim_step
                continue
            # (2)
            #(3)
            if self.traffic_lights[4*intersection+1] :
                self.traffic_lights[4*intersection+1] = 0
                self.traffic_lights[4*intersection+2] = 1
                self.last_change [ intersection ] = 0
                self.k.traffic_light.set_state(node_id='center{}'.format(intersection),state="rGrG")
                continue
            if self.traffic_lights[4*intersection+3] :
                self.traffic_lights[4*intersection+3] = 0
                self.traffic_lights[4*intersection+0] = 1
                self.last_change [ intersection ] = 0
                self.k.traffic_light.set_state(node_id='center{}'.format(intersection),state="GrGr")
                continue
            #(4)
            if self.traffic_lights[4*intersection+0] and rl_mask[intersection]:
                self.last_change [ intersection ] += self.sim_step
                continue
            if self.traffic_lights[4*intersection+2] and  not rl_mask[intersection]:
                self.last_change [ intersection ] += self.sim_step
                continue
            #(5)
            if self.traffic_lights[4*intersection+0] and  not rl_mask[intersection]:
                self.traffic_lights[4*intersection+1] = 1
                self.traffic_lights[4*intersection+0] = 0
                self.last_change [ intersection ] = 0
                self.k.traffic_light.set_state(node_id='center{}'.format(intersection),state="yryr")
                continue
            if self.traffic_lights[4*intersection+2] and rl_mask[intersection]:
                self.traffic_lights[4*intersection+3] = 1
                self.traffic_lights[4*intersection+2] = 0
                self.last_change [ intersection ] = 0
                self.k.traffic_light.set_state(node_id='center{}'.format(intersection),state="ryry")
                continue

    def compute_reward(self, rl_actions, **kwargs):
        vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))
        maxwell = len(vel)*CONFIG.SPEED_LIMIT
        return np.sum(vel)/maxwell