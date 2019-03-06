class CONFIG_BASE:
    # time horizon of a single rollout
    HORIZON = 1000

    # grid_array
    INNER_LENGTH = 1000.0
    LONG_LENGTH = 1000.0
    SHORT_LENGTH = 1000.0
    N_ROWS = 3
    N_COLUMNS = 3
    NUM_CARS_LEFT = 30
    NUM_CARS_RIGHT = 30
    NUM_CARS_TOP = 30
    NUM_CARS_BOT = 30

    # additional_net_params
    SPEED_LIMIT = 35
    HORIZONTAL_LANES = 1
    VERTICAL_LANES = 1

    # vehicles
    V_ENTER = 30
    MINGAP = 2.5

    # SumoParams
    SIM_STEP = 1
    RENDER = False

    # additional_env_params
    SWITCH_TIME = 3.0
    DISCRETE = True
    TL_TYPE = 'actuated' #'controlled','actuated'

#class CONFIG_Q_WEIGHT_OBSERVATION (CONFIG_BASE):
    # additional_env_params
    MAX_PHASE_LENGTH = 120.0
    OBSERVATION_DISTANCE= 100.1
    OBSERVATION_MODE = "Q_WEIGHT"

#class CONFIG_SEGMENT_OBSERVATION (CONFIG_BASE):
    # additional_env_params
    MAX_PHASE_LENGTH = 120.0
    SEGMENT_LENGTH= 100.1
    OBSERVATION_MODE = "SEGMENT"


