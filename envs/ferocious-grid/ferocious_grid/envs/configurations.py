class CONFIG:
    # time horizon of a single rollout
    HORIZON = 2000
    # number of rollouts per training iteration
    N_ROLLOUTS = 24
    # number of parallel workers
    N_CPUS = 6

    # grid_array
    INNER_LENGTH = 300
    LONG_LENGTH = 100
    SHORT_LENGTH = 300
    N_ROWS = 3
    N_COLUMNS = 3
    NUM_CARS_LEFT = 1
    NUM_CARS_RIGHT = 1
    NUM_CARS_TOP = 1
    NUM_CARS_BOT = 1

    # additional_env_params
    SWITCH_TIME = 3.0
    NUM_OBSERVED = 2
    DISCRETE = False
    TL_TYPE = 'controlled'

    # additional_net_params
    SPEED_LIMIT = 35
    HORIZONTAL_LANES = 1
    VERTICAL_LANES = 1

    # vehicles
    V_ENTER = 30
    MINGAP = 2.5

    # SumoParams
    SIM_STEP = 1
    RENDER = True