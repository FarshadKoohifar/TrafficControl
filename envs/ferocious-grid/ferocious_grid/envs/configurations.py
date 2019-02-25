class CONFIG:
    # time horizon of a single rollout
    HORIZON = 2000
    # number of rollouts per training iteration
    N_ROLLOUTS = 24
    # number of parallel workers
    N_CPUS = 6

    # grid_array
    INNER_LENGTH = 1000
    LONG_LENGTH = 1000
    SHORT_LENGTH = 1000
    N_ROWS = 3
    N_COLUMNS = 3
    NUM_CARS_LEFT = 10
    NUM_CARS_RIGHT = 10
    NUM_CARS_TOP = 10
    NUM_CARS_BOT = 10

    # additional_env_params
    SWITCH_TIME = 3.0
    MAX_PHASE_LENGTH = 120.0
    DISCRETE = False
    TL_TYPE = 'controlled'
    OBSERVATION_DISTANCE= 100.0

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