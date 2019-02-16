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
TARGET_VELOCITY = 50
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

# PPO
ALG_RUN = 'PPO'
GAMMA = 0.999
HIDDEN_LAYERS = [32, 32]
USE_GAE = True
LAMBDA = 0.97
KL_TARGET = 0.02
NUM_SGD_ITER = 10
CLIP_ACTIONS = False

# Tune
CHECKPOINT_FREQ = 20
MAX_FAILURES = 999
TRAINING_ITERATION = 200