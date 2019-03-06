import getpass
username = getpass.getuser()

# time horizon of a single rollout
HORIZON = 1000
# number of rollouts per training iteration

if username == "ferocious":
    N_ROLLOUTS = 4
    # number of parallel workers
    N_CPUS = 6
    N_GPUS = 1
else :
    N_ROLLOUTS = 40
    # number of parallel workers
    N_CPUS = 46
    N_GPUS = 0


# Tune
CHECKPOINT_FREQ = 20
MAX_FAILURES = 2
TRAINING_ITERATION = 2000
ALG_RUN = 'APEX' # 'PPO' , 'APEX'


# PPO
if ALG_RUN == 'PPO':
    GAMMA = 0.999
    HIDDEN_LAYERS = [32, 32]
    USE_GAE = True
    LAMBDA = 0.97
    KL_TARGET = 0.02
    NUM_SGD_ITER = 10
    CLIP_ACTIONS = False
    OBSERVATION_FILTER = "MeanStdFilter"

# APEX
if ALG_RUN == 'APEX':
    pass

""" APEX default for atari
    GAMMA = 0.999
    LR: .0001
    TARGET_NETWORK_UPDATE_FREQ: 50000
    NUM_WORKERS: 32
"""