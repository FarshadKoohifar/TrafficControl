# time horizon of a single rollout
HORIZON = 2000
# number of rollouts per training iteration
N_ROLLOUTS = 24
# number of parallel workers
N_CPUS = 6

# PPO
ALG_RUN = 'PPO'
GAMMA = 0.999
HIDDEN_LAYERS = [32, 32]
USE_GAE = True
LAMBDA = 0.97
KL_TARGET = 0.02
NUM_SGD_ITER = 10
CLIP_ACTIONS = False
OBSERVATION_FILTER = "MeanStdFilter"

# Tune
CHECKPOINT_FREQ = 20
MAX_FAILURES = 2
TRAINING_ITERATION = 200
