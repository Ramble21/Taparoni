import torch

# Hyperparameters
WNN_LEN = 67 # length of a WNN string
BATCH_SIZE = 32 # number of sequences being processed by the transformer in parallel
N_EMBD = 128 # number of embedding dimensions
NUM_HEADS = 4 # number of heads in each multi-head attention module
NUM_BLOCKS = 4 # number of blocks inside the transformer
DROPOUT = 0.1 # rate at which dropout is applied (i.e 0.2 = 20% of neurons)
LR = 3e-4 # learning rate parameter for optimization

# Other various magic numbers
MAX_CENTIPAWNS = 1000 # the maximum amount of centipawns used to determine "completely winning"
NUM_STEPS = 20000 # number of steps used in model training
NUM_SAMPLES = 5 # number of samples taken as a test after model training
LOSS_BENCH = 200 # number of steps inbetween loss benchmarking prints during optimization

# Misc
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # device used for computations