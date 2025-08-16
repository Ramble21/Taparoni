import torch

# Hyperparameters
WNN_LEN = 69 # length of a WNN string (nice)
BATCH_SIZE = 16 # number of sequences being processed by the transformer in parallel
N_EMBD = 128 # number of embedding dimensions
NUM_HEADS = 4 # number of heads in each multi-head attention module
NUM_BLOCKS = 6 # number of blocks inside the transformer
DROPOUT = 0.1 # rate at which dropout is applied (i.e 0.2 = 20% of neurons)
LR = 3e-4 # learning rate parameter for optimization

# Other various magic numbers
NUM_STEPS = 1000 # number of steps used in model training
LOSS_BENCH = 50 # number of steps inbetween loss benchmarking prints during optimization

# Misc
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # device used for computations