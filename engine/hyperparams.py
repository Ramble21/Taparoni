import torch

# Basic hyperparameters
BATCH_SIZE = 32 # number of sequences being processed by the transformer in parallel
N_EMBD = 128 # number of embedding dimensions
NUM_HEADS = 4 # number of heads in each multi-head attention module
NUM_BLOCKS = 4 # number of blocks inside the transformer
DROPOUT = 0.1 # rate at which dropout is applied (i.e 0.2 = 20% of neurons)
LR = 3e-4 # learning rate parameter for optimization

# Hyperparameters used in move-pred head
CONV_CHANNELS = 32 # number of out channels in the convolution in pred head
KERNEL_SIZE = 3 # size of conv window (KERNEL_SIZE * KERNEL_SIZE) used in pred head to capture local patterns
NUM_PLANES = 67 # number of "types of moves" the model can make - AlphaZero uses 73, ignoring underpromotions and reclassifying promotions gives 67

# Other various magic numbers
MAX_CENTIPAWNS = 1000 # the maximum amount of centipawns used to determine "completely winning"
TRAINING_SIZE = 0.9 # amount of the dataset used in training vs dev, i.e 0.9 = 90% training, 10% dev
NUM_SAMPLES = 5 # number of samples taken as a test after model training

# Numbers used for model training
NUM_STEPS_PRETRAIN = 140 # number of steps used in model pretraining
NUM_STEPS_FINETUNE = 60 # number of steps used in model finetuning
MAX_FINETUNE_EVAL_WEIGHT = 0.4 # maximum percentage of the total loss assigned to the evaluation model during finetuning
LOSS_BENCH = 25 # number of steps inbetween loss benchmarking prints during optimization

# Pytorch device used for computations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'