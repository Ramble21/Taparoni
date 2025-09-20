import torch

# Basic hyperparameters
BATCH_SIZE = 32 # number of sequences being processed by the transformer in parallel
N_EMBD = 128 # number of embedding dimensions
N_EMBD_EVAL = 256 # number of embedding dimensions in the eval head MLP
NUM_HEADS = 4 # number of heads in each multi-head attention module
NUM_BLOCKS = 4 # number of blocks inside the transformer
DROPOUT = 0.1 # rate at which dropout is applied (i.e 0.2 = 20% of neurons)
LR = 1e-4 # learning rate parameter for optimization

# Hyperparameters used in move-pred head
CONV_CHANNELS = 32 # number of out channels in the convolution in pred head
KERNEL_SIZE = 3 # size of conv window (KERNEL_SIZE * KERNEL_SIZE) used in pred head to capture local patterns
NUM_PLANES = 67 # number of "types of moves" the model can make - AlphaZero uses 73, ignoring underpromotions and reclassifying promotions gives 67

# Other various magic numbers
MAX_CENTIPAWNS = 1000 # the maximum amount of centipawns used to determine "completely winning"
MATE_VALUE = 100 # value (in pawns) used to represent mate
TRAINING_SIZE = 0.9 # amount of the dataset used in training vs dev, i.e 0.9 = 90% training, 10% dev
NUM_SAMPLES = 5 # number of samples taken as a test after model training

# Hyperparameters used at search time
PRED_WEIGHT = 0.075 # weight that is assigned to the probabilities spit out by the model
MODEL_DEPTH = 3 # minimax depth of the model
MAX_LINES = 3 # maximum number of candidate moves selected, ignoring captures, checks, hanging pieces
QUIESCENCE_DEPTH = 2 # additional depth added for quiescence after captures
MIN_MATERIAL = 10 # maximum material in which a position is considered 100% an endgame
MAX_MATERIAL = 70 # minimum material in which endgame heuristics are completely ignored

# Bonuses given for endgame heuristics
SIMPLIFICATION_BONUS = 0.4
PASSED_PAWN_BONUS = 0.1
ROOK_BEHIND_PAWN_BONUS = 0.3

# Chess piece relative values in centipawns (https://arxiv.org/pdf/2009.04374 page 16)
PAWN_VALUE = 100
KNIGHT_VALUE = 305
BISHOP_VALUE = 333
ROOK_VALUE = 563
QUEEN_VALUE = 950

# Numbers used for model training
NUM_STEPS_PRETRAIN = 0 # number of steps used in model pretraining
NUM_STEPS_FINETUNE = 250000 # number of steps used in model finetuning
MAX_FINETUNE_EVAL_WEIGHT = 0.4 # maximum percentage of the total loss assigned to the evaluation model during finetuning
LOSS_BENCH = 250 # number of steps inbetween loss benchmarking prints during optimization
TRN_SC = 10000 # number of iterations of training in between each "checkpoint" that saves model weights

# Pytorch device used for computations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'