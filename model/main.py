from data.parse_data import get_dataset
from model import Transformer

# Hyperparameters
from hyperparams import *

# Open dataset
features_raw, labels_raw = get_dataset()
print(f"Dataset size: {len(features_raw):,} FENs")

# Character-level tokenization
chars = sorted(list({c for f in features_raw for c in f}))
vocab_size = len(chars)

s_to_i = {ch: i for i, ch in enumerate(chars)}
i_to_s = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [s_to_i[c] for c in s]
decode = lambda l: ''.join([i_to_s[i] for i in l])

features = torch.stack([torch.tensor(encode(f), dtype=torch.long) for f in features_raw])
labels = torch.tensor(labels_raw, dtype=torch.float).unsqueeze(1)

# 90% training, 10% dev
n = int(0.9*len(features))
features_tr =  features[:n]
features_dev = features[n:]
labels_tr = labels[:n]
labels_dev = labels[n:]

def get_batch(split):
  features_x = features_tr if split == 'train' else features_dev
  labels_x = labels_tr if split == 'train' else labels_dev
  index = torch.randint(len(features_x), (BATCH_SIZE, ))
  x = torch.stack([features_x[i] for i in index])
  y = torch.stack([labels_x[i] for i in index])
  x, y = x.to(DEVICE), y.to(DEVICE)
  return x, y

def train(m):
  # Pytorch optimizer object, used for training models in production
  # torch.optim.SGD represents the classic stochastic gradient descent optimization, but AdamW is more advanced and popular
  optimizer = torch.optim.AdamW(m.parameters(), lr=LR)
  # Train our model
  loss_log = []
  for i in range(1, NUM_STEPS + 1):
    X_b, Y_b = get_batch('train')
    # Forward pass
    probs, loss = m(X_b, Y_b)
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # Log
    loss_log.append(loss.item())
    if i % LOSS_BENCH == 0:
        recent_losses = loss_log[-LOSS_BENCH:]
        approx_loss = sum(recent_losses) / len(recent_losses)
        print(f"{i} / {NUM_STEPS}: approx. loss={approx_loss:.4f}")
  return loss_log

# ---------------- Transformer ---------------------------------

m = Transformer(vocab_size)
m.to(DEVICE)
log = train(m)
print("Training finished!")
