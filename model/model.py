import torch.nn as nn
from torch.nn import functional as F
from hyperparams import *

# THIS MODEL DOES NOT WORK YET!

# Implementation module of a single head of self-attention: the basis of the transformer
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # Dropouts are used right before reconnecting with residual breakaways, preventing some nodes from communicating
        # Dropouts were introduced as a method to lower the probability of overfitting
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, batch):
        B, T, C = batch.shape
        k = self.key(batch) # (B, T, C)
        q = self.query(batch) # (B, T, C)
        # Compute attention "affinities" following the formula in Attention Is All You Need
        w = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, C) @ (B, C, T) -> (B, T, T), then multiply for unit gaussian normalization
        w = F.softmax(w, dim=1) # Softmax to create full matrix
        w = self.dropout(w)
        v = self.value(batch) # (B, T, C)
        out = w @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    # Module to implement multiple heads of self attention in parallel followed by a linear projection
    def __init__(self, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, batch):
        out = torch.cat([h(batch) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    # Linear + nonlinearity block
    def __init__(self, n_embd):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.Tanh(),
            nn.Linear(4 * n_embd, n_embd), # Linear projection
            nn.Dropout(DROPOUT)
        )
    def forward(self, batch):
        return self.seq(batch)

class TransformerBlock(nn.Module):
    # The body of the transformer: communication followed by computation
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // NUM_HEADS
        self.mha = MultiHeadAttention(head_size)
        self.ffwd = FeedForward(N_EMBD)
        # LayerNorm layers- very similar to BatchNorm but normalizes along rows instead of columns
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
    def forward(self, batch):
        batch = batch + self.mha(self.ln1(batch))
        batch = batch + self.ffwd(self.ln2(batch))
        return batch

class Transformer(nn.Module):
    # The model itself
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_emb_table = nn.Embedding(WNN_LEN, N_EMBD)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_BLOCKS)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, 1)

    def forward(self, batch, targets=None):
        B, T = batch.shape
        tok_emb = self.token_emb_table(batch) # (B,T,C)
        pos_emb = self.position_emb_table(torch.arange(T, device=DEVICE)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        x = self.lm_head(x) # (B,T,1)
        preds = x.mean(dim=1) # (B*T, 1)

        # If no targets are given, avoid an error by nulling loss
        if targets is None:
            loss = None
        else:
            loss_criterion = nn.L1Loss()
            loss = loss_criterion(preds, targets)
        return preds, loss
