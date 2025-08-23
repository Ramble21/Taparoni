import torch.nn as nn
from torch.nn import functional as F

from hyperparams import *

# Implementation module of a single head of self-attention: the basis of the transformer
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, batch):
        B, T, C = batch.shape
        k = self.key(batch) # (B,T,C)
        q = self.query(batch) # (B,T,C)
        # Compute attention "affinities" following the formula in Attention Is All You Need
        w = q @ k.transpose(-2, -1) * (C ** -0.5) # (B,T,C) @ (B,C, ) -> (B,T,T), then multiply for unit gaussian normalization
        w = F.softmax(w, dim=-1) # Softmax to create full matrix
        w = self.dropout(w)
        v = self.value(batch) # (B,T,C)
        out = w @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
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
            nn.GELU(),
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

class TransformerPretrain(nn.Module):
    def __init__(self, backbone, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, pieces_batch, colors_batch, ttm_batch, index, targets=None):
        from main import get_legal_move_mask
        from main import decode_moves
        x = self.backbone(pieces_batch, colors_batch, ttm_batch) # (B,T,C)
        cls_token = x[:, 0, :] # (B,C)
        logits = self.lm_head(cls_token) # (B, vocab_size)
        # Mask illegal moves
        mask = get_legal_move_mask(index)
        masked_logits = logits.masked_fill(~mask, float('-inf')).to(DEVICE)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(masked_logits, targets)
        return masked_logits, loss

class TransformerEval(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = TransformerBody() if backbone is None else backbone
        self.eval_head = nn.Linear(N_EMBD, 1)

    def forward(self, pieces_batch, colors_batch, ttm_batch, targets=None):
        x = self.backbone(pieces_batch, colors_batch, ttm_batch) # (B,T,C)
        cls_token = x[:, 0, :] # (B,C)
        preds = self.eval_head(cls_token) # (B,1)

        if targets is None:
            loss = None
        else:
            loss_criterion = nn.L1Loss()
            loss = loss_criterion(preds, targets)
        return preds, loss

class TransformerBody(nn.Module):
    # The model itself
    def __init__(self):
        super().__init__()
        self.piece_emb = nn.Embedding(7, N_EMBD) # 7 pieces -> r,n,b,q,k,p + empty
        self.color_emb = nn.Embedding(3, N_EMBD) # 3 colors -> w,b + empty
        self.ttm_emb = nn.Embedding(2, N_EMBD) # 2 potential turns to move -> w,b
        self.square_emb = nn.Embedding(64, N_EMBD) # 64 squares on a chess board

        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_BLOCKS)])
        self.ln_f = nn.LayerNorm(N_EMBD)

    def forward(self, pieces_batch, colors_batch, ttm_batch):
        if pieces_batch.dim() == 1:
            pieces_batch = pieces_batch.unsqueeze(0)
            colors_batch = colors_batch.unsqueeze(0)
            ttm_batch = ttm_batch.unsqueeze(0)
        B, T = pieces_batch.shape
        squares = torch.arange(T, device=DEVICE).unsqueeze(0).expand(B, T) # fixed code to generate the same board every time
        x = (
                self.piece_emb(pieces_batch) +
                self.color_emb(colors_batch) +
                self.ttm_emb(ttm_batch) +
                self.square_emb(squares)
        ) # (B, 64, C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        return x