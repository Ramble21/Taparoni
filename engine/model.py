import torch.nn as nn
from torch.nn import functional as F
from parse_data import fen_to_wnn
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
        k = self.key(batch) # (B,T,C)
        q = self.query(batch) # (B,T,C)
        # Compute attention "affinities" following the formula in Attention Is All You Need
        head_size = q.size(-1)
        w = q @ k.transpose(-2, -1) * (head_size ** -0.5) # (B,T,C) @ (B,C, ) -> (B,T,T), then multiply for unit gaussian normalization
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

class TwoHeadTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TransformerBody()
        self.prediction_head = PredictionHead()
        self.evaluation_head = EvalHead()

    def forward(self, pieces, colors, ttm, index=None, fen=None, eval_weight=0.5, pred_targets=None, eval_targets=None, return_preds=False):
        x = self.backbone(pieces, colors, ttm)

        zero = torch.tensor(0.0, device=DEVICE)
        if eval_weight < 1:
            pred_probs, pred_loss = self.prediction_head(x, index=index, targets=pred_targets) if index is not None else self.prediction_head(x, fen=fen, targets=pred_targets)
        else:
            pred_probs, pred_loss = None, zero
        if eval_weight > 0:
            evaluation, eval_loss = self.evaluation_head(x, eval_targets)
        else:
            evaluation, eval_loss = None, zero

        if return_preds:
            return evaluation, pred_probs
        total_loss = (eval_weight * eval_loss) + ((1 - eval_weight) * pred_loss)
        return total_loss

class PredictionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pred_head = nn.Sequential(
            # expects input of size (B,C,8,8)
            nn.Conv2d(N_EMBD, CONV_CHANNELS, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(CONV_CHANNELS, CONV_CHANNELS, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2),
            nn.Conv2d(CONV_CHANNELS, NUM_PLANES, kernel_size=1)
        )

    def forward(self, x, index=None, fen=None, targets=None):
        from main import get_legal_move_mask, get_wnns_for_lmm
        # x -> (B,65,C), result of backbone.forward()
        board_tokens = x[:, 1:, :] # get rid of CLS token -> (B,64,C)
        board_tokens = board_tokens.view(-1, 8, 8, N_EMBD) # (B,8,8,C)
        board_tokens = board_tokens.permute(0, 3, 1, 2) # (B,C,8,8)
        logits = self.pred_head(board_tokens) # (B,NUM_PLANES,8,8)

        # Mask illegal moves
        wnns = get_wnns_for_lmm(index) if index is not None else [fen_to_wnn(fen)]
        mask = get_legal_move_mask(wnns)
        masked_logits = logits.masked_fill(~mask, float('-inf')).to(DEVICE)
        probs = F.softmax(masked_logits, dim=1)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(masked_logits, targets)
        return probs, loss

class EvalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.eval_head = nn.Linear(N_EMBD, 1)

    def forward(self, x, targets=None):
        # x -> (B,T,C), result of backbone.forward()
        cls_token = x[:, 0, :] # (B,C)
        preds = self.eval_head(cls_token) # (B,1)

        if targets is None:
            loss = None
        else:
            loss_criterion = nn.L1Loss()
            loss = loss_criterion(preds, targets)
        return preds, loss

class TransformerBody(nn.Module):
    # The body of the model itself
    def __init__(self):
        super().__init__()
        self.piece_emb = nn.Embedding(7, N_EMBD) # 7 pieces -> r,n,b,q,k,p + empty
        self.color_emb = nn.Embedding(3, N_EMBD) # 3 colors -> w,b + empty
        self.ttm_emb = nn.Embedding(2, N_EMBD) # 2 potential turns to move -> w,b
        self.square_emb = nn.Embedding(64, N_EMBD) # 64 squares on a chess board

        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_BLOCKS)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, N_EMBD))

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
        ) # (B,64,C)

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B,1,C)
        x = torch.cat([cls_tokens, x], dim=1) # (B,65,C) -> CLS token + 64 board square tokens

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        return x