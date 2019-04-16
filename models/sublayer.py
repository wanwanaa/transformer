import torch
import torch.nn as nn
import math
import numpy as np


# Scaled Dot Product Attention
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_k = config.model_size

        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # q, k, v (batch, len, model_size)
        attn = torch.bmm(q, k.transpose(1, 2)) # (batch, len, len)
        attn = attn / math.sqrt(self.d_k)
        if mask is not None:
            # Fills elements of self tensor with value where mask is one
            attn = attn.masked_fill(mask, -1e18)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        out = torch.bmm(attn, v) # (batch, len, size)

        return out, attn


# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_size = config.model_size
        self.h = config.n_head
        self.d_k = config.model_size // self.h

        self.attention = Attention(config)

        self.linear_q = nn.Linear(config.model_size, config.model_size)
        self.linear_k = nn.Linear(config.model_size, config.model_size)
        self.linear_v = nn.Linear(config.model_size, config.model_size)
        # nn.init.normal_(self.linear_q.weight, mean=0, std=np.sqrt(2.0 / (self.model_size + self.d_k)))
        # nn.init.normal_(self.linear_k.weight, mean=0, std=np.sqrt(2.0 / (self.model_size + self.d_k)))
        # nn.init.normal_(self.linear_v.weight, mean=0, std=np.sqrt(2.0 / (self.model_size + self.d_k)))

        self.linear_out = nn.Linear(config.model_size, config.model_size)
        # nn.init.xavier_normal_(self.linear_out.weight)

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.model_size, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # q, k , v(batch, len, model_size)
        residual = q
        len_q = q.size(1)
        len_k = k.size(1)
        len_v = v.size(1)

        # (batch, len, h, d_k)
        q = self.linear_q(q).contiguous().view(-1, len_q, self.h, self.d_k)
        k = self.linear_q(k).contiguous().view(-1, len_k, self.h, self.d_k)
        v = self.linear_q(v).contiguous().view(-1, len_v, self.h, self.d_k)

        # (batch*h, len, d_k)
        q = q.transpose(1, 2).contiguous().view(-1, len_q, self.d_k)
        k = k.transpose(1, 2).contiguous().view(-1, len_k, self.d_k)
        v = v.transpose(1, 2).contiguous().view(-1, len_v, self.d_k)

        if mask is not None:
            mask = mask.repeat(self.h, 1, 1) # (h*batch, len, len)

        # (batch*h, len, d_k)
        attn, w = self.attention(q, k, v, mask)
        # -> (batch, h, len, d_k)
        attn = attn.view(-1, self.h, len_q, self.d_k)
        # -> (batch, len, h, d_k) -> (batch, len, model_size)
        attn = attn.transpose(1, 2).contiguous().view(-1, len_q, self.model_size)

        attn = self.dropout(self.linear_out(attn))
        out = self.layer_norm(residual + attn)

        return out, w


# Position-wise Feed-Forward Networks
class Posfeedward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Conv1d(config.model_size, config.d_ff, 1)
        self.w2 = nn.Conv1d(config.d_ff, config.model_size, 1)
        # self.w1 = nn.Linear(config.model_size, config.d_ff)
        # self.w2 = nn.Linear(config.d_ff, config.model_size)

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(config.model_size, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # (batch, len, model_size)
        out = x.transpose(1, 2) # (batch, model_size, len)
        out = self.w2(self.relu(self.w1(out)))
        out = out.transpose(1, 2)
        out = self.dropout(out)
        out = self.layer_norm(x + out)

        return out
