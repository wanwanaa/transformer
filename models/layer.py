import torch
import torch.nn as nn
from models.sublayer import MultiHeadAttention, Posfeedward


# two layers
# multi-head self-attention mechanism + position-wise fully connected feed-forward network
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feedward = Posfeedward(config)

    def forward(self, x, non_pad_mask=None, attn_mask=None):
        attn, w = self.attention(x, x, x, mask=attn_mask)
        # attn (batch, len, model_size)
        # pad_mask (batch, len, 1)
        attn = attn*non_pad_mask.type(torch.float)

        enc_output = self.feedward(attn)
        enc_output = enc_output*non_pad_mask.type(torch.float)

        return enc_output, w


# three layers
# multi-head attention over encoder outputs + multi-head self-attention + feed-forward network
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.enc_attention = MultiHeadAttention(config)
        self.feedward = Posfeedward(config)

    def forward(self, dec_input, enc_output, non_pad_mask=None, attn_self_mask=None, enc_dec_attn_mask=None):
        dec_output, dec_self_w = self.self_attention(dec_input, dec_input, dec_input, mask=attn_self_mask)
        dec_output = dec_output * non_pad_mask.type(torch.float)

        dec_output, dec_w = self.enc_attention(dec_output, enc_output, enc_output, mask=enc_dec_attn_mask)
        dec_output = dec_output * non_pad_mask.type(torch.float)

        dec_output = self.feedward(dec_output)
        dec_output = dec_output * non_pad_mask.type(torch.float)

        return dec_output, dec_self_w, dec_w