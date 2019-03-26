import torch
import torch.nn as nn
from models.layer import EncoderLayer, DecoderLayer
import math


def get_pad_mask(seq, pad):
    # (batch, len)
    mask = seq.eq(pad)
    # (batch, len, 1)
    mask = mask.unsquenze(-1)
    return mask


def get_attn_pad_mask(seq, pad):
    # (batch, len)
    len = seq.size(1)
    pad_mask = seq.eq(pad)
    # (batch, len, len)
    pad_mask = pad_mask.unsquenze(1).expand(-1, len, -1)
    return pad_mask


# decoder self-attention
def get_dec_mask(seq):
    batch, len = seq.size()
    # (len, len)
    mask = torch.triu(
        torch.ones((len, len), dtype=torch.uint8), diagonal=1
    )
    # (batch, len, len)
    mask = mask.unsqueeze(0).expand(batch, -1, -1)
    return mask


def positional_encoding(len, model_size, pad):
    # (len, model_size)
    pe = torch.zeros(len, model_size)
    position = torch.arange(0, len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, model_size, 2) *
                         (-math.log(10000.0) / model_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe[pad] = 0.
    return pe


# implement label smoothing
class LabelSmoothing(nn.Module):
    def __init__(self):


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layer = config.n_layer
        self.vocab_size = config.vocab_size
        self.model_size = config.model_size
        self.pad = config.pad

        # word embedding
        self.embedding = nn.Embedding(self.vocab_size, self.model_size)

        # positional Encoding
        self.position_enc = nn.Embedding.from_pretrained(
            positional_encoding(config.len+1, config.model_size, config.pad), freeze=True
        )

        self.encoder_stack = nn.ModuleList([
            EncoderLayer(config) for _ in range(self.n_layer)
        ])

    def forward(self, x, pos):
        # mask
        attn_mask = get_pad_mask(x, self.pad)
        pad_mask = get_attn_pad_mask(x, self.pad)

        enc_output = self.embedding(x) + self.position_enc(pos)
        for layer in self.encoder_stack:
            enc_output, enc_attn = layer(enc_output, pad_mask, attn_mask)

        return enc_output


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad = config.pad

        self.embedding = nn.Embedding(config.vocab_size, config.model_size)

        self.position_dec = nn.Embedding.from_pretrained(
            positional_encoding(config.len + 1, config.model_size, config.pad), freeze=True
        )

        self.decoder_stack = nn.ModuleList([
            DecoderLayer(config) for _ in range(self.n_layer)
        ])

    def forward(self, y, pos, enc_output):
        pad_mask = get_pad_mask(y, self.pad)
        pad_attn_mask = get_pad_mask(y, self.pad)
        attn_mask = get_dec_mask(y)
        attn_self_mask = pad_attn_mask + attn_mask

        dec_output = self.embedding(y) + self.position_dec(pos)
        for layer in self.decoder_stack:
            dec_output, _, _ = layer(dec_output, enc_output, pad_mask, attn_self_mask, pad_attn_mask)

        return dec_output


# A sequence to sequence model with attention mechanism.
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_size = config.model_size
        self.vocab_size = config.vocab_size

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.linear_out = nn.Sequential(
            nn.Linear(self.model_size, self.vocab_size),
            nn.Softmax(-1)
        )

        self.loss_func = nn.CrossEntropyLoss()

        # share the same weight matrix between the encoder and decoder embedding
        self.encoder.embedding.weight = self.decoder.embedding.weight

    def compute_loss(self, out, y):
        y = y.view(-1)
        out = out.view(-1, self.vocab_size)
        loss = self.loss_func(out, y)
        return loss

    def forward(self, x, x_pos, y, y_pos):
        # y, y_pos = y[:, :-1], y_pos[:, :-1]
        enc_output = self.encoder(x, x_pos)
        dec_output = self.decoder(y, y_pos, enc_output)
        out = self.linear_out(dec_output)

        loss = self.compute_loss(out, y)
        return out, loss