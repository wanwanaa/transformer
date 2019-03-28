import torch
import torch.nn as nn
from models.layer import EncoderLayer, DecoderLayer
import math


def get_non_pad_mask(seq, pad):
    # (batch, len)
    mask = seq.ne(pad)
    # (batch, len, 1)
    mask = mask.unsqueeze(-1)
    return mask


def get_pad_mask(seq_k, seq_q, pad):
    len_q = seq_q.size(1)
    mask = seq_k.eq(pad)
    mask = mask.unsqueeze(1).expand(-1, len_q, -1)
    return mask


def get_attn_pad_mask(seq, pad):
    # (batch, len)
    len = seq.size(1)
    pad_mask = seq.eq(pad)
    # (batch, len, len)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len, -1)
    return pad_mask


# decoder self-attention
def get_dec_mask(seq):
    batch, len = seq.size()
    # (len, len)
    mask = torch.triu(
        torch.ones((len, len), dtype=torch.uint8), diagonal=1
    )
    if torch.cuda.is_available():
        mask = mask.type(torch.cuda.ByteTensor)
    # (batch, len, len)
    mask = mask.unsqueeze(0).expand(batch, -1, -1)
    return mask


def positional_encoding(len, model_size, pad):
    # (len, model_size)
    pe = torch.zeros(len, model_size)
    position = torch.arange(0., len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., model_size, 2) *
                         (-math.log(10000.0) / model_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # pe[pad] = 0.

    return pe


# implement label smoothing KL
class LabelSmoothing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.ls = config.ls
        self.vocab_size = config.vocab_size
        self.pad = config.pad

    def forward(self, out, y):
        # out (batch, len, vocab_size)
        # y (batch, len)
        y = y.view(-1)
        word = y.ne(self.pad).sum().item()
        out = out.view(-1, self.vocab_size)

        true_dist = torch.zeros_like(out)
        true_dist.fill_(self.ls / (self.vocab_size-1))

        true_dist.scatter_(1, y.unsqueeze(1), (1-self.ls))

        mask = torch.nonzero(y == self.pad)
        true_dist.index_fill_(1, mask.squeeze(), 0.0)

        loss = self.criterion(out, true_dist)
        return loss/word


# # implement label smoothing one-hot
# class LabelSmoothing(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.ls = config.ls
#         self.vocab_size = config.vocab_size
#         self.pad = config.pad
#
#     def forward(self, out, y):
#         # out (batch, len, vocab_size)
#         # y (batch, len)
#         out = out.view(-1, self.vocab_size)
#         y = y.view(-1)
#
#         one_hot = torch.zeros_like(out).scatter(1, y.view(-1, 1), 1)
#
#         one_hot = one_hot * (1 - self.ls) + (1 - one_hot) * self.ls / (self.vocab_size - 1)
#
#         pad_mask = y.ne(self.pad)
#
#         word = pad_mask.sum().item()
#         loss = -(one_hot * out).sum(dim=1)
#         loss = loss.masked_select(pad_mask).sum()
#
#         return loss/word


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
            positional_encoding(config.max_len, config.model_size, config.pad), freeze=True
        )

        self.encoder_stack = nn.ModuleList([
            EncoderLayer(config) for _ in range(self.n_layer)
        ])

    def forward(self, x, pos):
        # mask
        pad_mask = get_non_pad_mask(x, self.pad) # (batch, len, 1)
        attn_mask = get_attn_pad_mask(x, self.pad) # (batch, len, len)

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
            positional_encoding(config.max_len, config.model_size, config.pad), freeze=True
        )

        self.decoder_stack = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layer)
        ])

    def forward(self, x, y, pos, enc_output):
        pad_mask = get_non_pad_mask(y, self.pad)
        attn_mask = get_dec_mask(y)
        attn_self_mask = (pad_mask + attn_mask).gt(0)
        pad_attn_mask = get_pad_mask(x, y, self.pad)

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
        self.s_len = config.s_len
        self.bos = config.bos

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.linear_out = nn.Sequential(
            nn.Linear(self.model_size, self.vocab_size),
            nn.LogSoftmax(-1)
        )

        self.loss_func = nn.CrossEntropyLoss()

        # share the same weight matrix between the encoder and decoder embedding
        self.encoder.embedding.weight = self.decoder.embedding.weight

        self.smoothing = LabelSmoothing(config)

    # add <bos> to sentence
    def convert(self, x):
        """
        :param x:(batch, s_len) (word_1, word_2, ... , word_n)
        :return:(batch, s_len) (<bos>, word_1, ... , word_n-1)
        """
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    def compute_loss(self, result, y):
        result = result.contiguous().view(-1, 4000)
        y = y.contiguous().view(-1)
        loss = self.loss_func(result, y)
        return loss

    def beam_sample(self, x, x_pos, y, y_pos):
        pass

    def sample(self, x, x_pos, y, y_pos):
        enc_output = self.encoder(x, x_pos)
        out = torch.ones(x.size(0)) * self.bos
        result = None
        out = out.unsqueeze(1)
        for i in range(self.s_len):
            # print(out)
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)
            # print(y_pos[:, :(i+1)])
            dec_output = self.decoder(x, out, y_pos[:, :(i+1)], enc_output)
            gen = self.linear_out(dec_output[:, -1, :])
            # print(dec_output.size())
            gen = torch.argmax(gen, dim=1).unsqueeze(1)
            out = torch.cat((out, gen), dim=1)
            result = dec_output
        result = self.linear_out(result)
        idx = torch.argmax(result, dim=2)
        loss = self.smoothing(result, y)
        return idx, loss

    def forward(self, x, x_pos, y, y_pos):
        # y, y_pos = y[:, :-1], y_pos[:, :-1]
        enc_output = self.encoder(x, x_pos)

        gold = y
        y = self.convert(y)

        dec_output = self.decoder(x, y, y_pos, enc_output)

        out = self.linear_out(dec_output)

        loss = self.smoothing(out, gold)
        # loss = self.compute_loss(out, gold)

        return out, loss
