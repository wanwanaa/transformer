import torch
import torch.nn as nn
import torch.nn.functional as F


# implement label smoothing KL
class LabelSmoothing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.ls = config.ls
        self.tgt_vocab_size = config.tgt_vocab_size
        self.pad = config.pad

    def forward(self, out, y):
        # out (batch, len, vocab_size)
        # y (batch, len)
        y = y.view(-1)
        word = y.ne(self.pad).sum().item()
        out = out.view(-1, self.tgt_vocab_size)

        true_dist = torch.zeros_like(out)
        true_dist.fill_(self.ls / (self.tgt_vocab_size-2))

        true_dist.scatter_(1, y.unsqueeze(1), (1-self.ls))

        true_dist[:, self.pad] = 0

        mask = torch.nonzero(y == self.pad)
        true_dist = true_dist.transpose(0, 1)
        true_dist.index_fill_(1, mask.squeeze(), 0.0)
        true_dist = true_dist.transpose(0, 1)

        out = torch.nn.functional.log_softmax(out, dim=-1)
        loss = self.criterion(out, true_dist)
        # loss = F.binary_cross_entropy_with_logits(out, true_dist, size_average=False)

        return loss/word


# implement label smoothing one-hot
class LabelSmoothing_Onehot(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ls = config.ls
        self.vocab_size = config.vocab_size
        self.pad = config.pad

    def forward(self, out, y):
        # out (batch, len, vocab_size)
        # y (batch, len)
        out = out.view(-1, self.vocab_size)
        out = torch.nn.functional.log_softmax(out, dim=-1)
        y = y.view(-1)

        one_hot = torch.zeros_like(out).scatter(1, y.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.ls) + (1 - one_hot) * self.ls / (self.vocab_size - 1)

        pad_mask = y.ne(self.pad)
        word = pad_mask.sum().item()
        loss = -(one_hot * out).sum(dim=1)
        loss = loss.masked_select(pad_mask).sum()

        return loss/word