import torch
import torch.utils.data as data_util


def data_load(filename, batch_size, shuffle):
    data = torch.load(filename)
    data_loader = data_util.DataLoader(data, batch_size, shuffle=shuffle, num_workers=2)
    return data_loader


# convert idx to words, if idx <bos> is stop, return sentence
def index2sentence(index, idx2word):
    sen = []
    for i in range(len(index)):
        if idx2word[index[i]] == '<eos>':
            break
        if idx2word[index[i]] == '<bos>':
            continue
        else:
            sen.append(idx2word[index[i]])
    if len(sen) == 0:
        sen.append('<unk>')
    return sen