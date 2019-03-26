import torch
import torch.utils.data as data_util


def data_load(filename, batch_size, shuffle):
    data = torch.load(filename)
    data_loader = data_util.DataLoader(data, batch_size, shuffle=shuffle, num_workers=2)
    return data_loader