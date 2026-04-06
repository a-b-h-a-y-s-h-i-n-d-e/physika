import torch


def append(arr, val):
    return torch.cat([arr, torch.tensor([val])])

def prepend(arr, val):
    return torch.cat([torch.tensor([val]), arr])

def zeros_array(*args):
    return torch.zeros(args)
