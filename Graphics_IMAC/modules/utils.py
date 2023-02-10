import torch
import torch.nn as nn

def initializer(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

def mse(true, pred):
    return torch.mean(torch.square(true - pred))
