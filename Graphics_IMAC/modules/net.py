import torch.nn as nn
import torch.nn.functional as F
from . import utils

class Net(nn.Module):
    def __init__(self, input_dim, width, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, output_dim)
        self.apply(utils.initializer)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        return x
