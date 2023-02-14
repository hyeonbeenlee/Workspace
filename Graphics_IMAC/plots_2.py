from modules.net import Net
from modules.trainer import StandardTrainer
import pandas as pd
import torch

def df2tensor(data):
    data = torch.FloatTensor(data.to_numpy())
    if len(data.shape) == 1:
        data = data.unsqueeze(1)
    return data

def train_informedNet():
    data = pd.read_csv('data/01_manufactured_acceleration.csv')
    data_input = df2tensor(data[['x', 'y', 'yDot']])
    data_output = df2tensor(data['yDDot'])
    
    net = Net(3, 100, 1)
    trainer = StandardTrainer(net)
    trainer.fit(data_input, data_output, epochs=1000, filename='net_informed')

if __name__ == '__main__':
    train_informedNet()
