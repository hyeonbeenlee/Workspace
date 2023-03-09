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
    trainer.fit(data_input, data_output, epochs=5000, filename='net_informed')

def visualize_informedNet():
    data = pd.read_csv('data/01_manufactured_acceleration.csv')
    data_input = df2tensor(data[['x', 'y', 'yDot']])
    data_output = df2tensor(data['yDDot'])
    
    result = torch.load('models/net_informed.pt')
    net = Net(3, 100, 1)
    net.load_state_dict(result['state_dict'])
    scaler_i = result['scaler_i'].to('cpu')
    scaler_o = result['scaler_o'].to('cpu')
    
    prediction = scaler_o.unscale(net(scaler_i.scale(data_input))).detach()
    
    import MyPackage as mp
    import matplotlib.pyplot as plt
    mp.visualize.PlotTemplate(20)
    fig, axes = plt.subplots(1, 1, figsize=(9, 6))
    styleTrue = dict(linestyle='dashed', linewidth=2, color='grey')
    stylePred = dict(linewidth=1.5, color='black')
    axes.plot(data['x'], data_output.flatten(), **styleTrue)
    axes.plot(data['x'], prediction.flatten(), **stylePred)
    axes.set_xlabel('$t$')
    axes.set_ylabel('$\ddot{y}$')
    import numpy as np
    axes.set_yticks(np.linspace(-500,500,6, endpoint=True))
    fig.tight_layout()
    plt.show()

def visualize_data():
    data = pd.read_csv('data/01_manufactured_acceleration.csv')
    data_input = df2tensor(data[['x', 'y', 'yDot']])
    data_output = df2tensor(data['yDDot'])
    data=torch.cat([data_input,data_output], dim=1)
    import MyPackage as mp
    import matplotlib.pyplot as plt
    mp.visualize.PlotTemplate(20)
    style=dict(color='grey', linestyle='dashed', linewidth=1.5)
    fig, axes = plt.subplots(3, 1, figsize=(6, 9))
    ylabels=['$y$','$\dot{y}$', '$\ddot{y}$']
    for i in range(3):
        axes[i].plot(data_input[:,0],data[:,i+1], **style)
        axes[i].set(xticks=torch.linspace(data_input[0,0], data_input[-1,0],6), ylabel=ylabels[i])
    axes[2].set_xlabel('$t$')
    fig.tight_layout()
    plt.show()




if __name__ == '__main__':
    # train_informedNet()
    visualize_informedNet()
    # visualize_data()
