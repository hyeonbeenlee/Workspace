import torch
from modules.trainer import StandardTrainer
from modules.net import Net
import matplotlib.pyplot as plt
from modules.scaler import GaussianScaler
from torch import pi

def f(t, w=1, A=1, phi=0):
    return A * torch.sin(w * t + phi)

def gen_data():
    n_timesteps = 1000
    n_frequencies = 1000
    t = torch.linspace(0, 20, n_timesteps).reshape(-1, 1)
    y = f(t, 1, 1, torch.rand(1) * pi - pi / 2)
    w = torch.linspace(0, 4, n_frequencies)
    for i in range(n_frequencies):
        y += f(t, w[i], 1, torch.rand(1) * 2 * pi - pi)
    # y[:int(0.3 * n_timesteps)] = 0
    # y[int(0.7 * n_timesteps):n_timesteps] = 0
    torch.save({'t': t, 'y': y}, f'data/{filename}.pt')
    
    plt.plot(t,y)
    plt.show()

def train(filename: str, epochs: int):
    net = Net(1, 100, 1)
    trainer = StandardTrainer(net)
    trainer.fit(t, y, epochs=epochs, filename=filename)

def load(filename):
    net = Net(1, 100, 1)
    model = torch.load(f'models/{filename}.pt')
    net.load_state_dict(model['state_dict'])
    net.eval()
    scaler_i = model['scaler_i'].to('cpu')
    scaler_o = model['scaler_o'].to('cpu')
    with torch.no_grad():
        prediction = scaler_o.unscale(net(scaler_i.scale(t)))
    
    fig, axes = plt.subplots(figsize=(11, 5))
    axes.plot(t, y, linewidth=1, color='k', label='Training Data', marker='o',markersize=4)
    axes.plot(t, prediction, color='r', linewidth=3, label='NN')
    axes.legend(loc=4, fontsize=15)
    axes.set_xticks([])
    axes.set_xlabel('Time', fontsize=15)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    filename = 'net_impulse'
    gen_data()
    
    t=torch.load(f'data/{filename}.pt')['t']
    y=torch.load(f'data/{filename}.pt')['y']
    # train(filename, epochs=500)
    load(filename)
