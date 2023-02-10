import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .utils import mse
from .scaler import GaussianScaler, MinmaxScaler
from sklearn.metrics import r2_score

class StandardTrainer:
    def __init__(self, net: nn.Module):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = self.to_device(net)
    
    def to_device(self, tensor: torch.Tensor):
        return tensor.to(self.device, non_blocking=True)
    
    def setup_dataloader(self, data_input: torch.Tensor, data_output: torch.Tensor):
        self.data_input_shape = data_input.shape
        self.data_output_shape = data_output.shape
        self.dataloader = DataLoader(TensorDataset(data_input, data_output),
                                     batch_size=256,
                                     shuffle=True,
                                     num_workers=4,
                                     pin_memory=True,
                                     persistent_workers=True)
        self.scaler_i = GaussianScaler(data_input)
        self.scaler_o = GaussianScaler(data_output)
        self.scaler_i.to(self.device)
        self.scaler_o.to(self.device)
    
    def setup_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    def fit(self, data_input: torch.Tensor, data_output: torch.Tensor, epochs: int = 1000, filename: str = 'net'):
        self.setup_dataloader(data_input, data_output)
        self.setup_optimizer(lr=1e-2)
        
        for epoch in range(epochs):
            total_y_true = torch.empty(self.data_output_shape)
            total_y_pred = torch.empty(self.data_output_shape)
            batchsizes = torch.zeros(len(self.dataloader), dtype=torch.int)
            for batch_idx, batch in enumerate(self.dataloader):
                x_true, y_true = batch
                batchsizes[batch_idx] = x_true.shape[0]
                
                x_true = self.to_device(x_true)
                y_true = self.to_device(y_true)
                
                x_true = self.scaler_i.scale(x_true)
                y_true = self.scaler_o.scale(y_true)
                
                y_pred = self.net(x_true)
                loss = mse(y_true, y_pred)
                
                for param in self.net.parameters():
                    param.grad = None
                loss.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    total_y_true[batchsizes[:batch_idx].sum().item():batchsizes[:batch_idx + 1].sum().item(),
                    :] = self.scaler_o.unscale(y_true).cpu()
                    total_y_pred[batchsizes[:batch_idx].sum().item():batchsizes[:batch_idx + 1].sum().item(),
                    :] = self.scaler_o.unscale(y_pred).cpu()
            
            total_loss = mse(total_y_true, total_y_pred)
            total_r2 = r2_score(total_y_true, total_y_pred)
            print(f"Epoch {epoch + 1}, MSE={total_loss.item():.5f}, R2={total_r2:.5f}")
        
        result = {'scaler_i': self.scaler_i, 'scaler_o': self.scaler_o, 'state_dict': self.net.state_dict()}
        torch.save(result, f'models/{filename}.pt')
