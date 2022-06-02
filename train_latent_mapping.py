from sklearn.utils import shuffle
from zmq import device
from latent_mapping import MLP

import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from configs import global_config

class NumpyPairDataset(Dataset):
    def __init__(self, np1, np2):
        super().__init__()
        self.data1 = torch.Tensor(np1)
        self.data2 = torch.Tensor(np2)

    def __getitem__(self, index):
        return self.data1[index], self.data2[index]
    
    def __len__(self):
        return self.data1.shape[0]

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(global_config.device), y.to(global_config.device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(x)
            print(f"loss: {loss:.5f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(global_config.device), y.to(global_config.device)

            pred = model(x)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
    test_loss /= num_batches
    print(f"TEST Avg loss: {test_loss:.5f}")

def main():
    w_codes = np.load('rand_data/w_plus_vectors.npy')
    exp_codes = np.load('rand_data/codes/exp.npy')

    train_dataset = NumpyPairDataset(w_codes[:9000], exp_codes[:9000])
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataset = NumpyPairDataset(w_codes[9000:], exp_codes[9000:])
    test_dataloader = DataLoader(test_dataset, batch_size=10)

    mlp = MLP.mlp_18().to(global_config.device)
    # mlp.load_state_dict(torch.load('latent_mapping/checkpoints/model_19.pth'))
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_mse_fn = nn.MSELoss(reduction='mean')

    epoches = 40
    for t in range(epoches):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, mlp, loss_mse_fn, optimizer)
        test(test_dataloader, mlp, loss_mse_fn)
        torch.save(mlp.state_dict(), f"latent_mapping/checkpoints/model_{t}.pth")
        print(f"Saved Model State to latent_mapping/checkpoints/model_{t}.pth")

    
    

if __name__ == "__main__":
    main()