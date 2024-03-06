#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import os 
import importlib
import data_analysis
importlib.reload(data_analysis)
import utils
importlib.reload(utils)
from utils import get_freer_gpu

device = get_freer_gpu() if torch.cuda.is_available() else "cpu"




class SimuDataset(Dataset):

    def __init__(self, simu, rgb = False, transform = None):
        X = simu.X
        if rgb : 
            X = simu.X_rgb
        self.x = torch.from_numpy(X)
        self.n_snapshots = X.shape[1]
        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.x[:,index]
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.n_snapshots
    
    def plot_field(self, t, save=False, directory=None):
        fig, ax = plt.subplots(figsize=(15, 5))
        vmin = self.T[t, :, :].min()
        vmax = self.T[t, :, :].max()
        abs_max = max(abs(vmin), abs(vmax))
        ax.streamplot(
            self.x.T,
            self.z.T,
            self.u[t, :, :].T,
            self.w[t, :, :].T,
            color="k",
            arrowsize=0.7,
            linewidth=1,
        )
        levels = np.linspace(vmin, vmax, 20)
        cf0 = ax.contourf(
            self.x,
            self.z,
            self.T[t, :, :],
            levels=levels,
            cmap=cm.Spectral.reversed(),
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        )
        cbar = plt.colorbar(cf0, ax=ax, shrink=0.35, aspect=6, ticks= [-abs_max, 0, abs_max])
        cbar.ax.set_aspect("auto")
        ax.set_title(f"Temperature and velocity field at t = {t}")
        ax.set_aspect("equal")
        ax.set_ylim(0, 1)
        ax.set_xlim(-4, 4)


        if save:
            plt.savefig(directory, dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

class SimpleLinearAutoencoder(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.encoder = nn.Linear(N, 128)
        
        self.decoder = nn.Sequential(
            nn.Linear(128, N), 
            nn.Tanh())
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":

    from data_analysis import Simulation
    from data_analysis import compatible_path
    current_directory = compatible_path('../')

    simulation = Simulation(current_directory, normalize = True)
    time, x, z, u, w, T, umean, wmean, Tmean = simulation.import_data()

    h, l = np.shape(x)
    m = len(time)
    print(m)
    N = h*l*3

    batch_size = 4

    dataset = SimuDataset(simulation)
    first = dataset[0]
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size)

    dataiter = iter(dataloader)
    data = next(dataiter)
    print(torch.min(data), torch.max(data), data.shape)

    model = SimpleLinearAutoencoder(N)
    model.to(device) 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

    num_epoch = 1
    output = []
    for epoch in range(num_epoch):
        for map in dataloader:
            map = map.float().to(device)
            reconstructed = model(map)
            loss = criterion(reconstructed, map)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}, Loss:{loss.item() : .4f}")
        output.append((epoch, map, reconstructed))