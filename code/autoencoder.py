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
import torch.nn.functional as F

import data_analysis
from data_analysis import Simulation
from data_analysis import compatible_path

import utils
from utils import get_freer_gpu
from utils import info_text
import argparse


    
class SimuDataset(Dataset):
    """Dataset for the flow field, gets data from the Simulation class.

    Arguments:
        Dataset -- class from Pytorch to processing data
    """


    def __init__(self, simu, device, training_ratio = 1, rgb = False, transform = None):
        """Initiate SimuDataset class

        Arguments:
            simu -- instance from Simulation class

        Keyword Arguments:
            training_ratio -- ratio of dataset to consider in training (default: {1})
            rgb -- 3 channels arrays (default: {False})
            transform -- dataset transfomation / normalisation to consider (default: {None})
        """
        X = simu.X[:, :int(training_ratio * simu.m)]
        if rgb : 
            X = simu.X_rgb[:, :int(training_ratio * simu.m), :, :]
        self.x = torch.from_numpy(X).to(device)
        self.n_snapshots = X.shape[1]
        self.transform = transform
        self.rgb = rgb
        
    def __getitem__(self, index):
        sample = self.x[:,index]
        t = index
        
        if self.transform:
            sample = self.transform(sample)
        if self.rgb:
            sample = self.x[:,index, :, :]
        return sample, t 
    
    def __len__(self):
        return self.n_snapshots
    
class SimpleLinearAutoencoder(nn.Module):
    """Simple Naive Linear Autoencoder """

    def __init__(self, N, device, K = 20):
        """Initiates SimpleLinearAutoencoder class

        Arguments:
            N -- Size of Input data

        Keyword Arguments:
            K -- Size of latent space (default: {20})
            device -- where to run code (default: {device})
        """
        super().__init__()
        self.encoder = nn.Linear(N, K).to(device)
        
        self.decoder = nn.Sequential(
            nn.Linear(K, N), 
            nn.Tanh()).to(device)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class ConvolutionalAutoencoder_v1(nn.Module):

    def __init__(self, device, n_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super().__init__()
        #shape : B * 3 * 81 * 51
        self.encoder = nn.Sequential(
            nn.Conv2d(3, n_channels//2, kernel_size, stride = stride, padding = padding, bias = bias), 
            nn.ReLU(), 
            nn.Upsample(size=size1, mode='bilinear', align_corners=False), 
            nn.Conv2d(n_channels//2, n_channels, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(),
            nn.Upsample(size=size2, mode='bilinear', align_corners=False), 
            nn.Conv2d(n_channels, n_channels, kernel_size, stride = stride, padding = padding, bias = bias), 
            nn.ReLU(),
            nn.Upsample(size=size3, mode='bilinear', align_corners=False)).to(device)
        
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels, n_channels//2, kernel_size, stride = stride, padding = padding, bias = bias), 
            nn.ReLU(), 
            nn.Upsample(size=size2, mode='bilinear', align_corners=False), 
            nn.ConvTranspose2d(n_channels//2, n_channels, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.ReLU(),
            nn.Upsample(size=size1, mode='bilinear', align_corners=False), 
            nn.ConvTranspose2d(n_channels, 3, kernel_size, stride = stride, padding = padding, bias = bias), 
            nn.Upsample(size=(81, 51), mode='bilinear', align_corners=False)).to(device)
        
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvolutionalAutoencoder_v2(nn.Module):

    def __init__(self, device, sizes, n_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super().__init__()
        #shape : B * 3 * 81 * 51
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.device = device
        self.n_channels = n_channels
        self.sizes = sizes


        self.encoder = nn.Sequential(
            self.conv_block(3, self.n_channels), 
            nn.Upsample(size=sizes[0], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[1], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[2], mode='bilinear', align_corners=False), 
            
            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[3], mode='bilinear', align_corners=False),
            
            nn.Conv2d(n_channels, n_channels, kernel_size, stride = stride, padding = padding, bias = bias)).to(device)
        
        self.decoder = nn.Sequential(
            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[2], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[1], mode='bilinear', align_corners=False),

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[0], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=(81, 51), mode='bilinear', align_corners=False), 
            
            nn.ConvTranspose2d(n_channels, 3, kernel_size, stride = stride, padding = padding, bias = bias),
            nn.Tanh()).to(device)
    
    def conv_block(self, ch_in, ch_out):
         return nn.Sequential(nn.Conv2d(ch_in, ch_out, self.kernel_size, stride = self.stride, padding = self.padding, bias = self.bias),
                              nn.BatchNorm2d(ch_out),
                              nn.ReLU(), 
                              nn.Conv2d(ch_out, ch_out, self.kernel_size, stride = self.stride, padding = self.padding, bias = self.bias), 
                              nn.BatchNorm2d(ch_out),
                              nn.ReLU())
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class ConvolutionalAutoencoder(nn.Module):

    def __init__(self, device, sizes, K = 128, n_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super().__init__()
        #shape : B * 3 * 81 * 51
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.device = device
        self.n_channels = n_channels
        self.sizes = sizes
        self.N = n_channels * sizes[-1][0] * sizes[-1][1]
        self.K = K


        self.encoder = nn.Sequential(
            self.conv_block(3, self.n_channels), 
            nn.Upsample(size=sizes[0], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[1], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[2], mode='bilinear', align_corners=False), 
            
            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[3], mode='bilinear', align_corners=False),
            
            nn.Conv2d(n_channels, n_channels, kernel_size, stride = stride, padding = padding, bias = bias)).to(device)
        
        self.decoder = nn.Sequential(
            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[2], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[1], mode='bilinear', align_corners=False),

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[0], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=(81, 51), mode='bilinear', align_corners=False), 
            
            nn.ConvTranspose2d(n_channels, 3, kernel_size, stride = stride, padding = padding, bias = bias)).to(device)
        
        self.linear_encoder = nn.Linear(self.N, self.K).to(device)

        self.linear_decoder = nn.Sequential(nn.Linear(K, self.N), nn.Tanh()).to(device)
    
    def conv_block(self, ch_in, ch_out):
         return nn.Sequential(nn.Conv2d(ch_in, ch_out, self.kernel_size, stride = self.stride, padding = self.padding, bias = self.bias),
                              nn.BatchNorm2d(ch_out),
                              nn.ReLU(), 
                              nn.Conv2d(ch_out, ch_out, self.kernel_size, stride = self.stride, padding = self.padding, bias = self.bias), 
                              nn.BatchNorm2d(ch_out),
                              nn.ReLU())

    def forward(self, x):
        encoded = self.encoder(x)
        # print(encoded.shape)
        encoded = torch.reshape(encoded, (encoded.shape[0],-1))
        # print(encoded.shape)
        encoded = self.linear_encoder(encoded)
        # print(encoded.shape)

        
        decoded = self.linear_decoder(encoded)
        # print(decoded.shape)
        decoded = torch.reshape(decoded, (decoded.shape[0], self.n_channels, self.sizes[-1][0], self.sizes[-1][1]))
        # print(decoded.shape)
        decoded = self.decoder(decoded)
        return decoded


def main(K = 128, batch_size = 4):
    device = get_freer_gpu() if torch.cuda.is_available() else "cpu"

#Dataloading
    current_directory = compatible_path('../')

    simulation = Simulation(current_directory, normalize = True)
    time, x, z, u, w, T, umean, wmean, Tmean = map(lambda x: torch.tensor(x).to(device), simulation.import_data())
    simulation.image_rgb()

    h, l = np.shape(x)
    m = len(time)

#Setting  Dataset
    batch_size = batch_size

    dataset = SimuDataset(simulation, device,  rgb = True, training_ratio = 0.9)
    first = dataset[0]
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    dataiter = iter(dataloader)
    data, t = next(dataiter)
    print(torch.min(data), torch.max(data), data.shape)

# Model
    lr = 1e-4
    weight_decay = 0

    size1 = (40, 25)
    size2 = (20, 12)
    size3 = (10, 6)
    size4 = (5, 3)


    sizes = [size1, size2, size3, size4]
    print(K)
    model = ConvolutionalAutoencoder(device = device, n_channels = 64,  sizes = sizes, K = K)
    model.to(device) 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

#Training
    num_epoch = 100
    output = []
    info = []
    for epoch in range(num_epoch):
        for field, t in dataloader:
            field = field.float().to(device)
            reconstructed = model(field)
            loss = criterion(reconstructed, field)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}, Loss:{loss.item() : .4f}")
        info.append((epoch, loss.item()))
        output.append((epoch, field, reconstructed))

#Saving
    print(f"batchsize = {batch_size}")
    directory = f'{current_directory}/results/autoencoder/cnn/'
    torch.save(model.state_dict(), directory + f'model_bs{batch_size}_K{K}.pt')
    print("model saved !")
    info_text(directory, batch_size, sizes, info, version = 3, K = K, num_epoch = num_epoch, lr = lr, weight_decay = weight_decay, n_channels = model.n_channels)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script with parameter K')
    parser.add_argument('--K', type=int, default = 960, help='Value for K (default: 960)')
    parser.add_argument('--bs', type=int, default = 4, help='Value for batchsize (default: 4)')
    
    args = parser.parse_args()
    K = args.K
    bs = args.bs

    print(f"K = {K}")
    print(f"batchsize = {bs}")
    main(K = K, batch_size = bs)