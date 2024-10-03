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
import time

import data_analysis
from data_analysis import Simulation
from data_analysis import compatible_path

import utils
from utils import get_freer_gpu
from utils import info_text
import argparse
from training import training
from training import training_time
from training import SimuDataset

    

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



class ConvolutionalAutoencoder_v3(nn.Module):

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
            
            self.conv_block(self.n_channels, 3)).to(device)
        
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
            
            self.conv_block(self.n_channels, self.n_channels)).to(device)
        
        self.decoder = nn.Sequential(
            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[2], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[1], mode='bilinear', align_corners=False),

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[0], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=(81, 51), mode='bilinear', align_corners=False), 
            
            self.conv_special_block(self.n_channels, 3)).to(device)
        
        self.linear_encoder = nn.Sequential(nn.Linear(self.N, self.N), nn.Linear(self.N, self.K)).to(device)
        self.linear_decoder = nn.Sequential(nn.Linear(self.K, self.N)).to(device)

    
    def conv_block(self, ch_in, ch_out):
         return nn.Sequential(nn.Conv2d(ch_in, ch_out, self.kernel_size, stride = self.stride, padding = self.padding, bias = self.bias),
                              nn.BatchNorm2d(ch_out),
                              nn.ReLU(), 
                              nn.Conv2d(ch_out, ch_out, self.kernel_size, stride = self.stride, padding = self.padding, bias = self.bias), 
                              nn.BatchNorm2d(ch_out),
                              nn.ReLU())
    
    def conv_special_block(self, ch_in, ch_out):
         return nn.Sequential(nn.Conv2d(ch_in, ch_in, self.kernel_size, stride = self.stride, padding = self.padding, bias = self.bias),
                              nn.BatchNorm2d(ch_in),
                              nn.ReLU(), 
                              nn.Conv2d(ch_in, ch_out, self.kernel_size, stride = self.stride, padding = self.padding, bias = self.bias), 
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


class ConvolutionalAutoencoder_debug(nn.Module):

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
            
            self.conv_block(self.n_channels, self.n_channels)).to(device)
        
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
        
        self.linear_encoder = nn.Sequential(nn.Linear(self.N, self.N), nn.Linear(self.N, self.K)).to(device)
        self.linear_decoder = nn.Sequential(nn.Linear(self.K, self.N)).to(device)

    
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

class ConvolutionalAutoencoder_sizes(nn.Module):

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

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[4], mode='bilinear', align_corners=False),

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[5], mode='bilinear', align_corners=False),
            
            self.conv_block(self.n_channels, self.n_channels)).to(device)
        
        self.decoder = nn.Sequential(
            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[4], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[3], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[2], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[1], mode='bilinear', align_corners=False),

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=sizes[0], mode='bilinear', align_corners=False), 

            self.conv_block(self.n_channels, self.n_channels), 
            nn.Upsample(size=(81, 51), mode='bilinear', align_corners=False), 
            
            nn.ConvTranspose2d(n_channels, 3, kernel_size, stride = stride, padding = padding, bias = bias)).to(device)
        
        self.linear_encoder = nn.Sequential(nn.Linear(self.N, self.N), nn.Linear(self.N, self.K)).to(device)
        self.linear_decoder = nn.Sequential(nn.Linear(self.K, self.N)).to(device)

    
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
    

def main(K = 128, batch_size = 4, lr = 1e-4, n_channels = 64, training_mode = "num_epoch", Lambda = 1e-2, training_ratio = 0.9):
    """Main exec function

    Keyword Arguments:
        K -- size of latent space (default: {128})
        batch_size -- size of the batch (default: {4})
        lr -- learning rate (default: {1e-4})
        n_channels -- number of channels (default: {64})
        training_mode -- by epoch or by time (default: {"num_epoch"})
    """
    device = get_freer_gpu() if torch.cuda.is_available() else "cpu"

#Dataloading
    current_directory = compatible_path('../')

    simulation = Simulation(current_directory, normalize = True, Lambda = Lambda)
    time_array, x, z, u, w, T, umean, wmean, Tmean = map(lambda x: torch.tensor(x).to(device), simulation.import_data(mean_type= 'Znorm'))
    simulation.image_rgb()

    h, l = np.shape(x)
    m = len(time_array)
    epoch_30 = None
    print("ABOUT DATA")
    print(f"Lambda = {Lambda}")
    print(f"number of snapshots : {m}")

#Setting  Dataset
    training_ratio = training_ratio
    batch_size = batch_size
    dataset = SimuDataset(simulation, device,  rgb = True, training_ratio = training_ratio)
    first = dataset[0]
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    valset = SimuDataset(simulation, device,  rgb = True, mode = 'val', training_ratio = training_ratio)
    valloader = DataLoader(dataset = valset, batch_size = 4, shuffle = True)
# Model
    weight_decay = 0
    print("MODEL PARAMETERS")
    print(f"n_channels = {n_channels}")
    print(f"K = {K}")
    print(f"batch_size = {bs}")
    print(f"lr = {lr}")

    size1 = (40, 25)
    size2 = (20, 12)
    size3 = (10, 6)
    size4 = (5, 3)

    size1 = (50, 30)
    size2 = (30, 20)
    size3 = (20, 15)
    size4 = (15, 10)

    sizes = [size1, size2, size3, size4]
    # sizes = [size1, size2, size3, size4, size5, size6]
    # model = ConvolutionalAutoencoder_debug(device = device, n_channels = n_channels,  sizes = sizes, K = K)
    model = ConvolutionalAutoencoder_debug(device = device, n_channels = n_channels,  sizes = sizes, K = K)
    model.to(device) 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

#Training
    print(f"training mode : {training_mode}")
    num_epoch = 100
    if training_mode == "num_epoch":
        model, info, execution_time = training(dataloader, valloader, model, criterion, optimizer, device, num_epoch = num_epoch, noisy = False)

    if training_mode == "time":
        model, info, execution_time = training_time(dataloader, valloader, model, criterion, optimizer, device, training_time = 20, time_reached = False)

#Saving
    # print(f"batchsize = {batch_size}")
    filename = f'model_bs{batch_size}_K{K}_lr{lr}_nc{n_channels}'
    directory = f'{current_directory}/results/autoencoder/cnn/noisy'
    if Lambda == 0.0 :
        directory = f'{current_directory}/results/autoencoder/cnn/sizes/'
    if training_mode == "time" :
        directory = f'{current_directory}/results/autoencoder/cnn/time/'

    torch.save(model.state_dict(), directory + filename +'.pt')
    print("model saved !")
    info_text(directory, batch_size, info, title = filename + "_info.txt",  thirty_min_epoch = epoch_30, sizes = sizes,  K = K, num_epoch = num_epoch, lr = lr, weight_decay = weight_decay, n_channels = n_channels, execution_time_in_min = execution_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for autoencoder')
    parser.add_argument('--K', type=int, default = 128, help='Value for K (default: 960)')
    parser.add_argument('--bs', type=int, default = 4, help='Value for batchsize (default: 4)')
    parser.add_argument('--n_channels', type=int, default = 64, help='Value for n_channels (default: 64)')
    parser.add_argument('--lr', type=float, default = 1e-4, help='Value for learning rate (default: 1e-4)')
    parser.add_argument('--training_mode', type=str, default = "num_epoch", help='Value for training mode (default: "num_epoch")')
    parser.add_argument('--Lambda', type=float, default = 1e-2, help='Value for Lambda (default: 1e-2)')
    parser.add_argument('--training_ratio', type=float, default = 0.9, help='Value for training ratio (default: 0.9)')
    
    args = parser.parse_args()
    K = args.K
    bs = args.bs
    lr = args.lr
    n_channels = args.n_channels
    training_mode = args.training_mode
    Lambda = args.Lambda

    main(K = K, batch_size = bs, lr = lr, n_channels = n_channels, training_mode = training_mode, Lambda = Lambda)