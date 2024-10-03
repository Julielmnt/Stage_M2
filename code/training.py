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
from tqdm import tqdm



class SimuDataset(Dataset):
    """Dataset for the flow field, gets data from the Simulation class.

    Arguments:ssh 
        Dataset -- class from Pytorch to processing data
    """


    def __init__(self, simu, device, training_ratio=1, rgb=False, transform=None, mode='train', len_val=50, only_velocities=False, pandey_data=False):
        """Initiate SimuDataset class

        Arguments:
            simu -- instance from Simulation class

        Keyword Arguments:
            training_ratio -- ratio of dataset to consider in training (default: {1})
            rgb -- 3 channels arrays (default: {False})
            transform -- dataset transfomation / normalisation to consider (default: {None})
        """
        self.pandey = pandey_data
        print(f"pandey_data = {pandey_data}")
        print(f"rgb = {rgb}")

        if only_velocities == False:
            if rgb:
                X = simu.X_rgb
            else:
                X = simu.X
        if only_velocities:
                X = simu.velocities

        if mode == 'train':
            X = X[int(training_ratio * simu.m)]

        if mode == 'val':
            test_set = X[int(training_ratio * simu.m):]

            num_images = test_set.shape[0]
            random_selection = np.random.choice(num_images, size = len_val, replace = False)
            X = test_set[random_selection]
            
        if mode == 'test':
            X = X[int(training_ratio * simu.m):]

        if mode == 'whole':
            pass

        print(f"X SHAPE {X.shape}")
        self.x = torch.from_numpy(X).to(device)
        self.n_snapshots = X.shape[0]
        self.transform = transform
        self.rgb = rgb
        
    def __getitem__(self, index):
        t = index
        sample = self.x[index,:]
        if self.transform:
            sample = self.transform(sample)
                # print(f"sample shape {sample.shape}")
        return sample, t 
    
    def __len__(self):
        return self.n_snapshots
    
    
def test(model, test_dataloader, device, metric = "nmse"):
        
        model.eval()
        num_samples = 0
        measure = 0
        with torch.no_grad():
            for field, t in test_dataloader:
                field = field.float().to(device)
                reconstructed = model(field)

                if metric == "nmse":
                    nmse = compute_nmse(reconstructed, field)
                    measure += nmse.sum()
                num_samples += field.size(0)
        
        metric = measure / num_samples

        return metric


def compute_nmse(outputs, targets):
    outputs = outputs.reshape(outputs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    mse = torch.mean((outputs - targets)**2, dim = 1)
    mse_original = torch.mean(targets**2, dim = 1)
    nmse = mse / mse_original
    return nmse.detach().cpu().numpy()


def training(dataloader, valloader, model, criterion, optimizer,  device, scheduler = None, num_epoch = 100, time_reached = False, noisy = False,  wandb_on=False, regular_save=False, filename = None, directory = None):
    if noisy :
        print("noisy !")
    output = []
    info = []
    start_time = time.time()
    time_reached = time_reached
    print(f'num_epochs = {num_epoch}')

    # for epoch in tqdm(range(num_epoch)):
    for epoch in tqdm(range(num_epoch), bar_format="{desc}: {percentage:3.0f}%|{bar}|"):

        total_loss = 0.0
        num_samples = 0
        total_nmse_train = 0
        model.train()
        for field, t in dataloader:
            # print(f"FIELD SHAPE {field.shape}")
            optimizer.zero_grad()
            field = field.float().to(device)
            if noisy == False:
                reconstructed = model(field)
            if noisy:
                reconstructed = model(field + torch.randn_like(field)*.1) #adding noise
            # print(f"RECONSTRUCTED SHAPE {reconstructed.shape}")
            loss = criterion(reconstructed, field)
            total_loss += loss.item()



            loss.backward()
            optimizer.step()
            num_samples += field.size(0)

            with torch.no_grad():
                nmse_train = compute_nmse(reconstructed, field).sum()
                total_nmse_train += nmse_train

        average_loss = total_loss / num_samples
        nmse_train = total_nmse_train / num_samples
        num_samples = 0
        
    

        #validation
        average_nmse = test(model, valloader, device=device, metric='nmse')

        if scheduler is not None:
            scheduler.step()
        
        if regular_save :
            if epoch % 10 == 0 :
                torch.save(model.state_dict(), directory + "/regular_save/" + filename + f"epoch_{epoch}" + ".pt")

        if wandb_on:
            import wandb
            wandb.log({"loss": loss.item(), "nmse": average_nmse, "nmse_train" : nmse_train})

        current_time = time.time()
        time_epoch = (current_time - start_time)/ 60
        print(f"Epoch: {epoch+1}, Loss:{loss.item() : .4f}, nmse = {average_nmse*100 : .1f}%, nmse_train = {nmse_train*100 : .1f}%,  time = {round(time_epoch, 4)} min")
        info.append((epoch, loss.item(), average_nmse, time_epoch, average_loss, nmse_train))
        output.append((epoch, field, reconstructed))

        if (current_time - start_time) / 60 > 30 and time_reached == False:
            print(f"We just reached 30 min !")
            epoch_30 = epoch
            time_reached = True

        end_time = time.time()
        execution_time = (end_time - start_time) / 60

    return model, info, execution_time



def training_time(dataloader, valloader, model, criterion, optimizer, device, time_reached = False, training_time = 30):
    output = []
    info = []
    start_time = time.time()
    time_reached = time_reached

    for epoch in range(100):
        num_batches = 0
        total_loss = 0.0
        total_nmse = 0.0

        for field, t in dataloader:
            optimizer.zero_grad()
            field = field.float().to(device)
            reconstructed = model(field)
            loss = criterion(reconstructed, field)
            total_loss += loss.item()


            loss.backward()
            optimizer.step()
            num_batches += 1

        average_loss = total_loss / num_batches
        num_batches = 0
        
        #validation
        for field, t in valloader:
            field = field.float().to(device)
            reconstructed = model(field)
            with torch.no_grad():
                nmse = compute_nmse(reconstructed, field)
                total_nmse += nmse

            num_batches += 1
        
        average_nmse = total_nmse / num_batches

        current_time = time.time()
        time_epoch = (current_time - start_time)/ 60
        print(f"Epoch: {epoch+1}, Loss:{average_loss : .4f}, nmse = {average_nmse},  time = {round(time_epoch, 4)} min")
        info.append((epoch, loss.item(), average_nmse, time_epoch))
        output.append((epoch, field, reconstructed))

        if (current_time - start_time) / 60 > training_time and time_reached == False:
            print(f"We just reached 30 min !")
            epoch_30 = epoch
            time_reached = True
            break

        end_time = time.time()
        execution_time = (end_time - start_time) / 60


    return model, info, execution_time