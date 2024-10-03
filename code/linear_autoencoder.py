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

import wandb
import data_analysis
from data_analysis import Simulation
from data_analysis import compatible_path

import utils
from utils import get_freer_gpu
from utils import info_text
import argparse
from autoencoder import SimuDataset
from autoencoder import training
    
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
        self.N = N
        self.K = K

        self.encoder = nn.Sequential(nn.Linear(self.N, self.N), nn.Linear(self.N, self.K)).to(device)

        self.decoder = nn.Linear(self.K, self.N).to(device)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




def main(
    batch_size=4,
    lr=1e-4,
    K=128,
    training_mode="num_epoch",
    Lambda=1e-2,
    training_ratio=0.9,
    num_epochs=100,
    directory = "root", 
    group = "run",
    wandb_on = True
):


    device = get_freer_gpu() if torch.cuda.is_available() else "cpu"
    min_30_reached = False
    epoch_30 = None

    # Dataloading
    current_directory = compatible_path("../")

    root = f"{current_directory}/results/fcnn/simplelinear/"

    print(f"saving NN at {directory}, root being : {root}")
    directory = root
    filename = f"model_K{K}_bs{batch_size}_lr{lr}"

    simulation = Simulation(current_directory, normalize=True, Lambda=Lambda)
    time_array, x, z, u, w, T, umean, wmean, Tmean = map(
        lambda x: torch.tensor(x).to(device), simulation.import_data(mean_type="Znorm")
    )


    h, l = np.shape(x)
    N = h * l * 3 
    print("ABOUT DATA")
    print(f"Lambda = {Lambda}")
    print(f"number of snapshots : {simulation.m}")


    # Setting  Dataset
    batch_size = batch_size
    training_ratio = training_ratio

    dataset = SimuDataset(simulation, device, rgb=False, training_ratio=training_ratio)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    valset = SimuDataset(
        simulation, device, rgb=False, mode="val", training_ratio=training_ratio
    )
    valloader = DataLoader(dataset=valset, batch_size=4, shuffle=True)

    # Model

    weight_decay = 0
    print("MODEL PARAMETERS")
    print(f"K = {K}")
    print(f"batch_size = {batch_size}")
    print(f"lr = {lr}")

    model = SimpleLinearAutoencoder(device=device, N=N, K=K)
    # model = fcnn_debug(device=device, N=Q, U=U, K=K)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(num_epochs * 0.8))



    # wandb
    if wandb_on:
        name = filename
        
        wandb.init(
            project = "SimpleLinear",

            config = {
                "Lambda" : Lambda,
                "K" : K,
                "batch_size " : batch_size,
                "lr" : lr,
                "training_mode" : training_mode,
                "training_ratio" : training_ratio,
                "directory" : directory,
                },

                name=name,

                group = group
        )
        # wandb.run.summary["run_color"] = color(K) 


    # Training
    print(f"training mode : {training_mode}")
    num_epoch = num_epochs
    if training_mode == "num_epoch":
        model, info, execution_time = training(
            dataloader,
            valloader,
            model,
            criterion,
            optimizer,
            device,
            scheduler=scheduler,
            num_epoch=num_epoch,
            noisy=False,
            wandb=wandb,
            directory = directory, 
            filename = filename,
        )

    # Saving
    # print(f"batchsize = {batch_size}")
    
    torch.save(model.state_dict(), directory + filename + ".pt")
    print("model saved !")
    info_text(
        directory,
        batch_size,
        info,
        title=filename + "_info.txt",
        Lambda = Lambda,
        K = K,
        thirty_min_epoch=epoch_30,
        num_epoch=num_epoch,
        lr=lr,
        weight_decay=weight_decay,
        execution_time_in_min=execution_time
    )
    if wandb_on:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script with parameter bs and n_channels"
    )
    parser.add_argument(
        "--bs", type=int, default=4, help="Value for batchsize (default: 4)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Value for learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--K", type=int, default=128, help="Value for learning rate (default: 128)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of num_epochs (default: 100)",)

    parser.add_argument(
        "--depth", type=int, default=4, help="Depth of MLP (default: 4)",)
    
    parser.add_argument(
        "--directory", type=str, default="root", help="Where to save the NN (default: root)",)
    
    parser.add_argument(
        "--group", type=str, default="run", help="group in which to save the model in wandb (default: run)",)
    
    parser.add_argument(
        "--wandb_on", type=bool, default=True, help="wandbing (default: True)")


    args = parser.parse_args()
    bs = args.bs
    lr = args.lr
    K = args.K
    num_epochs = args.num_epochs
    directory = args.directory
    group = args.group   
    wandb_on = args.wandb_on   

    main(batch_size=bs, lr=lr, K=K, num_epochs=num_epochs, directory = directory, group = group, wandb_on=wandb_on)  

