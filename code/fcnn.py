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
from utils import info_text, color
import argparse
from autoencoder import SimuDataset
from autoencoder import training, training_time
from POD import POD, KE_modes


class fcnn_debug(nn.Module):
    """fcnn"""

    def __init__(self, N, U, S, device, K=20):
        """Initiates fcnn class

        Arguments:
            N -- Size of Input data
            N -- Size of Input data

        Keyword Arguments:
            K -- Size of latent space (default: {20})
            device -- where to run code (default: {device})
        """
        super().__init__()
        self.N = N
        self.K = K
        self.U = U
        self.S = S
        self.mid = int((self.N + self.K) / 2)
        # self.mid = K
        # print(f'mid : {self.mid}')

        self.encoder = nn.Sequential(
            nn.Linear(self.N, self.K),
            #  nn.BatchNorm1d(self.mid),
            #  nn.ReLU(),
            nn.Linear(self.K, self.K),
            #  nn.BatchNorm1d(self.K),
            #  nn.ReLU(),
            #  nn.Linear(self.K, self.K),
            #  nn.BatchNorm1d(self.K),
            nn.ReLU(),
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(self.K, self.N),
            #  nn.BatchNorm1d(self.mid),
            #  nn.ReLU(),
            #  nn.Linear(self.mid, self.N),
            #  nn.BatchNorm1d(self.N),
            #  nn.ReLU(),
        ).to(device)

    def forward(self, x):
        projected_data = torch.matmul(torch.conj(self.U).t(), x.t()).t()
        normalized = projected_data * 1 / (torch.sqrt(self.S))
        print(normalized.device)
        encoded = self.encoder(normalized)

        decoded = self.decoder(encoded) * (torch.sqrt(self.S))
        decoded = torch.matmul(self.U, decoded.t()).t()
        return decoded


class resblock(nn.Module):
    def __init__(self, channels, device, batchnorm=False, bias=True, skip=True):
        super().__init__()
        self.channels = channels
        self.device = device
        self.batchnorm = batchnorm
        self.skip = skip
        self.bias = bias
        self.Linear = nn.Linear(self.channels, self.channels, bias=bias).to(self.device)

    def forward(self, x):
        out = self.Linear(x)
        out = nn.functional.relu(out).to(self.device)
        if self.skip:
            out = out + x

        return out.to(self.device)


class fcnn(nn.Module):
    """fcnn"""

    def __init__(self, N, U, S, device, depth=4, K=128, pca_only = False):
        super().__init__()
        self.N = N
        self.K = K
        self.U = U
        self.S = S
        self.device = device
        self.depth = depth
        self.mid = int((self.N + self.K) / 2)
        self.pca_only = pca_only

        self.MLPin = nn.Sequential()
        for l in range(self.depth):
            self.MLPin.append(resblock(self.N, device = device))

        self.encoder = nn.Sequential(
            nn.Linear(self.N, self.N),
            nn.ReLU(),
            nn.Linear(self.N, self.mid),
            nn.ReLU(),
            nn.Linear(self.mid, self.K),
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(self.K, self.K),
            nn.ReLU(),
            nn.Linear(self.K, self.mid),
            nn.ReLU(),
            nn.Linear(self.mid, self.N),
        ).to(device)

        self.MLPout = nn.Sequential()
        for l in range(depth):
            self.MLPout.append(resblock(self.N, device=self.device))

    def forward(self, x):
        projected_data = torch.matmul(self.U.t(), x.t()).t()
        normalized = projected_data * 1 / (torch.sqrt(self.S))

        if not self.pca_only:
            encoded = self.encoder(self.MLPin(normalized))

            decoded = self.MLPout(self.decoder(encoded)) * (torch.sqrt(self.S))
            decoded = torch.matmul(self.U, decoded.t()).t()

        else : 
            normalized = normalized* (torch.sqrt(self.S))
            decoded = torch.matmul(self.U, normalized.t()).t()
            decoded = torch.tensor(decoded, requires_grad=True)
        return decoded


def main(
    batch_size=4,
    lr=1e-4,
    K=128,
    training_mode="num_epoch",
    Lambda=1e-2,
    Ra=1e8,
    training_ratio=0.9,
    num_epochs=100,
    depth = 4,
    saving_directory = "root", 
    regular_save = False,
    group = "run",
    pca_only = False,
    wandb_on = False,
    Q = 1000,
    only_velocities = False,
    pandey_data = False,    
    dynamics = None,
):
    """Main exec function

    Keyword Arguments:
        n_channels -- number of channels (default: {64})
        batch_size -- size of the batch (default: {4})
    """


    device = get_freer_gpu() if torch.cuda.is_available() else "cpu"
    min_30_reached = False
    epoch_30 = None

    # Dataloading
    current_directory = compatible_path("../")

    # root = f"{current_directory}/results/fcnn/new_architecture/"

    # print(f"saving NN at {directory}, root being : {root}")

    # if directory == "root":
    #     directory = f"{current_directory}/results/fcnn/new_architecture/pca_only/"

    # if directory == "num_epochs":
    #     directory = f"{current_directory}/results/fcnn/new_architecture/num_epochs/"

    current_directory = compatible_path("../")

    directory = f"{current_directory}/results/fcnn/"


    if Ra == 1e8 and not pandey_data:
        directory += "Ra1e8/"

    if Ra == 1e7 and not pandey_data:
        directory += "Ra1e7/"

    if Ra == 2e6 and not pandey_data:
        directory += "Ra2e6/"   

    if pandey_data:
        directory += "pandey_data/"

    if dynamics is not None:
        if dynamics == "HC":
            directory = f"{current_directory}/results/fcnn/HCRa1e8"
        if dynamics == "pureRB":
            directory = f"{current_directory}/results/fcnn/pureRB"
        group = dynamics

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")


    print(f"saving NN at {directory}")
    filename = f"model_K{K}_bs{batch_size}_lr{lr}"

    if pandey_data:
        simulation = Simulation(current_directory, normalize=True, Lambda=Lambda, Ra=Ra, pandey=pandey_data)
        simulation.import_data_pandey(mean_type="scaled")

    else:
        simulation = Simulation(current_directory, normalize=True, Lambda=Lambda, Ra=Ra)
        time_array, x, z, u, w, T, umean, wmean, Tmean = map(
            lambda x: torch.tensor(x).to(device), simulation.import_data(mean_type="Znorm")
        )
        # simulation.image_rgb()




    h, l = simulation.h, simulation.l
    N = h * l * 3 
    if pandey_data:
        N = h * l
    print("ABOUT DATA")
    print(f"Lambda = {Lambda}")
    print(f"Ra = {Ra}")
    print(f"number of snapshots : {simulation.m}")

    # SVD
    X = torch.from_numpy(np.swapaxes(simulation.X[: int(training_ratio * simulation.m)], 0, 1)).to(
        device
    )

    if pca_only:
        Q = K


    U, S, V = torch.pca_lowrank(X, q=Q)
    print("SVD")
    print(f"Q = {Q}")
    # print(U.shape, S.shape, V.shape)
    U = U.to(torch.float32)
    V = V.to(torch.float32)
    S = S.to(torch.float32)
    print(f"U shape : {U.shape}")
    # Setting  Dataset
    batch_size = batch_size
    training_ratio = training_ratio

    dataset = SimuDataset(simulation, device, rgb=False, training_ratio=training_ratio, only_velocities=only_velocities)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    valset = SimuDataset(
        simulation, device, rgb=False, mode="val", training_ratio=training_ratio, only_velocities=only_velocities,
    )
    valloader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=True)

    # Model

    weight_decay = 0
    print("MODEL PARAMETERS")
    print(f"K = {K}")
    print(f"batch_size = {bs}")
    print(f"lr = {lr}")
    print(f"depth = {depth}")

    model = fcnn(device=device, N=Q, U=U, S=S, K=K, depth=depth, pca_only=pca_only)
    # model = fcnn_debug(device=device, N=Q, U=U, K=K)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(num_epochs * 0.8))



    # wandb
    if wandb_on:
        name = filename
        if pca_only:
            name = filename + "_pca_only"   
        wandb.init(
            project = "FCNN",

            config = {
                "Lambda" : Lambda,
                "Ra" : Ra,
                "K" : K,
                "depth" : depth,
                "batch_size " : batch_size,
                "lr" : lr,
                "training_mode" : training_mode,
                "training_ratio" : training_ratio,
                "directory" : directory,
                "Ra_power": simulation.Ra_power,
                "pandey_data" : pandey_data,
                "Q" : Q,
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
            wandb_on=wandb_on,
            regular_save=regular_save,
            directory = directory, 
            filename = filename,
        )

    if training_mode == "time":
        model, info, execution_time = training_time(
            dataloader,
            valloader,
            model,
            criterion,
            optimizer,
            device,
            training_time=20,
            time_reached=False,
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
        Lambda=Lambda,
        Ra=Ra,
        K=K,
        thirty_min_epoch=epoch_30,
        num_epoch=num_epoch,
        depth=depth,
        lr=lr,
        weight_decay=weight_decay,
        execution_time_in_min=execution_time,
        Q=Q,
        pandey_data=pandey_data,
        wandb_on=wandb_on,
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
        "--saving_directory", type=str, default="root", help="Where to save the NN (default: root)",)
    
    parser.add_argument(
        "--regular_save", action="store_true", help="Whether or not to save regularly the model (default: False)",)
    
    parser.add_argument(
        "--group", type=str, default="run", help="group in which to save the model in wandb (default: run)",)
    
    parser.add_argument(
        "--pca_only", action="store_true", help="To compute only pca (default: False)")
    
    parser.add_argument(
        "--wandb_on", action="store_true", help="wandbing (default: False)")
    
    parser.add_argument(
        "--Ra", type=float, default=1e8, help="Value for Ra (default: 1e8)")
    
    parser.add_argument(
        "--Lambda", type=float, default=1e-2, help="Value for Lambda (default: 1e-2)")
    
    parser.add_argument(
        "--Q", type=int, default=1000, help="Size for svd (default: 1000)")
    
    parser.add_argument(
        "--pandey_data", action="store_true", help="Value for pandey data (default: False)")
    
    parser.add_argument(
        "--dynamics", type=str, default=None, help="Value for dynamics (default: None)")
    
    

    args = parser.parse_args()
    bs = args.bs
    lr = args.lr
    K = args.K
    num_epochs = args.num_epochs
    depth = args.depth
    saving_directory = args.saving_directory
    regular_save = args.regular_save
    group = args.group   
    pca_only = args.pca_only   
    wandb_on = args.wandb_on   
    Ra = args.Ra   
    Lambda = args.Lambda   
    Q = args.Q   
    pandey_data = args.pandey_data   
    dynamics = args.dynamics   

    main(Ra=Ra, Lambda=Lambda, Q=Q, batch_size=bs, lr=lr, K=K, num_epochs=num_epochs, depth = depth, saving_directory = saving_directory, regular_save = regular_save, group = group, pca_only=pca_only, wandb_on=wandb_on, pandey_data=pandey_data, dynamics=dynamics)  

