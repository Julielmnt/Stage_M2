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
import argparse
from training import compute_nmse
from autoencoder import SimuDataset
from fcnn import fcnn
from training import test

def plot_NMSE(K_values, nmse, nmse_train, nmse_test, title = "FCNN", i = 0, j = None, fontsize = 18, NMSE_pca = None, K_values_pca = None):

    if j is None:
        j = len(K_values)
    fig, ax2 = plt.subplots(figsize = (10, 7)) 
    # ax2.scatter(K_values[i:j], nmse[i:j], s = 40, c = 'orchid', label = r"NMSE for the whole data")
    # ax2.plot(K_values[i:j], nmse[i:j], c = 'orchid', linestyle='-')
    ax2.scatter(K_values[i:j], nmse_train[i:j], s = 40, c = 'palegreen', label = r"NMSE for the trained set")
    ax2.plot(K_values[i:j], nmse_train[i:j], c = 'palegreen', linestyle='-')
    ax2.scatter(K_values[i:j], nmse_test[i:j], s = 40, c = 'salmon', label = r"NMSE for the test set")
    ax2.plot(K_values[i:j], nmse_test[i:j], c = 'salmon', linestyle='-')

    ax2.scatter(K_values[i:j], NMSE_pca[i:i + len(K_values[i:j])], s = 40, c = 'orchid', label = r"NMSE for PCA")
    ax2.plot(K_values[i:j], NMSE_pca[i:i + len(K_values[i:j])], c = 'orchid', linestyle='-')
    # ax.set_title(r"Residual norm")
    ax2.set_ylabel(r"NMSE", fontsize = fontsize)
    ax2.set_xlabel(r'K', fontsize = fontsize)
    ax2.set_ylim(bottom = 0)
    # ax2.set_yscale('log') 
    # ax2.set_xscale('log') 
    ax2.legend(fontsize = fontsize)
    ax2.set_ylim(bottom = 0)
    fig.suptitle(title, fontsize = fontsize)
    # fig.suptitle(f'number of modes = {num_modes} ', fontsize = fontsize)
    fig.tight_layout()


def main(
    batch_size=4,
    lr=1e-4,
    K=128,
    Lambda=1e-2,
    training_ratio=0.9,
    num_epochs=100,
    depth = 4,
    directory = "root", 
    regular_save = False
):

    device = get_freer_gpu() if torch.cuda.is_available() else "cpu"

    # Dataloading
    current_directory = compatible_path("../")

    root = f"{current_directory}/results/fcnn/new_architecture/"

    print(f"saving NN at {directory}, root being : {root}")

    if directory == "root":
        directory = f"{current_directory}/results/fcnn/new_architecture/"

    if directory == "num_epochs":
        directory = f"{current_directory}/results/fcnn/new_architecture/num_epochs/"


    filename = f"model_K_{K}_bs{batch_size}_lr{lr}"

    simulation = Simulation(current_directory, normalize=True, Lambda=Lambda)
    time_array, x, z, u, w, T, umean, wmean, Tmean = map(
        lambda x: torch.tensor(x).to(device), simulation.import_data(mean_type="Znorm")
    )
    # simulation.image_rgb()




    h, l = np.shape(x)
    N = h * l * 3 
    print("ABOUT DATA")
    print(f"Lambda = {Lambda}")
    print(f"number of snapshots : {simulation.m}")

    # SVD
    X = torch.from_numpy(simulation.X[:, : int(training_ratio * simulation.m)]).to(
        device
    )
    Q = 1000
    U, S, V = torch.pca_lowrank(X, q=Q)
    print("SVD")
    print(f"Q = {Q}")
    # print(U.shape, S.shape, V.shape)
    U = U.to(torch.float32)
    V = V.to(torch.float32)
    S = S.to(torch.float32)

    # Setting  Dataset
    batch_size = batch_size
    training_ratio = training_ratio

    dataset = SimuDataset(simulation, device, rgb=False, training_ratio=training_ratio)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    test_set = SimuDataset(
        simulation, device, rgb=False, mode="test", training_ratio=training_ratio
    )
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


    # Model


    print("MODEL PARAMETERS")
    print(f"K = {K}")
    print(f"batch_size = {batch_size}")
    print(f"lr = {lr}")
    print(f"depth = {depth}")

    model = fcnn(device=device, N=Q, U=U, S=S, K=K, depth = depth)
    model.to(device)

    K_values = [1, 2, 3, 5, 10, 25, 50, 128, 256, 500, 700, 800, 900, 1000]
    K_values = [1, 2, 3, 5, 10, 25, 50, 128, 256, 500, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]

    nmse_test = []
    nmse_train = []

    if not regular_save :
        for i, K in enumerate(K_values):
            title = f'model_bs{batch_size}_K{K}_lr{lr}_info.txt'
            model = fcnn(device = device, U=U, K = K, N = Q, S = S, depth = depth)
            model.load_state_dict(torch.load(directory + f'model_bs{batch_size}_K{K}_lr{lr}.pt'))
            print(f"K = {K}")
            nmse_test.append(test(model, test_loader, device=device, metric = 'nmse'))
            nmse_train.append(test(model, train_loader, device=device, metric = 'nmse'))
        

        print(nmse_test, nmse_train)
    
    i = 0
j = len(K_values)
plot_NMSE(K_values, nmse, nmse_train, nmse_test, title = "FCNN", i = i, j = j, NMSE_pca = NMSE_pca)