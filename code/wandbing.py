import numpy as np
import torch
import modred as mr
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import os
import POD
import importlib
importlib.reload(POD)

import data_analysis
importlib.reload(data_analysis)

import autoencoder_analysis
importlib.reload(autoencoder_analysis)

import fcnn
importlib.reload(fcnn)

import utils
importlib.reload(utils)

import training
importlib.reload(training)


matplotlib.pyplot.rc('text', usetex=True)

fontsize = 18

from data_analysis import Simulation
current_directory = '../'

batch_size = 4
lr = 1e-4
nc = 8
depth = 4

IDs = ['efjqhw8n', 'bbnjh0ac', '4ton03ro', 'xv5z8jqr', 'jh7opf7c', 'hyoahse8', 'ovz7kyo7', 'aocqkftg', 'mzck83zd', 'vu9cujil', 'l51e56gw', 'jrhln40e', '8anc6dr1']
K_values = [2, 3, 5, 10, 25, 50, 128, 256, 500, 700, 800, 900, 1000]

directory = f'{current_directory}/results/fcnn/pandey_data/'

if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")


for i, K in enumerate(K_values):

    # title = f'model_bs{batch_size}_nc{nc}_lr{lr}'
    title = f'model_K{K}_bs{batch_size}_lr{lr}'
    from autoencoder_analysis import get_variables_from_info_text
    variables, sizes = get_variables_from_info_text(directory, title + '_info.txt')
    from utils import losses_from_info

    loss = losses_from_info(variables)

    if len(loss) == 4:
        epochs, losses, nmse, times = loss

    else :
        epochs, losses, nmse, times, average_loss, nmse_train = loss

    import wandb
    wandb.init(
                project = "FCNN",
                id = IDs[i],
                resume = "allow"
            )
    for epoch in range(100):
        wandb.log({"loss": losses[epoch], "nmse": nmse[epoch], "nmse_train" : nmse_train[epoch]})

    wandb.finish()