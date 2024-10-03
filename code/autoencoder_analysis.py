#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from autoencoder import ConvolutionalAutoencoder_debug
import torch
from training import compute_nmse
from data_analysis import compatible_path
import matplotlib.pyplot as plt
import numpy as np

current_directory = compatible_path('../')

def get_variables_from_info_text(directory, title):
    variables = {}
    sizes = []
    with open(directory + title, 'r') as file:
        for line in file:
            if line.startswith('size'):
                size = eval(line.split('=')[1].strip())
                sizes.append(size)
            if '=' in line:
                key, value = line.split('=')
                key = key.strip()
                value = eval(value.strip())
                variables[key] = value
            elif line.startswith('epoch'):
                if '=' not in line:
                    if len(line.split(',')) == 4 :
                        epoch, loss, nmse, times = line.split(',')
                        epoch = int(epoch.split()[1])
                        loss = float(loss.split()[1])
                        nmse = float(nmse.split()[1])
                        times = float(times.split()[1])
                        variables[f'epoch_{epoch}'] = {'loss': loss, 'nmse': nmse, 'time': times}

                    
                    if len(line.split(',')) == 6 :
                        epoch, loss, nmse, times, average_loss, nmse_train = line.split(',')
                        epoch = int(epoch.split()[1])
                        loss = float(loss.split()[1])
                        nmse = float(nmse.split()[1])
                        times = float(times.split()[1])
                        average_loss = float(average_loss.split()[1])
                        nmse_train = float(nmse_train.split()[1])
                        variables[f'epoch_{epoch}'] = {'loss': loss, 'nmse': nmse, 'time': times, 'average_loss' : average_loss, 'nmse_train' : nmse_train}

    for key, value in variables.items():
        print(f"{key}: {value}")

    return variables, sizes


def compute_nmse_on_dataset(test_loader, K, batch_size, device, nc = 64, lr = 1e-4, directory = f'{current_directory}/results/autoencoder/cnn/'):
   
    title = f'model_bs{batch_size}_K{K}_lr{lr}_nc{nc}_info.txt'
    variables, sizes = get_variables_from_info_text(directory, title)
    autoencoder = ConvolutionalAutoencoder_debug(device, sizes, K = K, n_channels = nc, bias = False)

    autoencoder.load_state_dict(torch.load(directory + f'model_bs{batch_size}_K{K}_lr{lr}_nc{nc}.pt'))
    autoencoder.eval()  # Set the model to evaluation mode

    
    with torch.no_grad():
        total_nmse = 0
        num_batches = 0

    for x, t in test_loader:
        
        x = x.to(device).float()
        x_noisy = x + torch.randn_like(x)*.1
        encoded_data = autoencoder.encoder(x_noisy)
        encoded_data = autoencoder.linear_encoder(torch.reshape(encoded_data,  (encoded_data.shape[0],-1)))

        decoded_data = autoencoder.linear_decoder(encoded_data)
        decoded_data = autoencoder.decoder(torch.reshape(decoded_data, (decoded_data.shape[0], autoencoder.n_channels, autoencoder.sizes[-1][0], autoencoder.sizes[-1][1])))
        
        nmse = compute_nmse(decoded_data, x)
        total_nmse += nmse
        num_batches += 1

    average_nmse = total_nmse / num_batches

    return average_nmse



def plot_info_training(directory, title, i = 0, fontsize = 18, return_sizes = False):
    """Plot the graph of loss and NMSE of trained model

    Arguments:
        variables -- _description_
        K -- _description_
        batch_size -- _description_
        lr -- _description_

    Keyword Arguments:
        fontsize -- _description_ (default: {18})
    """
    variables, sizes = get_variables_from_info_text(directory, title + '_info.txt')
    from utils import losses_from_info

    loss = losses_from_info(variables)

    if len(loss) == 4:
        epochs, losses, nmse, times = loss
    
    else :
        epochs, losses, nmse, times, average_loss, nmse_train = loss


    fig, ax1 = plt.subplots(figsize = (15, 4))
    ax1.plot(range(len(epochs[i:])), losses[i:], marker = ".", markersize = 10,  c = 'cadetblue', label = r"loss")
    ax1.plot(range(len(epochs[i:])), np.array(average_loss[i:]) * 4, marker = ".", markersize = 10, c = 'orange', label = 'averaged')

    # ax1.plot(range(len(epochs[i:])), nmse_train[i:],marker = '.',  markersize = 10, c = 'orchid', label = 'nmse train')
    # ax1.scatter(range(len(epochs[i:])), losses_cnn[i:], s = 20, c = 'orange', label = r"cnn alone")
    ax1.set_ylabel(r'$loss$', fontsize= fontsize)
    ax1.set_xlabel(r'$epoch$', fontsize = fontsize)
    ax1.set_ylim(bottom = 0)
    ax1.legend(fontsize = fontsize, loc = 'upper right')
    fig.tight_layout()

    fig, ax1 = plt.subplots(figsize = (15, 4))
    ax1.plot(range(len(epochs[i:])), nmse[i:], markersize = 10, marker = '.', c = 'orchid', label = 'nmse test')
    ax1.plot(range(len(epochs[i:])), nmse_train[i:],marker = '.',  markersize = 10, c = 'orange', label = 'nmse train')
    # ax1.scatter(range(len(epochs[i:])), nmse_cnn[i:], s = 20, c = 'orange', label = fr"cnn alone, bs = {batch_size}, lr = {lr}")
    ax1.set_ylabel(r'$nmse$', fontsize= fontsize)
    ax1.set_xlabel(r'$epoch$', fontsize = fontsize)
    ax1.legend(fontsize = fontsize, loc = 'upper right')
    ax1.set_ylim(bottom = 0)



    fig.tight_layout()
    if return_sizes:
        return sizes
if __name__ == "__main__":
    pass