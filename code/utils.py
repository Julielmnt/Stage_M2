#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Functions from DeepInv"
# cff-version: 0.0.1
# authors:
# - family-names: "Tachella"
#   given-names: "Julian"
#   orcid: "https://orcid.org/0000-0003-3878-9142"
# - family-names: "Chen"
#   given-names: "Dongdong"
#   orcid: "https://orcid.org/0000-0002-7016-9288"
# - family-names: "Hurault"
#   given-names: "Samuel"
#   orcid: "https://orcid.org/0000-0002-5163-2791"
# - family-names: "Terris"
#   given-names: "Matthieu"
#   orcid: "https://orcid.org/0009-0009-9726-6131"
# title: "DeepInverse: A deep learning framework for inverse problems in imaging"
# version: latest
# doi: 10.5281/zenodo.7982256
# date-released: 2023-06-30
# url: "https://github.com/deepinv/deepinv"



import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from natsort import natsorted
import imageio

def get_freer_gpu():
    """
    Function from DeepInv library
    Returns the GPU device with the most free memory.

    """
    try:
        if os.name == "posix":
            os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp")
            memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
        else:
            os.system('bash -c "nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp"')
            memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
        idx = np.argmax(memory_available)
        device = torch.device(f"cuda:{idx}")
        print(f"Selected GPU {idx} with {np.max(memory_available)} MB free memory ")
    except:
        device = torch.device(f"cuda")
        print("Couldn't find free GPU")

    try:
        os.remove("tmp")
    except FileNotFoundError:
        pass

    return device

def make_gif(image_directory, gif_path, fps):

    image_files = natsorted([os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith('.png')])
    with imageio.get_writer(gif_path, mode='I', fps = fps) as writer:
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)

    print(f"GIF created at: {gif_path}")


def plot_field(simu, u, w, T,  t, save=False, directory=None):
    fig, ax = plt.subplots(figsize=(15, 5))
    vmin = T.min()
    vmax = T.max()
    abs_max = max(abs(vmin), abs(vmax))
    ax.streamplot(
        simu.x.T,
        simu.z.T,
        u.T,
        w.T,
        color="k",
        arrowsize=0.7,
        linewidth=1,
    )
    levels = np.linspace(vmin, vmax, 20)
    cf0 = ax.contourf(
        simu.x,
        simu.z,
        T,
        levels=levels,
        cmap=cm.Spectral.reversed(),
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
    )
    cbar = plt.colorbar(cf0, ax=ax, shrink=0.35, aspect=6, ticks= [vmin, 0, vmax])
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



def info_text(directory, batch_size, sizes, info, version = 1, K = None, **training_params):
    from datetime import datetime
    
    if K == None : 
        title = f"model_bs{batch_size}_v{version}_info.txt"
    else :
        title = f"model_bs{batch_size}_K{K}_info.txt"

    with open(directory + title, 'w') as file:
        file.write(f"Information about Convolutionnal Autoencoder with batch size {batch_size}\n\n")
        current_date = datetime.now()
        file.write(f"date : {current_date}\n")
        if K is not None : 
            file.write(f"K = {K}\n")
        file.write(f"batch_size = {batch_size}\n")
        for i in range(len(sizes)):
            file.write(f"size{i+1} = {sizes[i]}\n")
            

        for key, value in training_params.items():
            file.write(f"{key} = {value}\n")
            
        for i in range(len(info)):
            file.write(f"epoch {info[i][0] + 1}, loss {info[i][1]}\n")
    print("info saved !")


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
                epoch, loss = line.split(',')
                epoch = int(epoch.split()[1])
                loss = float(loss.split()[1])
                variables[f'epoch_{epoch}'] = loss

    for key, value in variables.items():
        print(f"{key}: {value}")
    return variables, sizes

def losses_from_info(variables):
    epochs = []
    losses = []

    for key, value in variables.items():
        if key.startswith('epoch_'):
            epoch_number = int(key.split('_')[1])
            epochs.append(epoch_number)
            losses.append(value)
    return epochs, losses

def compute_residual_norm(X, reconstructed_data):
    residual = X - reconstructed_data
    residual_norm = np.linalg.norm(residual, 'fro')
    return residual_norm / np.linalg.norm(X, 'fro')