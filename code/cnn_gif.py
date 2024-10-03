
import numpy as np
import torch
import modred as mr
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import os
import importlib

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

from utils import get_freer_gpu

device = get_freer_gpu() if torch.cuda.is_available() else "cpu"
print(device)
lr = 1e-4
nc = 8
depth = 4
dynamics = None
pandey_data = False
Ra = 1e8
Lambda = 1e-2



def directory_to_chose_cnn(Ra, pandey_data, dynamics=None, convnext=False, convnext_kernel=7):
    directory = f'{current_directory}/results/cnn/pandeylike/'


    if convnext:
        directory += "convnext/"

    if Ra == 1e8 and not pandey_data and dynamics==None:
        directory += "Ra1e8/"

    if Ra == 1e7 and not pandey_data:
        directory += "Ra1e7/"

    if Ra == 2e6 and not pandey_data:
        directory += "Ra2e6/"   

    if pandey_data:
        directory += "pandey_data/"

    if convnext_kernel != 7:
        directory += f"kernel{convnext_kernel}/"

    
    if dynamics is not None:
        if dynamics == "HC":
            directory += "/HCRa1e8"
        if dynamics == "pureRB":
            directory += "pureRB"
    return directory


if pandey_data:
    simulation = Simulation(current_directory, normalize=True, Lambda=Lambda, Ra=Ra, pandey=pandey_data)
    simulation.import_data_pandey(mean_type="scaled")

else:
    simulation = Simulation(current_directory, normalize=True, Lambda=Lambda, Ra=Ra)
    time_array, x, z, u, w, T, umean, wmean, Tmean = map(
        lambda x: torch.tensor(x).to(device), simulation.import_data(mean_type="scaled")
    )
    simulation.image_rgb()
training_ratio = 0.9
X = torch.from_numpy(np.swapaxes(simulation.X[: int(training_ratio * simulation.m)], 0, 1)).to(device)

from autoencoder import SimuDataset
import torch
from torch.utils.data import DataLoader
from fcnn import fcnn
from cnn import CNN_pandeylike


def apply_cnn(directory, simulation, convnext=True, kernel_size=5, batch_size=4, title=None, pandey_data=False, lr=1e-4, nc=8, depth=4, training_ratio=0.9):

    test_set = SimuDataset(simulation, device, rgb=True, mode="test", training_ratio=training_ratio)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    size1 = (50, 30)
    size2 = (25, 15)
    size3 = (13, 8)
    size4 = (7, 4)
    size5 = (4, 2)
    size6 = (2, 1)
    sizes = [size1, size2, size3, size4, size5, size6]

    model = CNN_pandeylike(device=device, n_channels=8, sizes=sizes, convnext=convnext, kernel_size=kernel_size, pandey_data=pandey_data, h=simulation.h, l=simulation.l)
    model.to(device)

    model.load_state_dict(torch.load(directory + f'model_bs{batch_size}_nc{nc}_lr{lr}.pt'))
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        # encoded_data_batch = []
        decoded_data_batch = []

    for x, t in test_loader:
        x = x.to(device).float()
        # encoded_data = model.encoder(x)
        # encoded_data = model.linear_encoder(torch.reshape(encoded_data,  (encoded_data.shape[0],-1)))

        # decoded_data = model.linear_decoder(encoded_data)
        # decoded_data = model.decoder(torch.reshape(decoded_data, (decoded_data.shape[0], model.n_channels, model.sizes[-1][0], model.sizes[-1][1])))
        decoded_data = model.forward(x)
        # encoded_data_batch.append(encoded_data)
        decoded_data_batch.append(decoded_data)
        print("HERE")
    # encoded_data_batch = torch.cat(encoded_data_batch, dim=0)
    decoded_data_batch = torch.cat(decoded_data_batch, dim=0)

    return decoded_data_batch

directory = directory_to_chose_cnn(Ra, pandey_data, dynamics, convnext=True)
decoded_data = apply_cnn(directory, simulation, convnext=True, kernel_size=7, batch_size=4, title=None, pandey_data=False, lr=1e-4, nc=64, depth=4, training_ratio=0.9)

import matplotlib.gridspec as gridspec
def plot_field(simulation, X_reconstructed, t, t_min=0, t_max=6249, normalization=None, save=False, directory=None):
    fig = plt.figure(figsize=(18, 6))  # Increase figsize for better aspect ratio
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.02], hspace = 0.2, wspace=0.02)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    cax = fig.add_subplot(gs[:, 1])

    u, w, T = simulation.uwt_from_X()
    u = np.reshape(u, (simulation.m, simulation.h, simulation.l))
    w = np.reshape(w, (simulation.m, simulation.h, simulation.l))
    T = np.reshape(T, (simulation.m, simulation.h, simulation.l))

    U_reconstructed = X_reconstructed[:, :simulation.h * simulation.l]
    W_reconstructed = X_reconstructed[:, simulation.h * simulation.l:2 * simulation.h * simulation.l]
    T_reconstructed = X_reconstructed[:, 2 * simulation.h * simulation.l:]

    U_reconstructed = np.reshape(U_reconstructed, (simulation.m, simulation.h, simulation.l))
    W_reconstructed = np.reshape(W_reconstructed, (simulation.m, simulation.h, simulation.l))
    T_reconstructed = np.reshape(T_reconstructed, (simulation.m, simulation.h, simulation.l))

    vmin = T[t_min:t_max, :, :].min()
    vmax = T[t_min:t_max, :, :].max()
    vmin_reconstructed = T_reconstructed[t_min:t_max, :, :].min()
    vmax_reconstructed = T_reconstructed[t_min:t_max, :, :].max()

    abs_max = max(abs(vmin), abs(vmax))
    abs_max_reconstructed = max(abs(vmin_reconstructed), abs(vmax_reconstructed))
    maximum_value = max(abs_max, abs_max_reconstructed)

    nmse_u = u - U_reconstructed
    nmse_w = w - W_reconstructed
    nmse_T = T - T_reconstructed

    ax1.streamplot(
        simulation.x.T,
        simulation.z.T,
        u[t, :, :].T,
        w[t, :, :].T,
        color="k",
        arrowsize=0.7,
        linewidth=1,
        density=1.5
    )
    ax2.streamplot(
        simulation.x.T,
        simulation.z.T,
        U_reconstructed[t, :, :].T,
        W_reconstructed[t, :, :].T,
        color="k",
        arrowsize=0.7,
        linewidth=1,
        density=1.5
    )
    ax3.streamplot(
        simulation.x.T,
        simulation.z.T,
        nmse_u[t, :, :].T,
        nmse_w[t, :, :].T,
        color="k",
        arrowsize=0.7,
        linewidth=1,
        density=1.5
    )

    levels = np.linspace(vmin, vmax, 20)
    cf0 = ax1.contourf(
        simulation.x,
        simulation.z,
        T[t, :, :],
        levels=levels,
        cmap=plt.cm.Spectral.reversed(),
        norm=matplotlib.colors.Normalize(vmin=-maximum_value, vmax=maximum_value),
    )
    cf1 = ax2.contourf(
        simulation.x,
        simulation.z,
        T_reconstructed[t, :, :],
        levels=levels,
        cmap=plt.cm.Spectral.reversed(),
        norm=matplotlib.colors.Normalize(vmin=-maximum_value, vmax=maximum_value),
    )
    cf2 = ax3.contourf(
        simulation.x,
        simulation.z,
        nmse_T[t, :, :],
        cmap=plt.cm.Spectral.reversed(),
        norm=matplotlib.colors.Normalize(vmin=-maximum_value, vmax=maximum_value)
    )

    # Create a single colorbar for the entire figure
    cbar = fig.colorbar(cf0, cax=cax, aspect=6, ticks=[-maximum_value, 0, maximum_value], orientation = 'vertical', pad= 0.1)
    cbar.ax.set_aspect('auto')

    ax1.set_aspect("equal")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-4, 4)
    ax2.set_aspect("equal")
    ax2.set_ylim(0, 1)
    ax2.set_xlim(-4, 4)
    ax3.set_aspect("equal")
    ax3.set_ylim(0, 1)
    ax3.set_xlim(-4, 4)
    ax1.tick_params(axis='both', which='both', bottom=False, top=False,labelbottom=False, labelleft=False)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False,labelbottom=False, labelleft=False)
    ax3.tick_params(axis='both', which='both', bottom=False, top=False,labelbottom=False, labelleft=False)
    ax1.set_title(r"ground truth", fontsize = 20)
    ax2.set_title(r"reconstruction", fontsize = 20)
    ax3.set_title(r"difference", fontsize = 20)
    # plt.tight_layout()

    if save:
        plt.savefig(directory, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

t=1369
plot_field(simulation, decoded_data.cpu().detach().numpy(), t, save=False, directory=f"{current_directory}/presentation/fcnn/snapshots/snapshot_{t}")

# t = 5958
# for t in range(5940, 5991):
#     plot_field(simulation, decoded_data.cpu().detach().numpy(), t, save=True, directory=f"{current_directory}/presentation/fcnn/snapshots/snapshot_{t}")
