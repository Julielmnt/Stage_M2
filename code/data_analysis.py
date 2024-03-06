#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import os
import matplotlib
import imageio
import os
from natsort import natsorted
# matplotlib.use("TkAgg")  # Specify the backend
import matplotlib.pyplot as plt
from matplotlib import cm


class Simulation:
    """Simulation of a Rayleigh-Benard and Horizontal convection competition flow"""

    def __init__(
        self,
        current_directory,
        Ra=1e8,
        Gamma=8,
        Lambda=1e-2,
        ticks=[-0.04, 0, 0.05, 0.10, 0.15],
        normalize = False
    ):
        self.Ra = Ra
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.current_directory = current_directory
        self.ticks = ticks
        self.normalize = normalize

    def import_data(self):
        """Returns data from bulk.npz in a data folder in the current directory.

        Returns:
            arrays of time, x, z, u, w, T, umean, wmean and Tmean
        """
        bulk = np.load(compatible_path(f"{self.current_directory}/data/bulk.npz"))
        time = bulk["time"]
        x = bulk["x"]
        z = bulk["z"]
        u = bulk["u"]
        w = bulk["w"]
        T = bulk["T"]
        umean = np.mean(u, axis=0)
        wmean = np.mean(w, axis=0)
        Tmean = np.mean(T, axis=0)

        self.time = time
        self.x = x
        self.z = z
        self.u = u
        self.w = w
        self.T = T
        self.umean = umean
        self.wmean = wmean
        self.Tmean = Tmean
        self.m = len(time)
        self.h, self.l = np.shape(x)



        W = np.reshape(self.w - self.wmean, (self.m, self.h * self.l))
        U = np.reshape(self.u - self.umean, (self.m, self.h * self.l))
        T = np.reshape(self.T - self.Tmean, (self.m, self.h * self.l))


        if self.normalize:
            U = U/np.max(np.abs(U))
            W = W/np.max(np.abs(W))
            T = T/np.max(np.abs(T))

        self.X = np.swapaxes(np.concatenate([U, W, T], axis = 1), 0, 1)



        return self.time, self.x, self.z, self.u, self.w, self.T, self.umean, self.wmean, self.Tmean

    def image_rgb(self):
        h = self.h
        l = self.l
        image_rgb = np.zeros((3, np.shape(self.u)[0], np.shape(self.u)[1], np.shape(self.u)[2]))
        image_rgb[0, :, :, :] = (self.u-self.umean)/np.max(np.abs(self.u))
        image_rgb[1, :, :, :] = (self.w-self.wmean)/np.max(np.abs(self.w))
        image_rgb[2, :, :, :] = (self.T-self.Tmean)/np.max(np.abs(self.T))
        self.X_rgb = image_rgb
    
    def reconstruct_simulation(self, U_reconstructed, W_reconstructed, T_reconstructed):
        time, x, z, u, w, T, umean, wmean, Tmean = self.import_data()
        self.u = U_reconstructed
        self.w = W_reconstructed
        self.T = T_reconstructed
    
    def UZ(self):
        self.uz = np.mean(np.mean(self.u[:,:,25:51], axis = 2), axis = 1)
        self.uzmean = np.mean(self.uz)
        return self.uz, self.uzmean

    def KE(self):
        self.ke = 0.5 * (self.u**2 + self.w**2)
        return self.ke

    def divergence(self):
        self.div_x = np.gradient(self.u, self.x[:,0], axis = 1)
        self.div_z = np.gradient(self.w, self.z[0,:], axis = 2)
        self.div = self.div_x + self.div_z

        return self.div

    def plot_field(self, t, save=False, directory=None):
        fig, ax = plt.subplots(figsize=(15, 5))
        vmin = self.T[t, :, :].min()
        vmax = self.T[t, :, :].max()
        abs_max = max(abs(vmin), abs(vmax))
        ax.streamplot(
            self.x.T,
            self.z.T,
            self.u[t, :, :].T,
            self.w[t, :, :].T,
            color="k",
            arrowsize=0.7,
            linewidth=1,
        )
        levels = np.linspace(vmin, vmax, 20)
        cf0 = ax.contourf(
            self.x,
            self.z,
            self.T[t, :, :],
            levels=levels,
            cmap=cm.Spectral.reversed(),
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        )
        cbar = plt.colorbar(cf0, ax=ax, shrink=0.35, aspect=6, ticks= [-abs_max, 0, abs_max])
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

    def save_clip(self, t_start, t_end, directory):
        for t in range(t_start, t_end + 1):
            self.plot_field(t, save=True, directory=directory + f"snapshot_{t}")

    def plot_meanfield(self, map="umean", save=False, directory=None):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.streamplot(
            self.x.T,
            self.z.T,
            self.umean.T,
            self.wmean.T,
            color="k",
            arrowsize=0.7,
            linewidth=1,
            density=3,
        )
        if map == "umean":
            cf0 = ax.contourf(self.x, self.z, self.umean, levels=20, cmap=cm.Spectral)
        if map == "wmean":
            cf0 = ax.contourf(self.x, self.z, self.wmean, levels=20, cmap=cm.Spectral)
        cbar = plt.colorbar(cf0, ax=ax, shrink=0.35, aspect=6)
        cbar.ax.set_aspect("auto")
        # ax.set_title(f'Quiver plot at t = {t}')
        ax.set_aspect("equal")
        ax.set_ylim(0, 1)
        ax.set_xlim(-4, 4)

        if save:
            plt.savefig(directory, dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def plot_meancomponent(self, component="u"):
        fig, ax = plt.subplots(figsize=(10, 8))
        if component == "u":
            cf0 = ax.contourf(self.x, self.z, self.umean, levels=20, cmap=cm.Spectral)
        if component == "w":
            cf0 = ax.contourf(self.x, self.z, self.wmean, levels=20, cmap=cm.Spectral)
        plt.colorbar(cf0, aspect=5, shrink=0.13)
        ax.set_aspect("equal", "box")

def make_gif(image_directory, gif_path, fps):

    image_files = natsorted([os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith('.png')])
    with imageio.get_writer(gif_path, mode='I', fps = fps) as writer:
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)

    print(f"GIF created at: {gif_path}")


def plot_uz(time, uz_reconstructed, uz, uzmean, num_modes, nmax, residuals_uw, fontsize):
    fig, [ax1,ax2] = plt.subplots(1,2,figsize = (16, 5), gridspec_kw={'width_ratios': [2, 1]})


    ax1.scatter(time, uz, s = 4, c = 'orange', label = r'ground truth $u_z$')
    ax1.scatter(time, uz_reconstructed + uzmean, s = 10, c = 'cadetblue', label = r'reconstructed $u_z$')
    ax1.set_ylabel(r'$u_z$', fontsize= fontsize)
    ax1.set_xlabel(r'$t$', fontsize = fontsize)
    ax1.legend(fontsize = fontsize, loc = 'upper left')


    N_modes = np.arange(1, nmax + 1, step = 1)
    ax2.scatter(N_modes[:num_modes], residuals_uw[:num_modes], s = 30, c = 'orchid', label = r"residual norm for the reconstruction of stacked u and w")

    # ax.set_title(r"Residual norm")
    ax2.set_ylabel('Residual norm', fontsize = fontsize)
    ax2.set_xlabel('Number of modes considered', fontsize = fontsize)
    ax2.set_ylim(bottom = 0)
    ax2.legend(loc= "upper right")
    fig.suptitle(f'number of modes = {num_modes} ', fontsize = fontsize)
    fig.tight_layout()


def compatible_path(current_directory, verbose = False):
    if sys.platform.startswith('win'):
        if verbose :
            print("Running on Windows")
        current_directory = os.path.normpath(current_directory)
    elif sys.platform.startswith('linux'):
        if verbose :
            print("Running on Linux")
    return current_directory



if __name__ == "__main__":

    current_directory = "/home/julielimonet/Documents/Stage_M2/"
    simulation = Simulation(current_directory)
    time, x, z, u, w, T, umean, wmean = simulation.import_data()
