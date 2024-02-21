#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import modred as mr
import os
import matplotlib

matplotlib.use("TkAgg")  # Specify the backend
import matplotlib.pyplot as plt
from matplotlib import cm


class Simulation:
    def __init__(self, current_directory, Ra=10e8, Gamma=8, Lambda=10e-2):
        self.Ra = Ra
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.current_directory = current_directory
        self.ticks = [-0.04, 0, 0.05, 0.10, 0.15]

    def import_data(self):
        bulk = np.load(f"{self.current_directory}/data/bulk.npz")
        time = bulk["time"]
        x = bulk["x"]
        z = bulk["z"]
        u = bulk["u"]
        w = bulk["w"]
        T = bulk["T"]
        umean = np.mean(u, axis=0)
        wmean = np.mean(w, axis=0)

        self.time = time
        self.x = x
        self.z = z
        self.u = u
        self.w = w
        self.T = T
        self.umean = umean
        self.wmean = wmean
        self.m = len(time)
        self.h, self.l = np.shape(x)

        return self.time, self.x, self.z, self.u, self.w, self.T, self.umean, self.wmean
    
    def plot_field(self, t, save = False, directory = None):
        fig, ax = plt.subplots(figsize = (15,5))
        vmin=self.T.min() 
        vmax=self.T.max()
        ax.streamplot(self.x.T, self.z.T, self.u[t,:,:].T , self.w[t,:,:].T, color = 'k', arrowsize = 0.7,linewidth = 1)
        levels = np.linspace(vmin, vmax, 20)
        cf0 = ax.contourf(self.x, self.z, self.T[t, :, :], levels=levels, cmap=cm.Spectral.reversed(), norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(cf0, ax=ax, shrink=0.35, aspect = 6, ticks = self.ticks)
        cbar.ax.set_aspect('auto') 
        ax.set_title(f'Temperature and velocity field at t = {t}')
        ax.set_aspect('equal')
        ax.set_ylim(0,1)
        ax.set_xlim(-4,4)

        if save :
            plt.savefig(directory,dpi=300)
            plt.close()
        else : 
            plt.tight_layout()
            plt.show()

    def save_clip(self, t_start, t_end, directory):
        for t in range(t_start, t_end+1) :
            self.plot_T_field(t, save = True, directory = directory + f'snapshot_{t}')


    def plot_meanfield(self, map = 'umean', save = False, directory = None):

        fig, ax = plt.subplots(figsize = (15,5))
        ax.streamplot(self.x.T, self.z.T, self.umean.T ,self.wmean.T, color = 'k', arrowsize = 0.7,linewidth = 1, density = 3)
        if map == 'umean' :
            cf0 = ax.contourf(self.x, self.z, self.umean, levels=20, cmap=cm.Spectral)
        if map == 'wmean':
            cf0 = ax.contourf(self.x, self.z, self.wmean, levels=20, cmap=cm.Spectral)
        cbar = plt.colorbar(cf0, ax=ax, shrink=0.35, aspect = 6)
        cbar.ax.set_aspect('auto') 
        # ax.set_title(f'Quiver plot at t = {t}')
        ax.set_aspect('equal')
        ax.set_ylim(0,1)
        ax.set_xlim(-4,4)

        if save :
            plt.savefig(directory,dpi=300)
            plt.close()
        else : 
            plt.tight_layout()
            plt.show()

    def plot_meancomponent(self, component = 'u'):
        fig, ax = plt.subplots(figsize=(10,8))
        if component == 'u':
            cf0 = ax.contourf(self.x, self.z, self.umean, levels=20, cmap=cm.Spectral)
        if component == 'w':
            cf0 = ax.contourf(self.x, self.z, self.wmean, levels=20, cmap=cm.Spectral)
        plt.colorbar(cf0, aspect = 5, shrink = 0.13)
        ax.set_aspect('equal', 'box')

def plot_results(umean, x, z, POD_modes, KE_mode, num_modes=10):

    fig, ax = plt.subplots(figsize=(10, 1.5 * num_modes), nrows=num_modes + 1)
    cf0 = ax[0].contourf(x, z, umean, levels=20, cmap=cm.nipy_spectral)
    ax[0].set_title("Umean")
    plt.colorbar(cf0)
    for j in range(num_modes):
        cf0 = ax[j + 1].contourf(
            x, z, POD_modes[j + 10], levels=20, cmap=cm.nipy_spectral
        )
        plt.colorbar(cf0)
    [ax[j].set_aspect("equal", "box") for j in range(num_modes + 1)]
    [
        ax[j + 1].set_title(f"Mode {j+10}, KE = {np.round(KE_mode[j+10], 2)}%")
        for j in range(num_modes)
    ]
    plt.tight_layout()
    # plt.show()
    plt.savefig("pod_modes_lasts.png", dpi=300)
    # plt.close()


if __name__ == "__main__":

    current_directory = "/home/julielimonet/Documents/Stage_M2/"
    simulation = Simulation(current_directory)
    time, x, z, u, w, T, umean, wmean = simulation.import_data()
