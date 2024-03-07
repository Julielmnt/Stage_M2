#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import modred as mr
import os
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import cm


def POD(U, h, l, num_modes=100):

    POD = mr.compute_POD_arrays_snaps_method(
        np.swapaxes(U, 0, 1), mode_indices= np.arange(0,num_modes,1))
    
    modes = POD.modes
    eigvals = POD.eigvals
    proj_coef = POD.proj_coeffs
    eigvecs = POD.eigvecs

    return modes, eigvals, eigvecs, proj_coef


def KE_modes(eigvals, num_modes = 100):
    KE_modes = []
    TKE = np.sum(eigvals)
    for i in range(num_modes):
        KE_modes.append(eigvals[i] / TKE)

    return KE_modes

def KE_modes_two_components(eigvals_u, eigvals_w, num_modes = 100):
    KE_modes = []
    TKE = np.sum(eigvals_u) + np.sum(eigvals_w)
    for i in range(num_modes):
        KE_modes.append((eigvals_u[i] + eigvals_w[i]) / TKE)

    return KE_modes

def reconstruct_data(proj_coef_u, modes_u, num_modes):
    reconstructed_data = proj_coef_u[:num_modes, :].T @ modes_u[:,:num_modes].T

    return reconstructed_data


def residual_norm(X, reconstructed_data):
    residual = X - reconstructed_data
    residual_norm = np.linalg.norm(residual, 'fro')
    return residual_norm / np.linalg.norm(X, 'fro')

def residuals(proj_coef_u, modes_u, num_modes, U):
    residuals = np.zeros(num_modes)
    for i in range(1, num_modes + 1 ):
        reconstructed_data = reconstruct_data(proj_coef_u, modes_u, i)
        r = residual_norm(U, reconstructed_data)
        residuals[i-1] = r
    return residuals


def plot_residual(residuals, num_modes, fontsize = 18):

    fig,ax = plt.subplots(figsize = (10, 8))

    N_modes = np.arange(1, num_modes + 1, step = 1)
    
    ax.scatter(N_modes, residuals, s = 30, c = 'orange')

    # ax.set_title(r"Residual norm")
    ax.set_ylabel('Residual norm', fontsize = fontsize)
    ax.set_xlabel('Number of modes considered', fontsize = fontsize)
    ax.legend()

def plot_energy_contribution(KE_modes_all, j, fontsize) :

    N = np.arange(0, len(KE_modes_all[:j]), step = 1)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(N, KE_modes_all[:j], c = 'teal')
    ax.set_xticks(N)
    ax.axhline(0.01, label = r"0.1\% line", c = 'k')
    # ax.set_title(r'Energy contribution for each mode')
    ax.legend(fontsize=fontsize-3)
    ax.tick_params(axis='both', labelsize=fontsize-4)
    ax.set_xticks(np.arange(min(N), max(N)+1, 5))
    ax.set_xlabel(r"modes", fontsize = fontsize)
    ax.set_ylabel(r"\% of total energy", fontsize = fontsize)


def plot_map_mode(modes, KE_modes, n_mode, x, z):

    fig, ax = plt.subplots(figsize=(10,8))
    cf0 = ax.contourf(x, z, modes[:,:,n_mode], levels=20, cmap=cm.Spectral)
    plt.colorbar(cf0, aspect = 5, shrink = 0.13)
    ax.set_title(f'Mode {n_mode}, KE = {np.round(KE_modes[n_mode]*100, 2)}\%')
    ax.set_aspect('equal', 'box')
    
def streamplot_mode(modes_u, modes_w, cmapmodes, KE_modes, n_mode, x, z):

    fig, ax = plt.subplots(figsize = (15,5))

    ax.streamplot(x.T, z.T, modes_u[:,:,n_mode].T ,modes_w[:,:,n_mode].T, color = 'k', arrowsize = 0.7,linewidth = 1, density = 3)
    cf0 = ax.contourf(x, z, cmapmodes[:,:,n_mode], levels=20, cmap=cm.Spectral, norm=matplotlib.colors.Normalize(vmin=cmapmodes[:,:,n_mode].min(), vmax=cmapmodes[:,:,n_mode].max()))
    cbar = plt.colorbar(cf0, ax=ax, shrink=0.35, aspect = 6)
    cbar.ax.set_aspect('auto') 
    # ax.set_title(f'Quiver plot at t = {t}')
    ax.set_aspect('equal')
    ax.set_ylim(0,1)
    ax.set_xlim(-4,4)
    ax.set_title(f'Mode {n_mode}, KE = {np.round(KE_modes[n_mode]*100, 2)}\%')


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


# num_modes = 10
# fig, ax = plt.subplots(figsize=(10,1.5*num_modes),nrows=num_modes+1)
# cf0 = ax[0].contourf(x, z, umean, levels=20, cmap=cm.nipy_spectral)
# ax[0].set_title('Umean')
# plt.colorbar(cf0)
# for j in range(num_modes):
#   cf0 = ax[j+1].contourf(x, z, POD_modes[j+10], levels=20, cmap=cm.nipy_spectral)
#   plt.colorbar(cf0)
# [ax[j].set_aspect('equal', 'box') for j in range(num_modes+1)]
# [ax[j+1].set_title(f'Mode {j}, KE = {np.round(KE_mode[j], 2)}%') for j in range(num_modes)]
# plt.tight_layout()
# # plt.show()
# plt.savefig('pod_modes_firsts.png',dpi=300)
# # plt.close()

# N = np.arange(0, len(KE_mode), step = 1)
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(N, KE_mode, )

# ax.set_aspect('equal', 'box')
# ax.set_title(f'Mode {j}, KE = {KE_mode[j]}')
# plt.tight_layout()
# plt.show()


if __name__ == "__main__":
    current_directory = os.getcwd()

    time, x, z, u, w, T, umean = import_data(current_directory)

    h, l = np.shape(x)
    m = len(time)

    # reshape
    U = np.reshape(u, (m, h * l))
    print(h * l)
