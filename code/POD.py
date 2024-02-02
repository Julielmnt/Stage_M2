#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import modred as mr
import os
import matplotlib
matplotlib.use('TkAgg')  # Specify the backend
import matplotlib.pyplot as plt
from matplotlib import cm


def import_data(current_directory):
  bulk = np.load(f"{current_directory}/data/bulk.npz")
  time = bulk["time"]
  x = bulk["x"]
  z = bulk["z"]
  u = bulk["u"]
  w = bulk["w"]
  T = bulk["T"]
  umean = np.mean(u, axis=0)
  u = u - umean

  return time, x, z, u, w, T, umean


def POD(U, h, l, num_modes= 100):
  POD = mr.compute_POD_arrays_snaps_method(np.swapaxes(U,0,1), list(mr.range(num_modes)))
  modes = POD.modes
  eigvals = POD.eigvals
  proj_coef = POD.proj_coeffs

  # print(np.shape(modes))
  # print(np.shape(eigvals))

  KE_mode = []
  for i in range(20):
      TKE = np.sum(eigvals)
      KE_mode.append(eigvals[i]*100/TKE)

  POD_modes = []
  [POD_modes.append([]) for i in range(num_modes)]
  for i in range(num_modes):
    POD_modes[i] = modes[:,i]
    POD_modes[i] = POD_modes[i].reshape((h,l)) 

  return POD_modes , KE_mode, modes, eigvals, proj_coef


def plot_results(umean, x, z, POD_modes, KE_mode, num_modes = 10):

  fig, ax = plt.subplots(figsize=(10,1.5*num_modes),nrows=num_modes+1)
  cf0 = ax[0].contourf(x, z, umean, levels=20, cmap=cm.nipy_spectral)
  ax[0].set_title('Umean')
  plt.colorbar(cf0)
  for j in range(num_modes):
    cf0 = ax[j+1].contourf(x, z, POD_modes[j+10], levels=20, cmap=cm.nipy_spectral)
    plt.colorbar(cf0)
  [ax[j].set_aspect('equal', 'box') for j in range(num_modes+1)]
  [ax[j+1].set_title(f'Mode {j+10}, KE = {np.round(KE_mode[j+10], 2)}%') for j in range(num_modes)]
  plt.tight_layout()
  # plt.show()
  plt.savefig('pod_modes_lasts.png',dpi=300)
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
    print(h*l)