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
  wmean = np.mean(w, axis=0)

  return time, x, z, u, w, T, umean, wmean

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



if __name__ == "__main__":
    pass
    # current_directory = os.getcwd()

    # time, x, z, u, w, T, umean, wmean = import_data(current_directory)
    # mask = (z > 0.85) & (z < 0.95)

    # # Use the mask to get the indices where the condition is satisfied
    # indices = np.where(mask)
    # # print(indices)
    # print(np.shape(u))