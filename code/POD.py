#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import modred as mr
import os
import matplotlib
matplotlib.use('TkAgg')  # Specify the backend
import matplotlib.pyplot as plt
from matplotlib import cm

current_directory = os.getcwd()
# print(f"\n current directory : {current_directory}\n")

# --> Load 3D (t,x,y) data in python arrays format
bulk = np.load(f"{current_directory}/data/bulk.npz")
time = bulk["time"]
x = bulk["x"]
z = bulk["z"]
u = bulk["u"]
w = bulk["w"]
T = bulk["T"]
umean = np.mean(u, axis=0)
u = u - umean

h, l = np.shape(x)
m = len(time)


# reshape
U = np.reshape(u, (m, h * l))


# print(np.shape(x))
# print(np.shape(x), np.shape(time))
# U = np.reshape(u, (6249, 81*51))
# print(np.shape(U))
# print(np.array_equal(np.reshape(U, (6249,81,51)), u))


# Compute POD
num_modes = 20
POD = mr.compute_POD_arrays_snaps_method(np.swapaxes(U,0,1), list(mr.range(num_modes)))
modes = POD.modes
eigvals = POD.eigvals

print(np.shape(modes))
print(np.shape(eigvals))
print(h*l)
KE_mode = []
for i in range(20):
    TKE = np.sum(eigvals)
    KE_mode.append(eigvals[i]*100/TKE)

POD_modes = []
[POD_modes.append([]) for i in range(num_modes)]
for i in range(num_modes):
  POD_modes[i] = modes[:,i]
  POD_modes[i] = POD_modes[i].reshape((h,l)) 




# --> plot results
num_modes = 10
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


num_modes = 10
fig, ax = plt.subplots(figsize=(10,1.5*num_modes),nrows=num_modes+1)
cf0 = ax[0].contourf(x, z, umean, levels=20, cmap=cm.nipy_spectral)
ax[0].set_title('Umean')
plt.colorbar(cf0)
for j in range(num_modes):
  cf0 = ax[j+1].contourf(x, z, POD_modes[j+10], levels=20, cmap=cm.nipy_spectral)
  plt.colorbar(cf0)
[ax[j].set_aspect('equal', 'box') for j in range(num_modes+1)]
[ax[j+1].set_title(f'Mode {j}, KE = {np.round(KE_mode[j], 2)}%') for j in range(num_modes)]
plt.tight_layout()
# plt.show()
plt.savefig('pod_modes_firsts.png',dpi=300)
# plt.close()







# j=0
# fig, ax = plt.subplots(figsize=(10,15))
# cf0 = ax.contourf(x, z, POD_modes[j], levels=20, cmap=cm.nipy_spectral)
# plt.colorbar(cf0)
# ax.set_aspect('equal', 'box')
# ax.set_title(f'Mode {j}, KE = {KE_mode[j]}')
# plt.tight_layout()
# plt.show()