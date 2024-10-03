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
from math import log
dtype = np.float64


class Simulation:
    """Simulation of a Rayleigh-Benard and Horizontal convection competition flow"""

    def __init__(
        self,
        current_directory,
        Ra=1e8,
        Gamma=8,
        Lambda=1e-2, 
        ticks=[-0.04, 0, 0.05, 0.10, 0.15],
        normalize=True,
        pandey=False,

    ):
        self.Ra = Ra
        self.Ra_power = int(log(Ra, 10))
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.current_directory = current_directory
        self.ticks = ticks
        self.normalize = normalize
        self.Umax = None
        self.pandey=pandey

    def import_data(self, mean_type = "Znorm"):
        """Returns data from bulk.npz in a data folder in the current directory.

        Returns:
            arrays of time, x, z, u, w, T, umean, wmean and Tmean
        """
        if self.Ra_power == 8:
            path = f"{self.current_directory}/data/bulk_L{self.Lambda}_Ra{self.Ra_power}.npz"

        if self.Ra_power == 7:  
            path = f"{self.current_directory}/data/bulk_L{self.Lambda}_Ra{self.Ra_power}.npz"

        if self.Ra == 2e6:
            path = f"{self.current_directory}/data/bulk_Ra2e6.npz"


        bulk = np.load(compatible_path(path))
        print(f"data from {path}")
        if 'time' in bulk:
            time = bulk['time']
        elif 't' in bulk:
            time = bulk['t']
        else:
            raise KeyError("Neither 'time' nor 't' found in the .npz file")
        x = bulk["x"]
        z = bulk["z"]
        u = bulk["u"]
        w = bulk["w"]
        T = bulk["T"]
        umean = np.mean(u, axis=0, dtype = dtype)
        wmean = np.mean(w, axis=0, dtype = dtype)
        Tmean = np.mean(T, axis=0, dtype = dtype)

        self.time = time
        self.x = x
        self.z = z
        self.u = u
        self.w = w
        self.T = T
        self.umean = umean
        self.wmean = wmean
        self.Tmean = Tmean

        self.uspacial_mean = np.mean(u, axis = (1, 2))
        self.wspacial_mean = np.mean(w, axis = (1, 2))
        self.Tspacial_mean = np.mean(T, axis = (1, 2))
        self.m = len(time)

        if np.ndim(x) == 2:
            self.h, self.l = np.shape(x)
        else:
            self.h, self.l = np.shape(u[0,:,:])

        if mean_type == 'None':
            pass

        if mean_type == 'temporal':
            self.u = self.u - umean
            self.w = self.w - wmean
            self.T = self.T - Tmean
        
        if mean_type == 'total':
            self.utotalmean = np.mean(u)
            self.wtotalmean = np.mean(w)
            self.Ttotalmean = np.mean(T)
            self.u = self.u - np.mean(u)
            self.w = self.w - np.mean(w)
            self.T = self.T - np.mean(T)

        if mean_type == 'Znorm':
            self.u = (self.u-umean)/np.std(self.u)
            self.w = (self.w-wmean)/np.std(self.w)
            self.T = (self.T-Tmean)/np.std(self.T)

        if mean_type == 'scaled':
            self.u = self.scaled_data(self.u)
            self.w = self.scaled_data(self.w)
            self.T = self.scaled_data(self.T)

        self.X = self.X_from_uwt()[0]
        print(f"X shape {self.X.shape}")
        self.velocities = self.X_from_uwt()[1]
    

        return self.time, self.x, self.z, self.u, self.w, self.T, self.umean, self.wmean, self.Tmean
    
    
    
    def import_data_pandey(self, mean_type = "None"):
        uytheta = np.load(f"{self.current_directory}/data/Pr07_uytheta_8001_320_60_1by8.npy")
        print("mean uytheta", np.mean(uytheta))
        uytheta = uytheta - np.mean(uytheta, axis=0)
        if mean_type == 'Znorm':
            uytheta = (uytheta - np.mean(uytheta))/np.std(uytheta)

        if mean_type == 'scaled':
            uytheta = self.scaled_data(uytheta)

        if mean_type == 'None':
            pass

        self.m = uytheta.shape[0]
        self.uytheta = uytheta
        self.X = np.reshape(uytheta, (self.m, uytheta.shape[1]*uytheta.shape[2]))
        self.X_rgb = np.expand_dims(uytheta, axis = 1)
        self.h = uytheta.shape[1]
        self.l = uytheta.shape[2]



    
    def X_from_uwt(self):
        W = np.reshape(self.w, (self.m, self.h * self.l))
        U = np.reshape(self.u, (self.m, self.h * self.l))
        Temp = np.reshape(self.T, (self.m, self.h * self.l))
    
        
        if self.Umax == None :
            self.Umax = np.max(np.abs(U))
            self.Wmax = np.max(np.abs(W))
            self.Tmax = np.max(np.abs(Temp))

        if self.normalize:
            U = U/self.Umax
            W = W/self.Wmax
            Temp = Temp/self.Tmax

        self.X = np.concatenate([U, W, Temp], axis = 1, dtype = dtype)
        self.velocities = np.concatenate([U, W], axis = 1, dtype = dtype)

        return self.X, self.velocities
    
    def uwt_from_X(self):
        h = self.h
        l = self.l
        
        U = self.X[:, :h*l]
        W = self.X[:, h*l:2*h*l]
        T = self.X[:, 2*h*l:]

        return U, W, T
    
    def import_partial_data(self):

        path = f"{self.current_directory}/data/bulk_L{self.Lambda}.npz"
        bulk = np.load(compatible_path(path))
        time = bulk["time"]
        x = bulk["x"]
        z = bulk["z"]

        self.time = time
        self.x = x
        self.z = z
        self.m = len(time)
        self.h, self.l = np.shape(x)

        return self.time, self.x, self.z

    def image_rgb(self):
        image_rgb = np.zeros(( np.shape(self.u)[0],3, np.shape(self.u)[1], np.shape(self.u)[2]))
        
        image_rgb[:, 0, :, :] = np.reshape(self.X[:, : self.h * self.l], (self.m, self.h, self.l))
        image_rgb[:, 1, :, :] = np.reshape(self.X[:, self.h * self.l : 2 * self.h * self.l], (self.m, self.h, self.l))
        image_rgb[:, 2, :, :] = np.reshape(self.X[:, 2 * self.h * self.l :], (self.m, self.h, self.l))
        self.X_rgb = image_rgb
    

    def reconstruct_simulation(self, X_reconstructed, rgb = False, normalize = None):

        if not rgb : 
            self.m = np.shape(X_reconstructed)[0]
            time, x, z = self.import_partial_data()
            self.X = np.swapaxes(X_reconstructed, 0,1)

            self.u = np.reshape(X_reconstructed[:, :self.h*self.l], (self.m, self.h, self.l))
            self.w = np.reshape(X_reconstructed[:, self.h*self.l:2*self.h*self.l], (self.m, self.h,self.l))
            self.T = np.reshape(X_reconstructed[:, 2*self.h*self.l:], (self.m, self.h, self.l))

            if normalize is not None:
                self.Umax = normalize[0]
                self.Wmax = normalize[1]
                self.Tmax = normalize[2]
                self.u = self.u * self.Umax
                self.w = self.w * self.Wmax
                self.T = self.T * self.Tmax
        if rgb : 
            self.m = np.shape(X_reconstructed)[0]
            time, x, z = self.import_partial_data()
            self.X_rgb = np.swapaxes(X_reconstructed, 0, 1)
            self.u = X_reconstructed[0, :, :, :]
            self.w = X_reconstructed[1, :, :, :]
            self.T = X_reconstructed[2, :, :, :]

            if normalize is not None:
                self.Umax = normalize[0]
                self.Wmax = normalize[1]
                self.Tmax = normalize[2]
                self.u = X_reconstructed[0, :, :, :] * self.Umax
                self.w = X_reconstructed[1, :, :, :] * self.Wmax
                self.T = X_reconstructed[2, :, :, :] * self.Tmax

    def sampleHC(self):
        if self.Lambda == 1e-2:
            HC_indices = [405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4440, 4441, 4442, 4443, 4444, 4445, 4446, 4447, 4448, 4449, 4450, 4559, 4560, 4561, 4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569, 4699, 4700, 4701, 4702, 4703, 4704, 4705, 4706, 4707, 4708, 4709, 4852, 4853, 4854, 4855, 4856, 4857, 4858, 4859, 4860, 4861, 4862, 5062, 5063, 5064, 5065, 5066, 5067, 5068, 5069, 5070, 5071, 5072, 5153, 5154, 5155, 5156, 5157, 5158, 5159, 5160, 5161, 5162, 5163, 5378, 5379, 5380, 5381, 5382, 5383, 5384, 5385, 5386, 5387, 5388, 5538, 5539, 5540, 5541, 5542, 5543, 5544, 5545, 5546, 5547, 5548, 5954, 5955, 5956, 5957, 5958, 5959, 5960, 5961, 5962, 5963, 5964]
            return HC_indices
        
    def sampleRB(self):
        if self.Lambda == 1e-2:
            pass

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

    def plot_field(self, t, normalization = None, save=False, directory=None):
        fig, ax = plt.subplots(figsize=(15, 5))

        u,w,T = self.uwt_from_X()
        u = np.reshape(u, (self.m, self.h, self.l))
        w = np.reshape(w, (self.m, self.h, self.l))
        T = np.reshape(T, (self.m, self.h, self.l))
        # print(u.max(), w.max(), T.max())
        # print(u.shape)
        vmin = T[:, :, :].min()
        vmax = T[:, :, :].max()

        abs_max = max(abs(vmin), abs(vmax))

        ax.streamplot(
            self.x.T,
            self.z.T,
            u[t, :, :].T,
            w[t, :, :].T,
            color="k",
            arrowsize=0.7,
            linewidth=1,
            density = 1.5
        )
        levels = np.linspace(vmin, vmax, 20)
        cf0 = ax.contourf(
            self.x,
            self.z,
            T[t, :, :],
            levels=levels,
            cmap=cm.Spectral.reversed(),
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        )
        cbar = plt.colorbar(cf0, ax=ax, shrink=0.35, aspect=6, ticks= [vmin, 0, vmax])
        cbar.ax.set_aspect("auto")
        # ax.set_title(f"Temperature and velocity field at t = {t}")
        ax.set_aspect("equal")
        ax.set_ylim(0, 1)
        ax.set_xlim(-4, 4)
        # plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.tight_layout()

    
        if save:
            plt.savefig(directory, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def save_clip(self, t_start, t_end, directory, step = 1):
        for t in range(t_start, t_end + 1, step):
            self.plot_field(t, save=True, directory=directory + f"snapshot_{t}")

    def plot_meanfield(self, map="umean", save=False, directory=None):
        fig, ax = plt.subplots(figsize=(15, 5))

        umean = np.mean(self.u, axis=0)
        wmean = np.mean(self.w, axis=0)

        if map == "umean":
            abs_max = np.max(abs(umean))
            levels = np.linspace(-abs_max, abs_max, 50)
            cf0 = ax.contourf(self.x, self.z, umean, cmap=cm.Spectral, levels=levels)
        if map == "wmean":
            abs_max = np.max(abs(wmean))
            levels = np.linspace(-abs_max, abs_max, 50)
            cf0 = ax.contourf(self.x, self.z, wmean, cmap=cm.Spectral, levels=levels)
        if map == "whole mean":
            abs_max = np.max(abs((wmean + umean)/2))
            levels = np.linspace(-abs_max, abs_max, 50)
            cf0 = ax.contourf(self.x, self.z, (wmean + umean)/2, cmap=cm.Spectral, levels=levels)
            
        
        
        ax.streamplot(
            self.x.T,
            self.z.T,
            umean.T,
            wmean.T,
            color="k",
            arrowsize=0.7,
            linewidth=1,
            density=2,
        )
        
        cbar = plt.colorbar(cf0, ax=ax, shrink=0.35, aspect=6, ticks= [-abs_max, 0, abs_max])
        ticks_fontsize = 12
        cbar.ax.tick_params(labelsize=ticks_fontsize)
        cbar.ax.set_aspect("auto")
        # ax.set_title(f'Quiver plot at t = {t}')
        ax.set_aspect("equal")
        ax.set_ylim(0, 1)
        ax.set_xlim(-4, 4)
        ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)

        if save:
            plt.savefig(directory, dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def plot_meancomponent(self, component="u"):

        umean = np.mean(self.u, axis=0)
        wmean = np.mean(self.w, axis=0)
        fig, ax = plt.subplots(figsize=(10, 8))
        if component == "u":
            cf0 = ax.contourf(self.x, self.z, umean, levels=20, cmap=cm.Spectral)
        if component == "w":
            cf0 = ax.contourf(self.x, self.z, wmean, levels=20, cmap=cm.Spectral)
        if component == "whole":
            cf0 = ax.contourf(self.x, self.z,  (wmean + umean)/2, levels=20, cmap=cm.Spectral)
        plt.colorbar(cf0, aspect=5, shrink=0.13)
        ax.set_aspect("equal", "box")

    @staticmethod
    def scaled_data(x):
        x_znormed = (x - np.mean(x))/np.std(x)
        # x_znormed = x
        xmin = np.min(x_znormed)
        xmax = np.max([np.abs(x_znormed.min()), x_znormed.max()])
        
        return x_znormed / xmax

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
