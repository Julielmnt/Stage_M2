#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import modred as mr
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import svd, qr, eig, eigh, pinv

class DMD:
    def __init__(self, simu):
        self.U = np.reshape(simu.u - simu.umean, (simu.m, simu.h * simu.l))
        self.W = np.reshape(simu.w - simu.wmean, (simu.m, simu.h * simu.l))
        self.T = np.reshape(simu.T - simu.Tmean, (simu.m, simu.h * simu.l))
        self.X = np.swapaxes(np.concatenate([self.U, self.W, self.T], axis = 1), 0, 1)

    def compute_dmd(self):
        self.V1 = self.X[:, :-1]
        self.V2 = self.X[:, 1:]
        self.U, Sigma, W = svd(self.V1, full_matrices=False)
        W = W.T.conj()
        Sigma_inv = pinv(np.diag(Sigma))
        S = self.U.T.conj() @ self.V2 @ W @ Sigma_inv
        self.eigenvalues, self.eigenvectors = eig(S)
        self.Phi = self.U @ self.eigenvectors
        self.bk = pinv(self.Phi) @ self.X

    def compute_dmd_uw(self):
        self.X = np.swapaxes(np.concatenate([self.U, self.W], axis = 1), 0, 1)
        self.V1 = self.X[:, :-1]
        self.V2 = self.X[:, 1:]
        self.U, Sigma, W = svd(self.V1, full_matrices=False)
        W = W.T.conj()
        Sigma_inv = pinv(np.diag(Sigma))
        S = self.U.T.conj() @ self.V2 @ W @ Sigma_inv
        self.eigenvalues, self.eigenvectors = eig(S)
        self.Phi = self.U @ self.eigenvectors
        self.bk = pinv(self.Phi) @ self.X
        
    def reconstruct_data(self, modes_indexes):
        self.reconstructed_data = np.real(self.Phi[:,modes_indexes] @ self.bk[modes_indexes, :])
        return self.reconstructed_data
    
    def compute_residual_norm(self):
        residual = self.X - self.reconstructed_data
        self.residual_norm = np.linalg.norm(residual, 'fro')
        return self.residual_norm/np.linalg.norm(self.X, 'fro')
    
            
if __name__ == "__main__":
    pass
