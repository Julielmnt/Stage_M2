# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import svd, qr, eig, eigh
from scipy.linalg import pinv2 as pinv
from scipy.linalg import inv as inv
from scipy.sparse.linalg import svds







def dmd_analysis(x, y=None, rank=2, rtol=1e-6):
  """
  Implementation of the closed-form solution of the DMD optimization problem
  using its SVD formulation. This implementation will be far more efficient in
  the case where x and y are tall skinny matrices (i.e. far more degrees of
  freedom than samples, classical situation in fluid dynamics).
  INPUTS
  ------
  x : Array-like. shape (n, k).
      Input time-series of the state vector.
      n degrees of freedom and k samples have been collected.
  y : Array-like. shape (n, k). (optional).
      Input time-series of the state vector one time-step ahead.
  rank : int (default = 2).
         Desired rank for the DMD model to be identified.
  RETURNS
  -------
  sigma : array-like, shape (rank,)
          Leading eigenvalues of the symmetric positive definite matrix
          from the DMD optimization problem.
  P : array-like, shape (n, rank).
      Corresponding leading eigenvectors.
  Q : array-like, shape (n, rank).
      Matrix enabling the construction of the low-rank DMD operator as P @ Q.T.conj().
  """
  
  # --> Partition the data matrix into X and Y.
  if y is None:
    x, y = x[:, :-1], x[:, 1:]
    
  # --> Compute the SVD of x. {star=transpose conjugate}
  Ux, Sx, Vx = svd(x, full_matrices=False) #{Vx=Vstar and VxV=I UxUstar; @ is matrix multiplier}
  Vx = Vx.T.conj()
  
  # --> Compute the SVD of Y @ Vx @ Sx @ pinv(Sx)
  P, sigma, _ = svds(y @ Vx @ np.diag(Sx >= Sx.max()*rtol), k=rank)
  
  # --> Compute Q.
  Q = np.linalg.multi_dot([Ux, pinv(np.diag(Sx), cond=rtol, rcond=rtol), Vx.T.conj(), y.T, P])
  
  return sigma, P, Q

def evd_dmd(P, Q):
    """
    Utility function to return the eigendecomposition of the low-rank DMD operator.
    INPUTS
    ------
    P : Array-like, shape (n, k).
    Q : Array-like, shape (n, k).
    RETURNS
    -------
    Phi : Array-like, shape (n, k).
          Right eigenvectors of the low-rank DMD operator.
          Correspond to the classical DMD modes.
    Psi : Array-like, shape (n, k).
          Left eigenvectors of the low-rank DMD operator.
          Correspond to the adjoint DMD modes.
    mu : Array-like, shape (k,).
         Eigenvalues of the low-rank DMD operator.
    """

    # --> Compute the left and right eigenvectors of the low-dimensional DMD operators.
    vals, vecs_right = eig(Q.T.conj() @ P)
    vecs_left = inv(vecs_right).T.conj() # LAC: I had to import inv from linalg

    # --> Build the high-dimensional eigenvectors.
    Phi = P @ vecs_right
    Psi = Q @ vecs_left

    # --> Normalize the adjoint eigenvectors to respect the bi-orthogonality.
    mu = np.sum(Psi.conj() * Phi, axis=0) # LAC: I had to turn m into mu here
    Psi = Psi @ pinv(np.diag(mu))

    return Phi, Psi, mu


def svd_dmd(P, Q):
    """
    Utility function to return the singular value decomposition of the low-rank
    DMD operator.
    INPUTS
    ------
    P : Array-like, shape (n, k).
    Q : Array-like, shape (n, k).
    RETURNS
    -------
    U : Array-like, shape (n, k).
        Left singular vectors forming an orthonormal basis for the output space.
    V : Array-like shape (n, k).
        Right singular vectors forming an orthonormal basis for the input space.
    sigma : Array-like. shape (k,).
            Singular values of the low-rank DMD operator.
    """

    # --> Compute QR decomposition of the input Q matrix. (Bad name!)
    Q, R = qr(Q, mode='economic')

    # --> Compute the SVD of R.T.conj().
    u, sigma, vh = svd(R.T.conj())
    v = vh.T.conj()

    # --> Compute the SVD factorization of the high-dimensional low-rank DMD operator.
    U, V = P @ u, Q @ v

    return U, V, sigma
