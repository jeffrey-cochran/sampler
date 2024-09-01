
# Standard

# 3rd Party
import numpy as np
import scipy as sp

# Local
from sampler.boundary_conditions.base import BoundaryConditions
from sampler.utils.sparse_chol import chol
from sampler.utils.type_aliases import (
    NDArray,
    SparseMatrix
)
from sampler.sampler_base import __Sampler__


class GMRFSampler(__Sampler__):
    """Gaussian Markov Random Field Sampler
    
    """


    def __init__(
        self, *
        average:NDArray,
        prec_mat:SparseMatrix,
        boundary_conditions:BoundaryConditions
    ) -> None:

        self.average = average
        self.prec_mat = prec_mat
        self.boundary_conditions = boundary_conditions
        self.fixed_indices:NDArray = boundary_conditions.indices
        self.free_indices:NDArray = np.setdiff1d(
            np.arange(average.size),
            self.fixed_indices
        )

        # NOTE:
        #   In general, the indices of block `a` are free,
        #   and the indices of block `b` are fixed at the 
        #   assumed boundary conditions.
        self.mu_a = self.average[self.free_indices]
        
        # NOTE:
        #   The conditonal precision matrix does not need
        #   to be computed, because it's simply the submatrix
        #   extracted from the free indices.
        self.Q_aa = self.prec_mat[self.free_indices, :][:, self.free_indices]
        
        # NOTE:
        #   The cholesky factorization of a sparse matrix is
        #   not necessarily spartse. To increase the sparsity
        #   of the cholesky factor, we permute the rows and
        #   columns using the Reverse Cuthill-McKee ordering,
        #   which is returned as P_aa:
        #     P_aa @ Q_aa @ P_aa.T =  G_aa @ G_aa.T 
        self.G_aa, self.P_aa = chol(self.Q_aa)
        self.L_aa = self.P_aa.T @ self.G_aa

        # NOTE: If there are no boundary conditions, then
        # the conditional mean is just the mean.
        self.mu_cond = self.mu_a
        if self.boundary_conditions:

            self.mu_b = self.average[self.fixed_indices]
            self.Q_ab = self.prec_mat[self.free_indices, :][:, self.fixed_indices]

            # NOTE:
            #   This is the conditional beta for the canonical GMRF
            #   x_a - (mu_a | x_b) ~ Nc(beta_cond, Q_aa)
            beta_cond = self.Q_ab @ (self.mu_b - self.boundary_conditions.values)
            w = sp.sparse.linalg.spsolve(self.L_aa, beta_cond)
            self.mu_cond = sp.sparse.linalg.spsolve(self.L_aa.T, w)


    def __call__(self, num_samples:int) -> NDArray:

        z = np.random.standard_normal((num_samples, self.free_indices.size))
        v = sp.sparse.linalg.spsolve(self.L_aa.T, z.T).T

        out_samples = np.empty((num_samples, self.average.size))
        out_samples[:, self.free_indices] = self.mu_cond + v

        # If there are boundary conditions, apply them
        if self.boundary_conditions:
            out_samples[:, self.fixed_indices] = self.boundary_conditions.values

        return out_samples