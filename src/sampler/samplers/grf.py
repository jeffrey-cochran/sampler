
# Standard
from typing import Optional

# 3rd Party
import numpy as np
import scipy as sp

# Local
from sampler.boundary_conditions.base import BoundaryConditions
from sampler.samplers.base import __Sampler__
from sampler.utils.type_aliases import (
    NDArray
)

class GRFSampler(__Sampler__):
    """Gaussian Random Field Sampler"""


    def __init__(
        self,
        average:NDArray,
        cov_mat:NDArray,
        boundary_conditions:Optional[BoundaryConditions],
        seed:Optional[int]=None
    ):

        # If desired, fix the random seed
        if seed is not None:
            np.random.seed(seed)

        self.average = average
        self.cov_mat = cov_mat
        self.boundary_conditions = boundary_conditions
        self.fixed_indices:NDArray = [] if boundary_conditions is None else boundary_conditions.indices
        self.free_indices:NDArray = np.setdiff1d(
            np.arange(average.size),
            self.fixed_indices
        )

        # NOTE:
        #   In general, the indices of block `a` are free,
        #   and the indices of block `b` are fixed at the 
        #   assumed boundary conditions.
        self.mu_a = self.average[self.free_indices]

        # Blocks of the covariance matrix, Gamma = [Gamma_aa, Gamma_ab; Gamma_ab.T, Gamma_bb]
        self.Gamma_aa = self.cov_mat[self.free_indices, :][:, self.free_indices]

        # NOTE: If there are no boundary conditions, then
        # the conditional mean is just the mean, and the
        # conditional covariance is just the covariance.
        self.mu_cond = self.mu_a
        self.Gamma_cond = self.Gamma_aa
        if self.boundary_conditions:

            self.mu_b = self.average[self.fixed_indices]
            self.Gamma_ab = self.cov_mat[self.free_indices, :][:, self.fixed_indices]
            self.Gamma_bb = self.cov_mat[self.fixed_indices, :][:, self.fixed_indices]

            # Compute the conditional covariance:
            #   Gamma_cond = Gamma_aa - Gamma_ab * Gamma_bb^-1 * Gamma_ab.T
            tmp_Gamma = np.linalg.solve(self.Gamma_bb, self.Gamma_ab.T)
            self.Gamma_cond = self.Gamma_aa - self.Gamma_ab @ tmp_Gamma

            # Compute the conditional mean:
            #   mu_cond = mu_a + Gamma_ab * Gamma_bb^-1 * (x_b - mu_b)
            tmp_mu = np.linalg.solve(self.Gamma_bb, (self.boundary_conditions.values - self.mu_b))
            self.mu_cond = self.mu_a + self.Gamma_ab @ tmp_mu

        # Get the Cholesky decomposition of the conditional covariance
        self.L_cond = np.linalg.cholesky(self.Gamma_cond)


    def __call__(self, num_samples:int, std_normal_sample:NDArray = None) -> np.ndarray:
        """Retrieves num_samples samples"""

        if std_normal_sample is None:
            std_normal_sample = np.random.standard_normal((num_samples, self.free_indices.size))
        else:
            num_samples = std_normal_sample.shape[0]

        v = (self.L_cond @ std_normal_sample.T).T

        out_samples = np.empty((num_samples, self.average.size))
        out_samples[:, self.free_indices] = self.mu_cond + v

        # If there are boundary conditions, apply them
        if self.boundary_conditions:
            out_samples[:, self.fixed_indices] = self.boundary_conditions.values

        return out_samples