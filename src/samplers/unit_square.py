
# Standard
from functools import reduce
from typing import Tuple, Union

# 3rd Party
import numpy as np
import scipy as sp

# Local
from samplers.sampler_base import Sampler
from utils.boundary_conditions import BoundaryCondition, NeumannBC, DirichletBC

class UnitSquareSampler(Sampler):

    def __init__(
        self, *,
        average:np.ndarray,
        cov_mat:np.ndarray = None,
        prec_mat:sp.sparse.spmatrix = None,
        bc_top:BoundaryCondition = None,
        bc_bot:BoundaryCondition = None,
        bc_left:BoundaryCondition = None,
        bc_right:BoundaryCondition = None
    ):
        """Initialize UnitSquareSampler
        
        Required Keyword Arguments
        ==========================
        average (np.ndarray):
            An array of the average coefficient values at each knot. The dimensions
            of the knot vectors are determined by the size of this argument.
        cov_mat (sp.sparse.spmatrix):
            The covariance matrix of the Gaussian Random Field. The function only
            accepts cov_mat XOR prec_mat. If the field has a known, sparse precision
            matrix, then sampling may be faster using the precision matrix.
        prec_mat (sp.sparse.spmatrix):
            The precision matrix of the Gaussian Random Field. The function only
            accepts cov_mat XOR prec_mat. If the field does not have a known,
            sparse precision matrix, then it probably makes more sense to pass
            the covariance matrix.
        bc_top (BoundaryCondition):
            The boundary condition to be enforced on the top of the unit square (y=1).
            If None, the value at the top boundary will be random.
        bc_bot (BoundaryCondition):
            The boundary condition to be enforced on the bottom of the unit square (y=0).
            If None, the value at the bottom boundary will be random.
        bc_left (BoundaryCondition):
            The boundary condition to be enforced on the left of the unit square (x=0).
            If None, the value at the left boundary will be random.
        bc_right (BoundaryCondition):
            The boundary condition to be enforced on the right of the unit square (x=1).
            If None, the value at the right boundary will be random.
        """

        is_gmrf = cov_mat is None
        if is_gmrf == prec_mat is None:
            raise RuntimeError("Must provide exactly one of prec_mat XOR cov_mat. Aborting!")

        self.boundary_conditions = (bc_top, bc_bot, bc_left, bc_right)
        # self.sample = self.sample_gmrf if is_gmrf else self.sample_grf

    @property
    def boundary_conditions(self) -> Tuple[
        BoundaryCondition, # top
        BoundaryCondition, # bot
        BoundaryCondition, # left
        BoundaryCondition  # right
    ]:
        return self.boundary_conditions_

    @boundary_conditions.setter
    def boundary_conditions(
        self,
        in_boundary_conditions:Tuple[
            BoundaryCondition, # top
            BoundaryCondition, # bot
            BoundaryCondition, # left
            BoundaryCondition  # right
        ]
    ):
        """Sets the boundary conditions.
        
        The assumed order of the boundary conditions is:
            (bc_top, bc_bot, bc_left, bc_right)

        Currently, Neumann boundary conditions are not supported,
        and they raise a RuntimeError.
        """

        has_neumann_bcs = reduce(
            lambda has_neumann, next_bc: has_neumann or (next_bc is not None and isinstance(next_bc, NeumannBC)),
            in_boundary_conditions,
            False
        )

        if has_neumann_bcs:
            raise RuntimeError("Neumann conditions are not currently supported. Aborting!")

        self.boundary_conditions_ = in_boundary_conditions



    

