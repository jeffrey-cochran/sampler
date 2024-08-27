
# Standard
import contextlib
import io
from functools import reduce
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

# 3rd Party
import numpy as np
import scipy as sp
from nutils import topology as nutils_topology
from nutils import function as nutils_function
from nutils import mesh as nutils_mesh

# Local
from samplers.sampler_base import Sampler
from utils.boundary_conditions import (
    BoundaryConditions,
    BoundaryCondition,
    NeumannBC,
    DirichletBC
)

# Type aliases
NutilsFunctionArray = nutils_function.Array
NutilsTopology = nutils_topology.Topology
NDArray = np.ndarray
SparseMatrix = sp.sparse.spmatrix
ArrayPlaceHolder = Union[NDArray, bool, None]


class UnitSquareBoundaryConditions(BoundaryConditions):

    def __init__(
        self, *,
        top:Optional[BoundaryCondition],
        bot:Optional[BoundaryCondition],
        left:Optional[BoundaryCondition],
        right:Optional[BoundaryCondition],
        x_dim:int,
        y_dim:int,
        consistency_rtol:float = 1.e-5
    ):

        # Ravel the 2D indices to 1D global indices
        # NOTE:
        #   multi-index ravel does not seem to work with '-1', and
        #   it is necessary to use N-1 to ravel the last index value.
        self.bot   = self.__init_boundary_condition(bot,   x_idx=np.arange(x_dim),            y_idx=np.zeros((x_dim,)))
        self.top   = self.__init_boundary_condition(top,   x_idx=np.arange(x_dim),            y_idx=np.ones((x_dim,))*(x_dim-1))
        self.left  = self.__init_boundary_condition(left,  x_idx=np.zeros((y_dim,)),          y_idx=np.arange(y_dim))
        self.right = self.__init_boundary_condition(right, x_idx=np.ones((y_dim,))*(y_dim-1), y_idx=np.arange(y_dim))

        # Confirm that there are no inconsistent boundary conditions
        self.problem_corners = self.__get_problem_corners(rtol=consistency_rtol)
        if not self.are_consistent:
            raise RuntimeError(
                "The provided boundary conditions do not agree at the %s corner(s) "
                "of the unit square. Aborting!" % ', '.join(self.problem_corners)
            )

        self.values, self.indices = None, None
        if self:
            # Repeated indices correspond to corners, which are already required
            # to be consistent, so we can remove them.
            joint_idx = np.hstack([bc.indices for bc in self if bc is not None])
            joint_value = np.hstack([bc.value for bc in self if bc is not None])
            _, unique_idx = np.unique(joint_idx, return_index=True)
            #
            self.values = joint_value[unique_idx]
            self.indices = joint_idx[unique_idx]


    def __bool__(self):
        return reduce(
            lambda has_bcs, bc: has_bcs or bc is not None,
            self,
            False
        )


    def __init_boundary_condition(
            self,
            boundary_condition:BoundaryCondition,
            x_idx:NDArray,
            y_idx:NDArray
        ):

        if boundary_condition is not None:
            boundary_condition.indices = np.ravel_multi_index(
                (
                    x_idx.astype(int),
                    y_idx.astype(int)
                ),
                (x_idx.size, y_idx.size)
            )

        return boundary_condition


    def __iter__(self):
        return iter([self.top, self.bot, self.left, self.right])

    def __get_problem_corners(
            self,
            rtol:float = 1.e-5
        ) -> List[str]:

        # NOTE:
        #   - bcs are the boundary conditions that overlap
        #   - idx are the indices where boundary conditions overlap
        #   - key describes the location where overlap occurs
        bcs = [(self.left, self.top), (self.right, self.top), (self.left, self.bot), (self.right, self.bot)]
        idx = [(-1, 0), (-1, -1), (0, 0), (0, -1)]
        key = ["top-left", "top-right", "bot-left", "bot-right"]

        problem_corners = []
        for b, i, k in zip(bcs, idx, key):
            # NOTE:
            #   If either boundary condition is None, then 
            #   a conflict is impossible. Otherise, check
            #   that the overlapping values agree to within rtol
            bcs_are_consistent = (
                    b[0] is None
                or  b[1] is None
                or  np.isclose(b[0].value[i[0]], b[1].value[i[1]], rtol=rtol)
            )
            
            if not bcs_are_consistent:
                problem_corners.append(k)
        #
        return problem_corners


    @property
    def are_consistent(self):
        return len(self.problem_corners) == 0



class UnitSquareSampler(Sampler):


    def __init__(
        self, *,
        average:NDArray,
        poly_order:int,
        cov_mat:NDArray = None,
        prec_mat:SparseMatrix = None,
        bc_top:BoundaryCondition = None,
        bc_bot:BoundaryCondition = None,
        bc_left:BoundaryCondition = None,
        bc_right:BoundaryCondition = None,
        rtol:float = 1.e-5
    ):
        """Initialize UnitSquareSampler
        
        Required Keyword Arguments
        ==========================
        average (NDArray):
            An array of the average coefficient values at each knot. The dimensions
            of the knot vectors are determined by the size of this argument.
        poly_order (int):
            The polynomial order of the b-spline basis.
        cov_mat (SparseMatrix):
            The covariance matrix of the Gaussian Random Field. The function only
            accepts cov_mat XOR prec_mat. If the field has a known, sparse precision
            matrix, then sampling may be faster using the precision matrix.
        prec_mat (SparseMatrix):
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
        rtol (float):
            The relative tolerance with which boundary conditions are forced to be consistent
            at the corners.
        """

        self.poly_order = poly_order

        # NOTE: 
        #   Only storing one matrix: self.mat is either covariance or precision. If
        #   the precision matrix is stored, self.is_gmrf == True 
        self.x_dim, self.y_dim = self.__init_dims(average_coeffs=average)
        self.average = average.flatten()
        self.mat, self.is_gmrf = self.__init_mat(cov_mat=cov_mat, prec_mat=prec_mat)

        self.ns = nutils_function.Namespace()
        self.topo, self.ns.xy = self.__init_geom(poly_order=poly_order)
        self.ns.basis = self.__init_basis(topo=self.topo, poly_order=poly_order)

        self.boundary_conditions = self.__init_boundary_conditions(
            bc_top=bc_top,
            bc_bot=bc_bot,
            bc_left=bc_left,
            bc_right=bc_right,
            poly_order=poly_order,
            rtol=rtol
        )

        self.fixed_indices = self.boundary_conditions.indices
        self.free_indices = np.setdiff1d(np.arange(self.x_dim*self.y_dim), self.fixed_indices)

        # self.sample = self.sample_gmrf if is_gmrf else self.sample_grf


    def __init_boundary_conditions(
        self, *,
        bc_top:BoundaryCondition,
        bc_bot:BoundaryCondition,
        bc_left:BoundaryCondition,
        bc_right:BoundaryCondition,
        poly_order:int,
        rtol:float
    ):
        """Sets the boundary conditions.

        Currently, Neumann boundary conditions are not supported,
        and they raise a RuntimeError.
        """

        # Iterates over boundary conditions, checking for Neumann
        has_neumann_bcs = reduce(
            lambda has_neumann, next_bc: has_neumann or (next_bc is not None and isinstance(next_bc, NeumannBC)),
            [bc_top, bc_bot, bc_left, bc_right],
            False
        )

        if has_neumann_bcs:
            raise RuntimeError("Neumann conditions are not currently supported. Aborting!")
        
        # Create temporary topology, geometry, and basis for projecting the
        # boundary conditions. This is only done for convenience.
        x_topo, x_geom = nutils_mesh.rectilinear([np.linspace(0, 1, self.x_dim-poly_order+1)])
        y_topo, y_geom = nutils_mesh.rectilinear([np.linspace(0, 1, self.y_dim-poly_order+1)])
        #
        x_basis = x_topo.basis('spline', degree=poly_order)
        y_basis = y_topo.basis('spline', degree=poly_order)

        # Capturing the projection solver output because I don't see
        # an argument for "silent" or "verbose", and I find it annoying.
        with contextlib.redirect_stdout(io.StringIO()) as _:
                bc_top   = self.project_onto_boundary(boundary_condition=bc_top,   topo=x_topo, basis=x_basis, geometry=x_geom, poly_order=poly_order)
                bc_bot   = self.project_onto_boundary(boundary_condition=bc_bot,   topo=x_topo, basis=x_basis, geometry=x_geom, poly_order=poly_order)
                bc_left  = self.project_onto_boundary(boundary_condition=bc_left,  topo=y_topo, basis=y_basis, geometry=y_geom, poly_order=poly_order)
                bc_right = self.project_onto_boundary(boundary_condition=bc_right, topo=y_topo, basis=y_basis, geometry=y_geom, poly_order=poly_order)

        out_bcs = UnitSquareBoundaryConditions(
            top   = bc_top,
            bot   = bc_bot,
            left  = bc_left,
            right = bc_right,
            x_dim = len(x_basis),
            y_dim = len(y_basis)
        )

        return out_bcs


    def __init_basis(self, topo:NutilsTopology, poly_order:int) -> NutilsFunctionArray:
        return topo.basis('spline', degree=poly_order)
    
    
    def __init_dims(self, average_coeffs:NDArray) -> Tuple[int,int]:
        if len(average_coeffs.shape) != 2:
            raise RuntimeError("The array of average coefficients must be 2D. Aborting!")
    
        return average_coeffs.shape


    def __init_geom(self, poly_order:int) -> Tuple[NutilsTopology, NutilsFunctionArray]:
        # NOTE:
        #   The average values of the coefficients are provided; however, the
        #   number of knots must be inferred dynamically based on this and the
        #   degree of the spline basis.
        return nutils_mesh.rectilinear(
            [
                np.linspace(0, 1, self.x_dim-poly_order+1),
                np.linspace(0, 1, self.y_dim-poly_order+1)
            ]
        )


    def __init_mat(self, cov_mat:NDArray, prec_mat:SparseMatrix) -> Tuple[Union[NDArray, SparseMatrix], bool]:
        is_gmrf = cov_mat is None
        if is_gmrf == prec_mat is None:
            raise RuntimeError("Must provide exactly one of prec_mat XOR cov_mat. Aborting!")
        
        out_mat = prec_mat if is_gmrf else cov_mat
        if (
                self.x_dim*self.y_dim != out_mat.shape[0]
            or  out_mat.shape[0] != out_mat.shape[1]
        ):
            raise RuntimeError("Precision/covariance matrix must be NxN, where N = average.size. Aborting!")

        return (
            out_mat,
            is_gmrf
        )


    def __sample_gmrf(self, num_samples:int):

        Laa = getattr(self, 'Laa', None)

        if Laa is None:
            Laa = 
        except AttributeError:

            Qaa = self.mat[self.free_indices, :][:, self.free_indices]
            Qbb = self.mat[self.fixed_indices, :][:, self.fixed_indices]
            Qab = self.mat[self.free_indices, :][:, self.fixed_indices]

        return 0


    @classmethod
    def project_onto_boundary(
        cls, *,
        boundary_condition:BoundaryCondition,
        topo:NutilsTopology,
        basis:NutilsFunctionArray,
        geometry:NutilsFunctionArray,
        poly_order:int
    ) -> BoundaryCondition:
        
        if boundary_condition is None:
            return None

        # NOTE:
        #   If the boundaries are Dirichlet, then we can only ensure
        #   consistency of the values at the corners are 'exact'
        use_exact_boundaries = isinstance(boundary_condition, DirichletBC)

        if (func:= boundary_condition.func) is not None:
            boundary_condition.value = topo.project(
                func(geometry[0]),
                onto=basis,
                geometry=geometry,
                ptype="lsqr",
                degree=poly_order,
                exact_boundaries=use_exact_boundaries
            )

        print("HELLO")

        # if boundary_condition.value is None:
        #     raise RuntimeError("BoundaryCondition[%s] has neither function nor value. Aborting!" % str(boundary_condition.id))
        # elif boundary_condition.value.size != basis.size:
        #     raise RuntimeError("BoundaryCondition[%s].value.size does not match basis.size. Aborting!" % str(boundary_condition.id))
            
        return boundary_condition
