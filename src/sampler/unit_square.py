
# Standard
import contextlib
import io
from functools import reduce
from typing import Tuple, Union

# 3rd Party
import numpy as np
from nutils import function as nutils_function
from nutils import mesh as nutils_mesh
import matplotlib.tri as tri
import matplotlib.pyplot as plt

# Local
from sampler.boundary_conditions.base import (
    BoundaryCondition,
    NeumannBC,
    DirichletBC
)
from sampler.boundary_conditions.unit_square import UnitSquareBoundaryConditions
from sampler.sampler_base import (
    __Sampler__,
    Sampler,
    GMRFSampler,
    GRFSampler
)
from sampler.utils.type_aliases import (
    NDArray,
    SparseMatrix,
    NutilsFunctionArray,
    NutilsTopology
)


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
        cov_mat (NDArray):
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

        self.sample = self.__init_sampler(
            is_gmrf=self.is_gmrf,
            average=self.average,
            mat=self.mat,
            boundary_conditions=self.boundary_conditions
        )


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
            y_dim = len(y_basis),
            consistency_rtol=rtol
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
        if is_gmrf == (prec_mat is None):
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


    def __init_sampler(
        self, *,
        is_gmrf:bool,
        average:NDArray,
        mat:Union[NDArray, SparseMatrix],
        boundary_conditions:UnitSquareBoundaryConditions
    ) -> __Sampler__:

        out_sampler:__Sampler__ = None
        if is_gmrf:
            out_sampler = GMRFSampler(
                average=average,
                prec_mat=mat,
                boundary_conditions=boundary_conditions
            )
        else:
            out_sampler = GRFSampler(
                average=average,
                cov_mat=mat,
                boundary_conditions=boundary_conditions
            )

        return out_sampler


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
        #   consistency if the values at the corners are 'exact'
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

        if boundary_condition.value is None:
            raise RuntimeError("BoundaryCondition[%s] has neither function nor value. Aborting!" % str(boundary_condition.id))
        elif boundary_condition.value.size != basis.size:
            raise RuntimeError("BoundaryCondition[%s].value.size does not match basis.size. Aborting!" % str(boundary_condition.id))

        return boundary_condition


    def visualize_sample(self, sample:NDArray=None, degree=9, title='Func Value') -> None:

        viz_sampler = self.topo.sample('bezier', degree)

        sample = sample if sample is not None else self.sample(1).flatten()
        self.ns.f = np.dot(self.ns.basis, sample)

        x, f = viz_sampler.eval(['xy_i', 'f'] @ self.ns)

        f /= (1.1*np.abs(f).max())

        triangulation = tri.Triangulation(x[:,0], x[:,1], viz_sampler.tri)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        levels = np.arange(-1., 1., 0.05)

        ax1.set_title(title, fontsize=40)
        ax1.triplot(triangulation, lw=0.5, color='white')
        contour = ax1.tricontourf(triangulation, f.flatten(), levels=levels, cmap='bwr')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(contour, cax=cbar_ax)
        
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.show()
