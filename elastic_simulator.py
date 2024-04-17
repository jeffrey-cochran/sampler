from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np

import utils
import geometry


class ElasticSimulator(object):

    def __init__(
        self, *,
        geometry:geometry.Geometry = None
    ):
        self.geometry = geometry
        return

    def simulate(
        self, *,
        elastic_modulus:float = None,
        poissons_ratio:float = None,
        density:float = None
    ):
        lame_params = utils.get_lame_params(elastic_modulus=elastic_modulus, poissons_ratio=poissons_ratio)        

        L = self.geometry.length
        W = self.geometry.width
        delta = W / L
        gamma = 0.4 * delta**2
        g = gamma

        domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
                         [20, 6, 6], cell_type=mesh.CellType.hexahedron)
        V = fem.VectorFunctionSpace(domain, ("Lagrange", 1))

        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, utils.clamped_boundary)

        u_D = np.array([0, 0, 0], dtype=default_scalar_type)
        bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

        T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

        ds = ufl.Measure("ds", domain=domain)
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        f = fem.Constant(domain, default_scalar_type((0, 0, -density * g)))
        a = ufl.inner(
            utils.sigma(u, in_lambda=lame_params.LAMBDA, in_mu=lame_params.MU),
            utils.epsilon(v)
        ) * ufl.dx
        Lmat = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

        problem = LinearProblem(a, Lmat, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        return uh