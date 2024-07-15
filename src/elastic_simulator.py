from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

import utils
import constants
import domain


class ElasticSimulator(object):

    def __init__(
        self, *,
        domain:domain.Domain = None,
        poly_order:int = None
    ):
        self.domain = domain
        self.function_space = fem.VectorFunctionSpace(
            self.domain.mesh,
            ("Lagrange", poly_order)
        )
        return

    def simulate(
        self, *,
        elastic_modulus:float = None,
        poissons_ratio:float = None,
        density:float = None,
    ):
        # The variational formulation is stated in terms
        # of the Lame parameters mu and lambda, but engineers
        # typically discuss material properties in terms of 
        # the elastic modulus and poisson's ratio. There is a
        # one-to-one correspondence, so we can convert from 
        # the intuitive material properties to the variational ones.
        lame_params = utils.get_lame_params(elastic_modulus=elastic_modulus, poissons_ratio=poissons_ratio)        
        
        # Lagrangian discretized function space and
        # functions defined on that space
        u = ufl.TrialFunction(self.function_space)
        v = ufl.TestFunction(self.function_space)
        uh = fem.Function(self.function_space)

        # Without loss of generalization, we constrain the
        # left face to homogeneous dirichlet conditions. Any
        # other essential boundary conditions should be 
        # imposable by rewriting the system.
        u_D = np.array([0, 0, 0], dtype=default_scalar_type)
        bc = fem.dirichletbc(
            u_D,
            fem.locate_dofs_topological(
                self.function_space,
                self.domain.facet_dim,
                self.domain.left
            ),
            self.function_space
        )

        T = fem.Constant(self.domain.mesh, default_scalar_type((0, 0, -density * constants.GRAVITY)))
        f = fem.Constant(self.domain.mesh, default_scalar_type((0, 0, -density * constants.GRAVITY)))
        
        # Lefthand side (bilinear form)
        lhs = ufl.inner(
            utils.sigma(u, in_lambda=lame_params.LAMBDA, in_mu=lame_params.MU),
            utils.epsilon(v)
        ) * ufl.dx

        # Righthand side (forcing term)
        # NOTE:
        #   "ds" indicates integration over an external facet
        #   "dx" indicates integration over volume
        # Ignore body forces for now
        ds = ufl.Measure("ds", domain=self.domain.mesh)
        # rhs = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds 
        rhs = ufl.dot(T, v) * ds 
        # Solve
        problem = LinearProblem(
            a=lhs,
            L=rhs,
            bcs=[bc],
            u=uh,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        problem.solve()

        return uh