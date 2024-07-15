from collections import namedtuple

import ufl
import numpy as np
from constants import BoundaryFlag, DIM

LameParams = namedtuple("LameParams", ["MU", "LAMBDA"])

def get_lame_params(
    elastic_modulus:float,
    poissons_ratio:float
):
    return LameParams(
        MU = elastic_modulus / (2.0 * (1.0 + poissons_ratio)),
        LAMBDA = elastic_modulus * poissons_ratio / (
            (1. + poissons_ratio)*(1.-2.*poissons_ratio)
        )
    )

def get_locator(boundary_flag:BoundaryFlag, point_on_boundary):
    current_dim = DIM[boundary_flag]
    return (
        boundary_flag,
        lambda x: np.isclose(x[current_dim], point_on_boundary[current_dim])
    )
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma(u, in_lambda=None, in_mu=None):
    return in_lambda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * in_mu * epsilon(u)
