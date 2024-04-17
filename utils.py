from collections import namedtuple

import ufl
import numpy as np

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

def clamped_boundary(x):
    return np.isclose(x[0], 0)

def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma(u, in_lambda=None, in_mu=None):
    return in_lambda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * in_mu * epsilon(u)
