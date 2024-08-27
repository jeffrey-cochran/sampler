from samplers.unit_square import UnitSquareSampler
from utils.boundary_conditions import NeumannBC, DirichletBC
import numpy as np

from nutils import function, mesh


# topo, geom = mesh.rectilinear([np.linspace(0,1,9)])
# basis = topo.basis('spline', degree=3)

# exact = np.sin(geom[0])

# projected = topo.project(exact, onto=basis, geometry=geom, ptype='lsqr', degree=3)

# print(projected)

average = np.zeros((11,11))
cov = np.ones((121,121))

def sin_bc(x):
    return np.sin(2*np.pi*x)

def cos_bc(x):
    return np.cos(2*np.pi*x)

# NOTE:
#   Neumann conditions are currently unsupported
neumann = NeumannBC(func=sin_bc, id="neumann")

# NOTE:
#   If you want you reuse the same boundary condition,
#   you'll either need to do a deep copy or just create
#   multiple instances; otherwise, the computed indices
#   will be overwritten each time the referenced BC is
#   evaluated.
dirichlet_1 = DirichletBC(func=sin_bc, id="dirichlet1")
dirichlet_2 = DirichletBC(func=sin_bc, id="dirichlet2")
dirichlet_3 = DirichletBC(func=sin_bc, id="dirichlet3")

b = UnitSquareSampler(
    average=average,
    poly_order=4,
    cov_mat=cov,
    bc_top=dirichlet_1,
    bc_bot=None,
    bc_left=None,
    bc_right=None
)

if b.boundary_conditions:
    print(b.boundary_conditions.values, b.boundary_conditions.indices)

# import numpy as np

# import elastic_simulator
# import visualize
# import domain

# D = domain.Domain(
#     length=10.,
#     width=1.,
#     discretization_shape=(20,6,6)
# )

# ES = elastic_simulator.ElasticSimulator(domain=D, poly_order=1)

# uh = ES.simulate(elastic_modulus=100000, poissons_ratio=0.3, density=1.)
# u = np.reshape(uh.x.array, D.shape + (3,))
# visualize.visualize_deformation(
#     domain=D,
#     displacements=uh,
#     function_space=ES.function_space,
#     file_name="displacement",
#     exaggeration_factor=1.
# )