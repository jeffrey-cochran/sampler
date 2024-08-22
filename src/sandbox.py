from samplers.unit_square import UnitSquareSampler
from utils.boundary_conditions import NeumannBC, DirichletBC
import numpy as np

a = np.sin

neumann = NeumannBC(value=a)
dirichlet = DirichletBC(value=a)

b = UnitSquareSampler(
    average=a,
    cov_mat=a,
    bc_top=dirichlet,
    bc_bot=dirichlet,
    bc_left=None,
    bc_right=dirichlet
)

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