import numpy as np

import elastic_simulator
import visualize
import domain

D = domain.Domain(
    length=10.,
    width=1.,
    discretization_shape=(20,6,6)
)

ES = elastic_simulator.ElasticSimulator(domain=D, poly_order=1)

uh = ES.simulate(elastic_modulus=100000, poissons_ratio=0.3, density=1.)
u = np.reshape(uh.x.array, D.shape + (3,))
visualize.visualize_deformation(
    domain=D,
    displacements=uh,
    function_space=ES.function_space,
    file_name="displacement",
    exaggeration_factor=1.
)