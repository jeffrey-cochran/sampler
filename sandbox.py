import elastic_simulator
import geometry

G = geometry.Geometry(length=10., width=1.)

ES = elastic_simulator.ElasticSimulator(geometry=G)

uh = ES.simulate(elastic_modulus=1.e9, poissons_ratio=0.3, density=1.)