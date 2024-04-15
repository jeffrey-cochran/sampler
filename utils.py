from collections import namedtuple

LameParams = namedtuple("LameParams", ["MU", "LAMBDA"])

def get_lame_params(elastic_modulus, poissons_ratio):
    return LameParams(
        MU = elastic_modulus / (2.0 * (1.0 + poissons_ratio)),
        LAMBDA = elastic_modulus * poissons_ratio / (
            (1. + poissons_ratio)*(1.-2.*poissons_ratio)
        )
    )