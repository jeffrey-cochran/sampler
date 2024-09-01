
# Standard
import abc

# 3rd Party
import numpy as np

# Local
from sampler.utils.type_aliases import NDArray


class __Sampler__(abc.ABC):
    """Abstract base class for hidden samplers called in Sampler"""
    
    def __call__(self, num_samples:int) -> NDArray:
        """Retrieves num_samples samples"""


class Sampler(abc.ABC):
    """Abstract base class for samplers.
    
    Generally, will contain a __Sampler__ object that is called
    in Sampler.sample(), but doesn't have to.
    """

    def sample(self, num_samples:int) -> NDArray:
        """Retrieves num_samples samples"""
        pass

    def visualize_sample(self, sample:NDArray) -> None:
        """Visualize a single sample"""
        pass
