
# Standard
import abc

# 3rd Party
import numpy as np


class Sampler(abc.ABC):

    def sample(self, num_samples:int) -> np.ndarray:
        """Retrieves num_samples samples"""
        pass

    def visualize_sample(self, sample:np.ndarray) -> None:
        """Visualize a single sample"""
        pass
