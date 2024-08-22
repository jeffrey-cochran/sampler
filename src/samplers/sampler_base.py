
# Standard
import abc

# 3rd Party
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Local

class Sampler(abc.ABC):

    def sample(self, num_samples) -> np.ndarray:
        """Retrieves num_samples samples"""
        pass

    def visualize_sample(self, sample) -> None:
        """Visualize a single sample"""
        pass
