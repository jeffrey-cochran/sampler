# Standard
from typing import Union, Tuple, Callable

# 3rd Party
import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
from nutils import topology as nutils_topology
from nutils import function as nutils_function

# Local
# N/A


NutilsFunctionArray = nutils_function.Array
NutilsTopology = nutils_topology.Topology
NDArray = np.ndarray
SparseMatrix = sp.sparse.spmatrix
ArrayPlaceHolder = Union[NDArray, bool, None]
BoundaryConditionFunction = Callable[[ArrayLike], ArrayLike]
Indices = Tuple[ArrayLike, ...]
