
# Standard
from dataclasses import dataclass
from typing import Optional, Hashable
from abc import ABC

# 3rd Party
from numpy.typing import ArrayLike

# Local
from sampler.utils.type_aliases import BoundaryConditionFunction, NDArray, Indices


@dataclass
class BoundaryCondition:
    """Container for a boundary condition
    
    Properties
    ==========
    func (BoundaryConditionFunction):
        a function that maps the (x,y) coordinates of the boundary
        to the value of gradient(solution).dot(outward_pointing_normal)
    value (ArrayLike):
        an array of boundary condition values; note that value.size must be
        equal to indices[0].size
    indices (ArrayLike):
        the indices of the GRF at which the boundary condition values apply;
        note that indices[0].size must be equal to value.size
    """
    func:Optional[BoundaryConditionFunction] = None
    value:Optional[ArrayLike] = None
    indices:Optional[Indices] = None
    id:Optional[Hashable] = "no_id_provided"


@dataclass
class NeumannBC(BoundaryCondition):
    """Convience class to distinguish types of boundary conditions"""


@dataclass
class DirichletBC(BoundaryCondition):
    """Convience class to distinguish types of boundary conditions"""


class BoundaryConditions(ABC):
    """Abstract container for holding boundary conditions"""

    @property
    def values(self) -> NDArray:
        """Returns the values corresponding to self.indices"""

    @property
    def indices(self) -> NDArray:
        """Returns the indices of the boundary conditions"""

    @property
    def are_consistent(self) -> bool:
        """Returns true if the boundary conditions agree at the intersection of their domains"""
