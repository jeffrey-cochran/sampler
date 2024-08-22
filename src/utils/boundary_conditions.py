
# Standard
from dataclasses import dataclass
from typing import Callable
from abc import ABC

# 3rd Party
from numpy.typing import ArrayLike

class BoundaryCondition(ABC):
    pass

@dataclass
class NeumannBC(BoundaryCondition):
    """Container for a Neumann boundary condition
    
    Properties
    ==========
    value (Callable[[ArrayLike], ArrayLike]):
        a function that maps the (x,y) coordinates of the boundary
        to the value of gradient(solution).dot(outward_pointing_normal)
    """
    value: Callable[[ArrayLike], ArrayLike]

@dataclass
class DirichletBC(BoundaryCondition):
    """Container for a Dirichlet boundary condition
    
    Properties
    ==========
    value (Callable[[ArrayLike], ArrayLike]):
        a function that maps the (x,y) coordinates of the boundary
        to the value of the solution
    """
    value: Callable[[ArrayLike], ArrayLike]