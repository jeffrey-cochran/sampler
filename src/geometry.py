import abc
from functools import reduce
from enum import IntFlag, auto

import numpy as np
import numpy.typing as npt


ArrayInt = npt.NDArray[np.int_]


class Geometry(abc.ABC):

    @abc.abstractmethod
    def boundary_idx(self, *boundary_flags:IntFlag, complement:bool=False) -> ArrayInt:
        "Returns the indices corresponding to that boundary"


class UnitSquare(Geometry):
    
    class BoundaryFlag(IntFlag):
        BOTTOM = auto()
        LEFT = auto()
        TOP = auto()
        RIGHT = auto()
        X = LEFT | RIGHT
        Y = TOP | BOTTOM

    def __init__(self, x_dim:int, y_dim=None):
        self.x_dim = x_dim
        self.y_dim = x_dim if y_dim is None else x_dim

        x_arange = np.arange(self.x_dim)
        x_zeros = np.zeros((self.x_dim,), dtype=int)

        y_arange = np.arange(self.y_dim)
        y_zeros = np.zeros((self.y_dim,), dtype=int)

        self.boundary_idx_dict = {
            self.BoundaryFlag.BOTTOM: self.ravel(x_zeros, y_arange),
            self.BoundaryFlag.TOP: self.ravel(x_zeros+self.x_dim-1, y_arange),
            self.BoundaryFlag.LEFT: self.ravel(x_arange, y_zeros),
            self.BoundaryFlag.RIGHT: self.ravel(x_arange, y_zeros+self.y_dim-1)
        }

    def boundary_idx(self, *boundary_flags:IntFlag, complement:bool=False) -> ArrayInt:
        out_idx = reduce(
            np.union1d,
            [
                self.boundary_idx_dict[boundary_flag]
                for boundary_flag in boundary_flags
            ]
        )

        if complement:
            all_boundary_idx = reduce(
                np.union1d,
                [
                    self.boundary_idx_dict[boundary_flag]
                    for boundary_flag in self.boundary_idx_dict.values()
                ]
            )
            out_idx = np.setdiff1d(all_boundary_idx, out_idx)

        return out_idx

    def ravel(self, x_idx: ArrayInt, y_idx: ArrayInt):
        """Returns the indices in the flattened array"""
        return np.ravel_multi_index(
            (
                x_idx,
                y_idx
            ),
            (self.x_dim, self.y_dim)
        )
    


