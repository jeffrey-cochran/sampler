
# Standard
from functools import reduce
from typing import Optional, List

# 3rd Party
import numpy as np

# Local
from sampler.boundary_conditions.base import (
    BoundaryConditions,
    BoundaryCondition
)
from sampler.utils.type_aliases import (
    NDArray
)


class UnitSquareBoundaryConditions(BoundaryConditions):


    def __init__(
        self, *,
        top:Optional[BoundaryCondition],
        bot:Optional[BoundaryCondition],
        left:Optional[BoundaryCondition],
        right:Optional[BoundaryCondition],
        x_dim:int,
        y_dim:int,
        consistency_rtol:float = 1.e-5
    ):

        self.field_shape = (x_dim, y_dim)

        # Ravel the 2D indices to 1D global indices
        # NOTE:
        #   multi-index ravel does not seem to work with '-1', and
        #   it is necessary to use N-1 to ravel the last index value.
        self.bot   = self.__init_boundary_condition(bot,   x_idx=np.arange(x_dim),            y_idx=np.zeros((x_dim,)))
        self.top   = self.__init_boundary_condition(top,   x_idx=np.arange(x_dim),            y_idx=np.ones((x_dim,))*(y_dim-1))
        self.left  = self.__init_boundary_condition(left,  x_idx=np.zeros((y_dim,)),          y_idx=np.arange(y_dim))
        self.right = self.__init_boundary_condition(right, x_idx=np.ones((y_dim,))*(x_dim-1), y_idx=np.arange(y_dim))

        # Confirm that there are no inconsistent boundary conditions
        self.problem_corners = self.__get_problem_corners(rtol=consistency_rtol)
        if not self.are_consistent:
            raise RuntimeError(
                "The provided boundary conditions do not agree at the %s corner(s) "
                "of the unit square. Aborting!" % ', '.join(self.problem_corners)
            )

        self.__values, self.__indices = None, None
        if self:
            # Repeated indices correspond to corners, which are already required
            # to be consistent, so we can remove them.
            joint_idx = np.hstack([bc.indices for bc in self if bc is not None])
            joint_value = np.hstack([bc.value for bc in self if bc is not None])
            _, unique_idx = np.unique(joint_idx, return_index=True)
            #
            self.__values = joint_value[unique_idx]
            self.__indices = joint_idx[unique_idx]


    @property
    def are_consistent(self):
        return len(self.problem_corners) == 0


    @property
    def values(self) -> NDArray:
        return self.__values


    @property
    def indices(self) -> NDArray:
        return self.__indices


    def __bool__(self):
        return reduce(
            lambda has_bcs, bc: has_bcs or bc is not None,
            self,
            False
        )


    def __get_problem_corners(
            self,
            rtol:float = 1.e-5
        ) -> List[str]:

        # NOTE:
        #   - bcs are the boundary conditions that overlap
        #   - idx are the indices where boundary conditions overlap
        #   - key describes the location where overlap occurs
        bcs = [(self.left, self.top), (self.right, self.top), (self.left, self.bot), (self.right, self.bot)]
        idx = [(-1, 0), (-1, -1), (0, 0), (0, -1)]
        key = ["top-left", "top-right", "bot-left", "bot-right"]

        problem_corners = []
        for b, i, k in zip(bcs, idx, key):
            # NOTE:
            #   If either boundary condition is None, then 
            #   a conflict is impossible. Otherise, check
            #   that the overlapping values agree to within rtol
            bcs_are_consistent = (
                    b[0] is None
                or  b[1] is None
                or  np.isclose(b[0].value[i[0]], b[1].value[i[1]], rtol=rtol)
            )
            
            if not bcs_are_consistent:
                problem_corners.append(k)
        #
        return problem_corners


    def __init_boundary_condition(
            self,
            boundary_condition:BoundaryCondition,
            x_idx:NDArray,
            y_idx:NDArray
        ):

        if boundary_condition is not None:
            boundary_condition.indices = np.ravel_multi_index(
                (
                    x_idx.astype(int),
                    y_idx.astype(int)
                ),
                self.field_shape
            )

        return boundary_condition


    def __iter__(self):
        return iter([self.top, self.bot, self.left, self.right])
