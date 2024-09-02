
# Standard
from typing import Tuple

# 3rd Party
import numpy as np
import scipy as sp
import pytest
from hypothesis import given
from hypothesis.strategies import floats, integers, tuples


# Local
from sampler.boundary_conditions import (
    UnitSquareBoundaryConditions,
    DirichletBC
)


@given(
    integers(min_value=4, max_value=25),
    tuples(
        floats(min_value=-10., max_value=10.),
        floats(min_value=-10., max_value=10.),
        floats(min_value=-10., max_value=10.),
        floats(min_value=-10., max_value=10.)
    )
)
def test_bc_consistency(size:int, corner_values:Tuple[float, float, float, float]):
    top_left, top_right, bot_left, bot_right = corner_values

    top_values   = (np.random.rand(size)*10. - 5.)
    bot_values   = (np.random.rand(size)*10. - 5.)
    left_values  = (np.random.rand(size)*10. - 5.)
    right_values = (np.random.rand(size)*10. - 5.)

    top_values[0]   = top_left
    left_values[-1] = top_left

    top_values[-1]   = top_right
    right_values[-1] = top_right

    bot_values[0]   = bot_left
    left_values[0]  = bot_left

    bot_values[-1]  = bot_right
    right_values[0] = bot_right

    boundary_conditions = UnitSquareBoundaryConditions(
        top = DirichletBC(value=top_values),
        bot = DirichletBC(value=bot_values),
        left = DirichletBC(value=left_values),
        right = DirichletBC(value=right_values),
        x_dim=size,
        y_dim=size
    )

    assert boundary_conditions.values.size == (4*(size-1))
    assert boundary_conditions.are_consistent


@given(integers(min_value=4, max_value=25), floats(min_value=-10., max_value=10.))
def test_bc_inconsistency(size:int, corner_value:float):

    top_values   = (np.random.rand(size)*10. - 5.)
    left_values  = (np.random.rand(size)*10. - 5.)

    top_values[0]   = corner_value
    left_values[-1] = corner_value + 1.

    with pytest.raises(RuntimeError):
        _ = UnitSquareBoundaryConditions(
            top = DirichletBC(value=top_values),
            bot = None,
            left = DirichletBC(value=left_values),
            right = None,
            x_dim=size,
            y_dim=size
        )
