
import sampler.utils.bsplines.bspline as bspline
import sampler.utils.bsplines.tensor_basis as tensor_basis
import sampler.utils.bsplines.tensor_grid as tensor_grid
import sampler.utils.bsplines.functions as functions
import torch

x_basis = bspline.BSpline.uniform(
    lims=(0,1),
    n_segments=10,
    degree=3,
    dtype = torch.float64
)

xy_basis = tensor_basis.TensorBasis(x_basis, x_basis)

xy_grid = tensor_grid.TensorGrid(
    xs = torch.linspace(0,1,100)
)

weights = torch.randn(100,1, dtype=torch.float64)

# TODO: confirm the orderign of the weights
f = functions.Functions(xy_basis, weights)

