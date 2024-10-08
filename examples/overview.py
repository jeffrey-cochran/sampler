# Standard
# N/A

# 3rd Party
import numpy as np
import scipy as sp
import matplotlib.tri as tri
import matplotlib.pyplot as plt

# Local
from sampler.samplers.unit_square import UnitSquareSampler
from sampler.boundary_conditions.base import NeumannBC, DirichletBC
from sampler.utils.kernels import whittle_matern_precision, whittle_matern_covariance

# NOTE:
#   - The average is expected to be a 2D array of shape (num_bases_x, num_bases_y)
#   - You can provide either a precision matrix (`prec_mat=...`)
#     or a covariance matrix (`cov_mat=...`), but not both
#   - The `whittle_matern_covariance` function assumes a smoothness parameter of 2.5
#     (this isn't arbitrary, but a special case for which the kernel is simple)
num_bases_x = 7
num_bases_y = 7
average = np.zeros((num_bases_x, num_bases_y))
cov = whittle_matern_covariance(num_bases_x, 0.3)

# NOTE:
#   - The precision matrix is expected to be a 2D array of shape
#     (num_bases_x*num_bases_y, num_bases_x*num_bases_y)
#   - The Whittle-Matern precision matrix is only valid for uniform grids.
#   - The `whittle_matern_precision` function assumes a smoothness parameter of 2.0
#     (this isn't arbitrary, but a special case for which the precision can be
#     computed in closed form and the resulting precision matrix is very sparse)
# prec = whittle_matern_precision(num_bases_x, 1.)

# Example boundary condition functions
# NOTE:
#   Since the boundary conditions are applied to the
#   edges of the unit-square, they only take 1D args
#   (i.e. x or y) corresponding to which ever variable
#   is free on that boundary.
def sin_bc(x):
    return np.sin(2*np.pi*x)

def constant(in_constant=0.):
    def f(x):
        return in_constant
    return f

def linear(in_slope=1., in_intercept=0.):
    def f(x):
        return in_slope*x + in_intercept
    return f

def zero(x):
    return 0.

# NOTE:
#   Neumann conditions are currently unsupported,
#   but they should be up and running soon. I'll
#   add in linear constraints at the same time.
not_supported = NeumannBC(func=sin_bc, id="neumann")

# NOTE:
#   If you want you reuse the same boundary condition,
#   you'll either need multiple instances; otherwise,
#   the computed indices will be overwritten each time
#   the referenced BC is evaluated.
tmp_sampler = UnitSquareSampler(
    average=average,
    cov_mat=cov,
    #prec_mat=prec,
    poly_order=4,
    bc_top=DirichletBC(func=constant(-1.)),
    bc_bot=DirichletBC(func=constant(1.)),
    bc_left=None,
    bc_right=None,
)

# To sample, just call `MySampler.sample(num_samples)`
# NOTE:
#   The matrix factorization is triggered by the first
#   call to `sample`. Subsequent calls will not recompute
#   the factorization.
ten_thousand_samples = tmp_sampler.sample(10000)

# To confirm your samples look correct,
# you can call `.visualize_sample()` to plot a random
# sample, or `.visualize_sample(some_sample)`
# to visualize a specific example. You can specify 
# `degree=bigger_int` to increase refinement. The default
# degree is 9.
# NOTE:
#   For the purpose of visualization, the field values are
#   normalized to fall within the range [-1, 1].
tmp_sampler.visualize_sample(title="Random Sample")
tmp_sampler.visualize_sample(ten_thousand_samples[777], title="Sample 777")
tmp_sampler.visualize_sample(ten_thousand_samples[777], degree=30, title="Sample 777 (Refined)")

# Underneath, the sampler is using `nutils` for all of its
# b-spline computation, and it stores this information if
# you want to use it. For example, here's the commented
# `visualize_sample` code:
degree = 9
sample = ten_thousand_samples[777]
title = "Commented `visualize_sample` Code"

# Create nutils sampler for visualization
viz_sampler = tmp_sampler.topo.sample('bezier', degree)

# Create nutils function for visualization using
# the tmp_sampler basis with the sampled coefficients
tmp_sampler.namespace.f = np.dot(tmp_sampler.namespace.basis, sample)

# Evaluate the function at the nutils sampler points
x, f = viz_sampler.eval(['xy_i', 'f'] @ tmp_sampler.namespace)

# Normalize function values for visualization
f /= (1.1*np.abs(f).max())

# Create the triangulation for visualization
triangulation = tri.Triangulation(x[:,0], x[:,1], viz_sampler.tri)

# Create figure and subplots
fig = plt.figure()
ax1 = fig.add_subplot(111)

# Create level sets for visualization
levels = np.arange(-1., 1., 0.05)

# Plot the contours
ax1.set_title(title, fontsize=40)
ax1.triplot(triangulation, lw=0.5, color='white')
contour = ax1.tricontourf(triangulation, f.flatten(), levels=levels, cmap='bwr')

# Add a colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(contour, cax=cbar_ax)

# Resize to full-screen
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

plt.show()
