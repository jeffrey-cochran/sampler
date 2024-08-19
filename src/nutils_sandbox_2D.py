from nutils import function, export, mesh, solver, testing, cli
from nutils.expression_v2 import Namespace
import numpy as np
import scipy as sp
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.cm as cm


DIM = 5
NUM_SAMPLES = 1

np.random.seed(seed=10)

ns = function.Namespace()

topo, geom = mesh.rectilinear([np.linspace(0, 1, DIM), np.linspace(0, 1, DIM)])

ns.x = geom
ns.basis = topo.basis('spline', degree=3)
bezier = topo.sample('bezier', 30)

# ================================================================================
n_coeffs = DIM+2

# Generate coefficients grid
# ==========================
X = np.zeros([n_coeffs**2, 2])
#
xv, yv = np.meshgrid(
    np.linspace(0, 1, n_coeffs),
    np.linspace(0, 1, n_coeffs)
)
#
X[:,0] = xv.flatten()
X[:,1] = yv.flatten()

# Compute indices of boundary coefficients
# ========================================
indices = np.arange(n_coeffs)
indices_sq = np.arange(n_coeffs**2)
ones = np.ones((n_coeffs,), dtype="int32")
zeros = np.zeros((n_coeffs,), dtype="int32")
#
BOT_BC_NEUMANN_1 = np.ravel_multi_index(
    (
        indices,
        zeros
    ),
    (n_coeffs, n_coeffs)
)
BOT_BC_NEUMANN_2 = np.ravel_multi_index(
    (
        indices,
        ones
    ),
    (n_coeffs, n_coeffs)
)
BOT_BC_NEUMANN_3 = np.ravel_multi_index(
    (
        indices,
        ones*2
    ),
    (n_coeffs, n_coeffs)
)
BOT_BC_NEUMANN_4 = np.ravel_multi_index(
    (
        indices,
        ones*3
    ),
    (n_coeffs, n_coeffs)
)
BOT_BC_NEUMANN = np.concatenate((BOT_BC_NEUMANN_1, BOT_BC_NEUMANN_2), axis=0)
#
TOP_BC_DIRICHLET = np.ravel_multi_index(
    (
        indices[1:-1],
        ones[1:-1]*(n_coeffs-1)
    ),
    (n_coeffs, n_coeffs)
)
#
#
# TOP_BC_DIRICHLET = np.ravel_multi_index(
#     (
#         zeros,
#         indices
#     ),
#     (n_coeffs, n_coeffs)
# )
# BOT_BC_DIRICHLET = np.ravel_multi_index(
#     (
#         ones*(n_coeffs-1),
#         indices
#     ),
#     (n_coeffs, n_coeffs)
# )
#
# REMAINDER = np.setdiff1d(
#     np.arange(n_coeffs**2),
#     np.concatenate(
#         (
#             BOT_BC_NEUMANN_1,
#             BOT_BC_NEUMANN_2,
#             RIGHT_BC_DIRICHLET,
#             TOP_BC_DIRICHLET,
#             BOT_BC_DIRICHLET
#         ),
#         axis=0
#     )
# )
REMAINDER = np.setdiff1d(
    indices_sq,
    BOT_BC_NEUMANN
)


m = BOT_BC_NEUMANN_1.size
n = n_coeffs**2
data = np.concatenate((np.broadcast_to(1, m), np.broadcast_to(-1,m)), axis=0)
row_idx = np.broadcast_to(np.arange(m), (2,m)).flatten()
col_idx = BOT_BC_NEUMANN

A = sp.sparse.csc_matrix((data, (row_idx, col_idx)), shape=(m,n))
V = A.transpose()
W = A.dot(V)
U = sp.sparse.linalg.spsolve(W, A)

x = np.random.randn(n_coeffs**2, NUM_SAMPLES)
#
c = A.dot(x)
#
x_star = (x - U.transpose().dot(c)).flatten()


# # ================================================================================

# print(x_star[BOT_BC_NEUMANN_1])
# print(x_star[BOT_BC_NEUMANN_2])

# blank = np.zeros((x_star.shape))
# blank[BOT_BC_NEUMANN_1] = 1
# blank[BOT_BC_NEUMANN_2] = 1
ns.f = np.dot(ns.basis, x_star)
ns.fgrad = 'f_,1'
x, f, fgrad = bezier.eval(['x_i', 'f', 'fgrad'] @ ns)

f /= (1.1*np.abs(f).max())
fgrad /= (1.1*np.abs(fgrad).max())

triangulation = tri.Triangulation(x[:,0], x[:,1], bezier.tri)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

levels = np.arange(-1., 1., 0.05)

ax1.set_title('dF/dy Value', fontsize=40)
ax1.triplot(triangulation, lw=0.5, color='white')
contour = ax1.tricontourf(triangulation, fgrad.flatten(), levels=levels, cmap='bwr')

ax2.set_title('Function Value', fontsize=40)
ax2.triplot(triangulation, lw=0.5, color='white')
contour = ax2.tricontourf(triangulation, f.flatten(), levels=levels, cmap='bwr')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(contour, cax=cbar_ax)

# xvals, vals, grad_vals = bezier.eval([ns.x, ns.f, ns.fgrad])

# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)

# ax1.set_title('Function Value')
# ax1.plot(xvals, vals)

# ax2.set_title('Gradient Value')
# ax2.plot(xvals, grad_vals)

plt.show()