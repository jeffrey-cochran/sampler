import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
    [1] An explicit link between Gaussian fields and
        Gaussian Markov random fields: the stochastic
        partial differential equation approach
"""

np.random.seed(seed=10)
modified_bessel_second_kind = sp.special.kv

# These are the values assumed in [1]
VARIANCE = (4.*np.pi)**(-1.)
NU = 1.

def matern_precision(points, length_scale):
    """Computes matern precision matrix for nu==1 on evenly spaced square grid"""
    NN = points.shape[0]
    N = int(np.sqrt(NN))

    # Guarantee it's a square
    assert N**2 == NN

    scaling_param = length_scale**(-2.)
    M0 = scaling_param * sp.sparse.eye(NN)

    L2 = sp.sparse.diags([1., -2., 1.], [-1, 0, 1], shape=(N,N))
    M1 = sp.sparse.kronsum(L2, L2)

    return M0 - M1

# def matern_covariance(points, length_scale):
#     """Computes matern covariance matrix for nu==1"""

#     coeff = VARIANCE / sp.special.gamma(NU)

#     scaled_distance_mat = (
#             (np.sqrt(2.) / length_scale)
#         *   distance_matrix(points)
#     )

#     M1 = np.power(scaled_distance_mat, NU)

#     NN = points.shape[0]
#     diag_mask = 
#     M2a = sp.special.kv(1, scaled_distance_mat)

#     return (
#             coeff
#         *   np.multiply(
#                 np.power(scaled_distance_mat, NU),
#                 sp.special.kv(1, scaled_distance_mat)
#             )
#     )

# def distance_matrix(points):
#     N = points.shape[0]
#     xx = np.broadcast_to(points, (N,N,2))
#     return np.linalg.norm(
#         (
#                 xx
#             -   np.transpose(
#                     xx,
#                     (1,0,2)
#                 )
#         )
#         ,
#         axis=2
#     )


# GP random field for f and k
# def normalized_gp_samples(NN, n_samples, length_scale):

#     # L = sp.sparse.diags([1,1,-4,1,1], [-NN,-1,0,1,NN], shape=(NN**2, NN**2))
#     # Gamma_inv = length_scale**(-2.)*sp.sparse.eye(NN**2) - Ljjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj988888888888888888888888888888888888888888888888888888888888884s



#     K = RBF(X, length_scale) + 1e-10*np.eye(NN**2)

#     L = np.linalg.cholesky(K)
#     gp_samples = L@ np.random.randn(NN**2, n_samples)
    
#     return gp_samples

DIM = 300
LEN_SCALE = 50
NUM_SAMPLES = 1
BOUNDARY_VALUE = 100.

X = np.zeros([DIM**2, 2])

xv, yv = np.meshgrid(
    np.linspace(0, 1, DIM),
    np.linspace(0, 1, DIM)
)

LEFT_HAND_SIDE = np.ravel_multi_index(
    (
        np.arange(DIM),
        np.zeros((DIM,), dtype="int32")
    ),
    (DIM, DIM)
)
REMAINDER = np.setdiff1d(
    np.arange(DIM**2),
    LEFT_HAND_SIDE
)

X[:,0] = xv.flatten()
X[:,1] = yv.flatten()

p_mat = matern_precision(
    X,
    LEN_SCALE
)

cols = p_mat[:, LEFT_HAND_SIDE][REMAINDER, :]
beta = BOUNDARY_VALUE * np.ones((DIM,1))
product = cols@beta

samples = np.full((DIM**2, NUM_SAMPLES), BOUNDARY_VALUE)

LHS = p_mat[:, REMAINDER][REMAINDER, :]
RHS = np.random.randn(REMAINDER.size, NUM_SAMPLES) - product
temp = sp.sparse.linalg.spsolve(LHS, RHS)
samples[REMAINDER, :] = np.expand_dims(temp, 1)

plt.imsave(
    "example.png",
    samples.reshape((DIM,DIM)),
    cmap=cm.RdYlGn,
    vmin=-1.*BOUNDARY_VALUE,
    vmax=BOUNDARY_VALUE
)
# fig, ax = plt.subplots()
# im = ax.imshow(samples.reshape((DIM,DIM)), cmap=cm.RdYlGn)
# plt.close(im)
# print(type(im))
# plt.close(im)
# cbar = fig.colorbar(im)
# fig.savefig("example.png")
# plt.close(fig)
