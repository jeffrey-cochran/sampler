import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(seed=10)
modified_bessel_second_kind = sp.special.kv
NU = 1.

def matern_precision(points, length_scale, variance_scale=1.):
      """Computes matern precision matrix for nu==1 on evenly spaced square grid"""
      NN = points.shape[0]
      N = np.sqrt(NN)

      assert int(N)**2 == NN
      return

def matern_covariance(points, length_scale, variance_scale=1.):
    """Computes matern covariance matrix for nu==1"""

    coeff = variance_scale**2. * 2.**(1-NU) / sp.special.gamma(NU)

    scaled_distance_mat = (
            (sp.sqrt(2.*NU) / length_scale)
        *   distance_matrix(points)
    )

    return (
            coeff
        *   np.multiply(
                np.power(scaled_distance_mat, NU),
                sp.special.kv(scaled_distance_mat)
            )
    )

def distance_matrix(points):
    N = points.shape[0]
    xx = np.broadcast_to(points, (N,N,2))
    return np.linalg.norm(
        (
                xx
            -   np.transpose(
                    xx,
                    (1,0,2)
                )
        )
        ,
        axis=2
    )
def RBF(x, length_scale):
        r = (dist_mat/length_scale)**2        
        return np.exp(-0.5 * r)


# GP random field for f and k
def normalized_gp_samples(NN, n_samples, length_scale):

    # L = sp.sparse.diags([1,1,-4,1,1], [-NN,-1,0,1,NN], shape=(NN**2, NN**2))
    # Gamma_inv = length_scale**(-2.)*sp.sparse.eye(NN**2) - Ljjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj988888888888888888888888888888888888888888888888888888888888884s

    X = np.zeros([NN**2, 2])

    xv, yv = np.meshgrid(np.linspace(0, 1, NN), np.linspace(0, 1, NN))
    X[:,0] = xv.flatten()
    X[:,1] = yv.flatten()

    K = RBF(X, length_scale) + 1e-10*np.eye(NN**2)

    L = np.linalg.cholesky(K)
    gp_samples = L@ np.random.randn(NN**2, n_samples)
    
    return gp_samples
DIM = 30
a = normalized_gp_samples(DIM, 1, 0.1).reshape((DIM,DIM))

plt.imshow(
    a, 
    interpolation='bilinear', 
    cmap=cm.RdYlGn,
    origin='lower', 
    extent=[-3, 3, -3, 3],
    vmax=abs(a).max(),
    vmin=-abs(a).max()
)
plt.show()