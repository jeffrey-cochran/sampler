import numpy as np
import scipy as sp

def whittle_matern_precision(num_bases_one_d, length_scale):
    """Generate a precision matrix for a Whittle-Matern covariance function on a uniform 2D grid.

    This precision matrix is valid for a uniform grid with smoothness parameter 2.0.

    Arguments
    =========
    num_bases (int)
        Number of basis functions in the x-direction and y-direction.
    length_scale (float)
        Length scale of the covariance function.

    Returns
    =======
        scipy.sparse.csc_matrix: Precision matrix for the Whittle-Matern covariance function.
    """
    total_num_bases = num_bases_one_d * num_bases_one_d

    eye = sp.sparse.eye(num_bases_one_d)
    L2 = sp.sparse.diags(
        diagonals = (
            -2*np.ones(num_bases_one_d),
            np.ones(num_bases_one_d-1),
            np.ones(num_bases_one_d-1)
        ),
        offsets = (0, -1, 1),
        shape = (num_bases_one_d, num_bases_one_d)
    )

    scaling = 1./(length_scale**2.)
    factor = (
            scaling * sp.sparse.eye(total_num_bases, format="csc")
        -   sp.sparse.kron(eye, L2)
        -   sp.sparse.kron(L2, eye)
    )

    # NOTE:
    #   If the code were structured differently, we could
    #   just pass the factor to the sampler directly--no need to
    #   compute the precision matrix and then do a Cholesky
    #   factorization. However, I wanted to make the sampler
    #   more general, so it accepts a precision matrix.
    return sp.sparse.csc_matrix(factor @ factor.T)


def whittle_matern_covariance(
        num_bases_one_d,
        length_scale,
        variance_scale=1.
    ):
    """Generate a precision matrix for a Whittle-Matern covariance function on a uniform 2D grid.

    The smoothness parameter is fixed at 2.5

    Arguments
    =========
    num_bases (int)
        Number of basis functions in the x-direction and y-direction.
    length_scale (float)
        Length scale of the covariance function.
    scaling_parameter (float)
        Scaling parameter for the covariance function.
    Returns
    =======
        nump.ndarray: Covariance matrix for the Whittle-Matern covariance function.
    """
    # NOTE:
    #   The smoothness parameter is fixed at 2.5, but this function
    #   can be generalized. I just wanted to keep things simple for now.
    #
    disc = np.linspace(0, 1, num_bases_one_d)
    X, Y = np.meshgrid(disc, disc)
    x = np.atleast_2d(X.flatten())
    y = np.atleast_2d(Y.flatten())
    dist = np.sqrt((x - x.T)**2 + (y - y.T)**2)
    scaled_dist = np.sqrt(5) * dist / length_scale

    out_cov = (
            variance_scale
        *   np.multiply(
                (
                        1.
                    +   scaled_dist
                    +   (np.power(scaled_dist, 2.)/3.)
                ),
                np.exp(-scaled_dist)
        )
    )

    return out_cov
