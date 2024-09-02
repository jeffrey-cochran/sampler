
# Standard
from typing import Tuple

# 3rd Party
import scipy as sp
import numpy as np

# Local
from sampler.utils.type_aliases import (
    SparseMatrix,
    NDArray
)

def chol(A:SparseMatrix, use_cuthill_mckee:bool=True) -> Tuple[SparseMatrix, NDArray]:
    """Cholesky decomposition of a sparse matrix A.

    Computes a sparse Cholesky decomposition PAP^T = LL^T,
    where P is the reverse cuthill mckee permutation.

    Arguments
    =========
    A (SparseMatrix)
        Sparse, symmetric positive definite matrix.

    Returns
    =======
    L (SparseMatrix)
        Lower triangular matrix.
    P (SparseMatrix)
        Permutation matrix.
    """

    # NOTE:
    #   The Reverse Cuthill-McKee permutation is used to reduce the
    #   profile (not bandwidth) of the matrix. This decreases the
    #   density of the resulting Cholesky factor.
    P = sp.sparse.eye(A.shape[0])
    if use_cuthill_mckee:
        p = sp.sparse.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
        P = sp.sparse.csc_array(
            (
                np.ones(A.shape[0]),        # data
                (
                    np.arange(A.shape[0]),  # row idx
                    p                       # col idx
                )
            ),
            shape=A.shape
        )

    return sp.sparse.csc_array(np.linalg.cholesky((P@A@P.T).toarray())), P