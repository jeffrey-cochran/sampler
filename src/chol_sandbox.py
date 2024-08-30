
import numpy as np
import scipy as sp
from utils.sparse_chol import chol


A = np.eye(100) + sp.sparse.random(100,100,density=0.02).toarray()
A = sp.sparse.csc_array(A @ A.T)

L = np.linalg.cholesky(A.toarray())
L = sp.sparse.csc_matrix(L)

L, P = chol(A)