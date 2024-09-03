
# 3rd Party
import numpy as np
import scipy as sp
from hypothesis import given, settings

# Local
from sampler.utils.test import (
    is_positive_definite,
    positive_definite_matrix,
    converge_to_same_distribution
)
from sampler.boundary_conditions import (
    UnitSquareBoundaryConditions,
    DirichletBC
)
from sampler.utils.type_aliases import (
    NDArray,
    SparseMatrix
)
from sampler.samplers import(
    GMRFSampler,
    GRFSampler
)


@settings(max_examples=10, deadline=None)
@given(positive_definite_matrix)
def test_grf_gmrf_consistency(precision:SparseMatrix):

    assert is_positive_definite(precision.toarray())

    num_nodes = precision.shape[0]
    average = np.random.rand(num_nodes)*10. - 5.
    covariance = sp.sparse.linalg.inv(precision).toarray()

    one_dim = int(np.sqrt(num_nodes))
    top_values = (np.random.rand(one_dim)*10. - 5.)

    bc_gmrf = UnitSquareBoundaryConditions(
        top = DirichletBC(value=top_values),
        bot = None,
        left = None,
        right = None,
        x_dim=one_dim,
        y_dim=one_dim
    )
    gmrf = GMRFSampler(
        average=average,
        prec_mat=precision,
        boundary_conditions=bc_gmrf,
        use_cuthill_mckee=False
    )


    bc_grf = UnitSquareBoundaryConditions(
        top = DirichletBC(value=top_values),
        bot = None,
        left = None,
        right = None,
        x_dim=one_dim,
        y_dim=one_dim
    )
    grf = GRFSampler(
        average=average,
        cov_mat=covariance,
        boundary_conditions=bc_grf
    )

    # First confirm that the computed means are the same,
    # before checking that the sample means and sample cov
    # agree.
    assert np.allclose(gmrf.mu_b, grf.mu_b)
    assert np.allclose(gmrf.mu_a, grf.mu_a)
    assert np.allclose(gmrf.mu_cond, grf.mu_cond)
    #
    assert converge_to_same_distribution(gmrf, grf)


@settings(max_examples=10, deadline=None)
@given(positive_definite_matrix)
def test_gmrf_permutation_consistency(precision:SparseMatrix):

    assert is_positive_definite(precision.toarray())

    num_nodes = precision.shape[0]
    average = np.random.rand(num_nodes)*10. - 5.

    one_dim = int(np.sqrt(num_nodes))
    left_values = (np.random.rand(one_dim)*10. - 5.)

    bc_gmrf_1 = UnitSquareBoundaryConditions(
        top = None,
        bot = None,
        left = DirichletBC(value=left_values),
        right = None,
        x_dim=one_dim,
        y_dim=one_dim
    )
    gmrf_1 = GMRFSampler(
        average=average,
        prec_mat=precision,
        boundary_conditions=bc_gmrf_1,
        use_cuthill_mckee=False
    )


    bc_gmrf_2 = UnitSquareBoundaryConditions(
        top = None,
        bot = None,
        left = DirichletBC(value=left_values),
        right = None,
        x_dim=one_dim,
        y_dim=one_dim
    )
    gmrf_2 = GMRFSampler(
        average=average,
        prec_mat=precision,
        boundary_conditions=bc_gmrf_2,
        use_cuthill_mckee=True
    )

    # First confirm that the computed means are the same,
    # before checking that the sample means and sample cov
    # agree.
    assert np.allclose(gmrf_1.mu_b, gmrf_2.mu_b)
    assert np.allclose(gmrf_1.mu_a, gmrf_2.mu_a)
    assert np.allclose(gmrf_1.mu_cond, gmrf_2.mu_cond)
    #
    assert converge_to_same_distribution(gmrf_1, gmrf_2)