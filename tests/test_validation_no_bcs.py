
# 3rd Party
import numpy as np
import scipy as sp
from hypothesis import given, settings
from hypothesis.strategies import integers

# Local
from sampler.utils.test import (
    is_positive_definite,
    positive_definite_matrix,
    converges_to_distribution
)
from sampler.utils.type_aliases import (
    NDArray,
    SparseMatrix
)
from sampler.samplers import(
    GMRFSampler,
    GRFSampler
)


@given(positive_definite_matrix)
def test_positive_definiteness(psd:SparseMatrix):
    assert is_positive_definite(psd.toarray())


@settings(max_examples=10, deadline=None)
@given(positive_definite_matrix)
def test_grf_validation_no_bcs(precision:SparseMatrix):

    assert is_positive_definite(precision.toarray())

    num_nodes = precision.shape[0]
    average = np.random.rand(num_nodes)*10. - 5.
    covariance = sp.sparse.linalg.inv(precision).toarray()

    grf = GRFSampler(
        average=average,
        cov_mat=covariance,
        boundary_conditions=None
    )

    assert converges_to_distribution(
        sampler=grf,
        average=average,
        covariance=covariance
    )


@settings(max_examples=10, deadline=None)
@given(positive_definite_matrix)
def test_gmrf_validation_no_permutation_no_bcs(precision:SparseMatrix):

    assert is_positive_definite(precision.toarray())

    num_nodes = precision.shape[0]
    average = np.random.rand(num_nodes)*10. - 5.
    covariance = sp.sparse.linalg.inv(precision).toarray()

    gmrf = GMRFSampler(
        average=average,
        prec_mat=precision,
        boundary_conditions=None,
        use_cuthill_mckee=False
    )

    assert converges_to_distribution(
        sampler=gmrf,
        average=average,
        covariance=covariance
    )


@settings(max_examples=10, deadline=None)
@given(positive_definite_matrix)
def test_gmrf_validation_cuthill_mckee_no_bcs(precision:SparseMatrix):

    assert is_positive_definite(precision.toarray())

    num_nodes = precision.shape[0]
    average = np.random.rand(num_nodes)*10. - 5.
    covariance = sp.sparse.linalg.inv(precision).toarray()

    gmrf = GMRFSampler(
        average=average,
        prec_mat=precision,
        boundary_conditions=None,
        use_cuthill_mckee=True
    )

    assert converges_to_distribution(
        sampler=gmrf,
        average=average,
        covariance=covariance
    )
