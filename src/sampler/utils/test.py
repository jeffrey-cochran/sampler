
import numpy as np
import scipy as sp
from tabulate import tabulate
from hypothesis.strategies import integers
from sampler.samplers.base import __Sampler__
from sampler.utils.type_aliases import (
    NDArray,
    SparseMatrix
)


def converges_to_distribution(
        sampler:__Sampler__,
        average:NDArray,
        covariance:NDArray,
        max_iter=8,
        rtol:float=5e-2,
        atol:float=1e-2
) -> None:
    """Check if sampler converges to specified distribution to within rtol.
    
    Arguments
    ---------
    sampler (__Sampler__)
        A sampler object
    average (NDArray)
        Average of the distribution
    covariance (NDArray)
        Covariance of the distribution
    max_iter (int)
        Maximum number of iterations; each iteration increases
        the sample size by a factor of 100
    rtol (float)
        Relative tolerance
    """
    num_samples = np.power(10, np.arange(1, max_iter)).astype(int)
    converges = False
    avg_residual = []
    cov_residual = []
    for n in num_samples:
        sample = sampler(n)
        sample_covariance = np.cov(sample.T)
        sample_average = np.average(sample, axis=0)

        if (
                np.allclose(sample_average, average, rtol=rtol, atol=atol)
            and np.allclose(sample_covariance, covariance, rtol=rtol, atol=atol)
        ):
            converges = True
            break

        avg_residual.append(
            (np.abs(sample_average - average) - rtol*np.abs(average)).max()
        )
        cov_residual.append(
            (np.abs(sample_covariance - covariance) - rtol*np.abs(covariance)).max()
        )

    if not converges:
        failure_mode = "`cov`" if np.min(cov_residual) > np.min(avg_residual) else "`mu`"
        print(f"The max {failure_mode} relative residual exceeds {atol} after {n} samples")
        print(
            tabulate(
                {
                    'Sample Size': num_samples,
                    'Mu Max Rel. Residual': avg_residual,
                    'Cov. Max Rel. Residual': cov_residual
                },
                headers='keys',
                tablefmt='pretty'
            )
        )

    return converges


def converge_to_same_distribution(
        sampler_1:__Sampler__,
        sampler_2:__Sampler__,
        max_iter=8,
        rtol:float=5e-2,
        atol:float=1e-2
) -> None:
    """Check if samplers converges to same distribution to within rtol.
    
    Arguments
    ---------
    sampler_1 (__Sampler__)
        A sampler object
    sampler_2 (__Sampler__)
        A sampler object
    max_iter (int)
        Maximum number of iterations; each iteration increases
        the sample size by a factor of 10
    rtol (float)
        Relative tolerance
    """
    num_samples = np.power(10, np.arange(1, max_iter)).astype(int)
    converges = False
    avg_residual = []
    cov_residual = []
    for n in num_samples:

        sample_1 = sampler_1(n)
        sample_2 = sampler_2(n)

        sample_1_covariance = np.cov(sample_1.T)
        sample_1_average = np.average(sample_1, axis=0)

        sample_2_covariance = np.cov(sample_2.T)
        sample_2_average = np.average(sample_2, axis=0)

        if (
                np.allclose(sample_1_average, sample_2_average, rtol=rtol, atol=atol)
            and np.allclose(sample_1_covariance, sample_2_covariance, rtol=rtol, atol=atol)
        ):
            converges = True
            break

        avg_residual.append(
            (np.abs(sample_1_average - sample_2_average) - rtol*np.abs(sample_1_average)).max()
        )
        cov_residual.append(
            (np.abs(sample_1_covariance - sample_2_covariance) - rtol*np.abs(sample_1_covariance)).max()
        )

    if not converges:
        failure_mode = "`cov`" if np.min(cov_residual) > np.min(avg_residual) else "`mu`"
        print(f"The max {failure_mode} relative residual exceeds {atol} after {n} samples")
        print(
            tabulate(
                {
                    'Sample Size': num_samples,
                    'Mu Max Rel. Residual': avg_residual,
                    'Cov. Max Rel. Residual': cov_residual
                },
                headers='keys',
                tablefmt='pretty'
            )
        )

    return converges


def is_positive_definite(matrix:NDArray) -> bool:
    """Check if a matrix is positive definite.
    
    Arguments
    ---------
    matrix (NDArray)
        A square matrix

    Returns
    -------
    is_psd (bool)
        True if the matrix is positive definite
    """
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


# Define a function for filtering out non-positive definite matrices
def get_positive_definite_matrix(size:int) -> SparseMatrix:
    """Check if a matrix is positive definite.
    
    Arguments
    ---------
    size (int)
        Size of each dim of the matrix

    Returns
    -------
    proposed_psd (SparseMatrix)
        A sparse positive definite matrix
    """
    proposed_psd:NDArray
    while True:
        tmp = sp.sparse.random(size, size, density=0.3, format='csr')
        proposed_psd = tmp @ tmp.T
        try:
            np.linalg.cholesky(proposed_psd.toarray())
            break
        except np.linalg.LinAlgError:
            continue

    return sp.sparse.csc_array(proposed_psd)


# Define a strategy for sampling positive definite matrices
positive_definite_matrix = integers(min_value=3, max_value=5) \
    .map(lambda n: get_positive_definite_matrix(n**2)) \
    .filter(lambda A: is_positive_definite(A.toarray()))

