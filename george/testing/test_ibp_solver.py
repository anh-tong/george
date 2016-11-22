# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_basic_solver"]

import numpy as np

from .. import kernels
from ..ibphodlr import IbpHODLRSolver

def test_ibp_holr_solver(**kwargs):

    N = 100
    se1 =  1e-4 * kernels.ExpSquaredKernel(1.0)

    x = np.random.randn(N, 1)
    K = se1.value(x)
    yerr = 1e-3 * np.ones(N)
    solver = IbpHODLRSolver(N,N)
    solver.compute(K, yerr)

    # Build the matrix.
    K[np.diag_indices_from(K)] += yerr ** 2

    # Check the determinant.
    sgn, lndet = np.linalg.slogdet(K)
    assert sgn == 1.0, "Invalid determinant"
    print("numpy:  {}".format(lndet))
    print("solver: {}".format(solver.log_determinant))
    assert np.allclose(solver.log_determinant, lndet), "Incorrect determinant"

    # Check the inverse.
    assert np.allclose(solver.apply_inverse(K), np.eye(N)), "Incorrect inverse"