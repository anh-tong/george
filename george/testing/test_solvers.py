# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_basic_solver"]

import numpy as np

from .. import kernels
from ..basic import BasicSolver
from ..hodlr import HODLRSolver
from ..latenthodlr import LatentHODLRSolver


def _test_solver(Solver, N=100, seed=1234):
    # Set up the solver.
    kernel = 1e-4 * kernels.ExpSquaredKernel(1.0)
    solver = Solver(kernel)

    # Sample some data.
    np.random.seed(seed)
    x = np.random.randn(N, kernel.ndim)
    yerr = 1e-3 * np.ones(N)
    solver.compute(x, yerr)

    # Build the matrix.
    K = kernel.value(x)
    K[np.diag_indices_from(K)] += yerr ** 2

    # Check the determinant.
    sgn, lndet = np.linalg.slogdet(K)
    assert sgn == 1.0, "Invalid determinant"
    assert np.allclose(solver.log_determinant, lndet), "Incorrect determinant"

    # Check the inverse.
    assert np.allclose(solver.apply_inverse(K), np.eye(N)), "Incorrect inverse"


def test_basic_solver(**kwargs):
    _test_solver(BasicSolver, **kwargs)


def test_hodlr_solver(**kwargs):
    _test_solver(HODLRSolver, **kwargs)

# def test_latent_holr_solver(**kwargs):
#     se1 = kernels.ExpSquaredKernel(-3.1)
#     se2 = kernels.ExpSquaredKernel(-0.5)
#     se3 = kernels.ExpSquaredKernel(-1.5)
#
#
#
#
#     latent_model_kernel = kernels.LatentModelKernel(kernels = [se1, se2, se3], pars = np.append(np.append(se1.pars, se2.pars), se3.pars), d = 3)
#     latent_model_kernel.set_ZTZ(ztz=np.array([ 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]))
#     x = np.random.randn(latent_model_kernel.d, latent_model_kernel.ndim)
#     print(se1.value(x))
#     print(se2.value(x))
#     inv1 = np.linalg.inv(se1.value(x))
#     print(inv1)
#     inv2 = np.linalg.inv(se2.value(x))
#     print(inv2)
#     inv3 = np.linalg.inv(se3.value(x))
#     print(inv3)
#     # se1.set_inversed(inv1, 4)
#     # se2.set_inversed(inv2, 4)
#
#     yerr = 1e-3 * np.ones(latent_model_kernel.k*latent_model_kernel.d)
#     latent_model_kernel.add_inversed(inv1)
#     latent_model_kernel.add_inversed(inv2)
#     latent_model_kernel.add_inversed(inv3)
#
#     # solver.compute(x, yerr)
#     K = latent_model_kernel.value_ij(x,latent_model_kernel.k*latent_model_kernel.d, latent_model_kernel.k, latent_model_kernel.d )
#     print(K)
#
#     solver = LatentHODLRSolver(latent_model_kernel)
#     solver.set_ZTZ(np.array([ 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]))
#     solver.add_inversed(inv1)
#     solver.add_inversed(inv2)
#     solver.add_inversed(inv3)
#     solver.compute(x, yerr)
#     lndet = np.linalg.det(K)
#     print(lndet)
#     print(solver.log_determinant)
    # assert np.allclose(solver.log_determinant, lndet), "Incorrect determinant"


