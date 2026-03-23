"""
Tests for core.oddity_task — Monte Carlo oddity-task simulation.

test_identity_noise_gives_circular_threshold
--------------------------------------------
With isotropic (scaled-identity) noise at both reference and comparison
locations, the oddity-task MC estimator should recover a threshold contour
that is approximately circular.  We verify two things:

    1. The code runs without error.
    2. The recovered threshold radii across N_THETA directions have a
       coefficient of variation < 25 % (the threshold is roughly circular).

Geometry
--------
We use ndims_cov=2, ndims_extra=3 (the paper's 2-D isoluminant-plane setup).

    U  = scale * eye(ndims_cov, ndims_extra)
    Σ  = U @ U.T + diag_term * I  =  (scale² + diag_term) * I   [isotropic]

With scale=0.05 and diag_term=1e-4 the threshold falls near 0.14 in the
search window [0.001, 0.5], confirmed by a quick sweep in probe_threshold.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from analysis.ellipses_tools import ellParamsQ_to_covMat, fit_2d_isothreshold_contour
from core import oddity_task

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
NDIMS_COV = 2
NDIMS_EXTRA = 3          # ndims_cov + extra_dims (1) as used in the paper
NOISE_SCALE = 0.05       # U = scale * eye  →  Σ ≈ 0.0025 * I
DIAG_TERM = 1e-4
BANDWIDTH = 0.005        # smoothing parameter for approx_cdf
TARGET_PC = 0.667        # 66.7 % correct = threshold

N_THETA = 8              # directions sampled around the unit circle
NGRID = 50               # vector-length grid points per direction
MC_SAMPLES = 500         # Monte Carlo samples per (direction, length) pair
VEC_LENGTH_BOUNDS = (0.001, 0.50)  # search window; threshold ≈ 0.14

CIRCULARITY_TOL = 0.25   # max allowed coefficient of variation across radii


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_identity_U(batch: int) -> np.ndarray:
    """Return (batch, NDIMS_COV, NDIMS_EXTRA) array of scaled-identity U."""
    U = NOISE_SCALE * np.eye(NDIMS_COV, NDIMS_EXTRA)
    return np.tile(U[None], (batch, 1, 1))


def _estimate_threshold_radii(w_ref: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Estimate threshold radius in each of N_THETA directions around w_ref.

    Returns
    -------
    radii : (N_THETA,) array of threshold distances
    """
    thetas = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    directions = np.column_stack([np.cos(thetas), np.sin(thetas)])
    vec_lengths = np.linspace(*VEC_LENGTH_BOUNDS, NGRID)

    radii = np.zeros(N_THETA)
    Uref = _build_identity_U(NGRID)

    for i, d in enumerate(directions):
        w_comps = w_ref[None, :] + vec_lengths[:, None] * d[None, :]
        w_ref_rep = np.tile(w_ref, (NGRID, 1))

        keys_i = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(seed), i), NGRID
        )

        pC = np.array(
            oddity_task.oddity_prediction(
                (
                    jnp.array(w_ref_rep),
                    jnp.array(w_comps),
                    jnp.array(Uref),
                    jnp.array(Uref),  # same noise at reference and comparison
                ),
                keys_i,
                MC_SAMPLES,
                BANDWIDTH,
                DIAG_TERM,
                oddity_task.simulate_oddity,
            )
        )
        radii[i] = vec_lengths[np.argmin(np.abs(pC - TARGET_PC))]

    return radii


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_identity_noise_gives_circular_threshold():
    """
    Isotropic noise should produce a circular threshold contour.

    Checks:
      - Runs without error.
      - All N_THETA threshold radii are positive.
      - Coefficient of variation across radii < CIRCULARITY_TOL.
    """
    w_ref = np.array([0.0, 0.0])
    radii = _estimate_threshold_radii(w_ref)

    assert np.all(radii > 0), f"Some radii are non-positive: {radii}"

    cv = radii.std() / radii.mean()
    assert cv < CIRCULARITY_TOL, (
        f"Threshold contour is not sufficiently circular (CV={cv:.3f} >= {CIRCULARITY_TOL}).\n"
        f"Radii: {np.round(radii, 4)}"
    )


def test_identity_noise_threshold_covariance_is_isotropic():
    """
    Fit an ellipse to the threshold contour points and verify the resulting
    covariance matrix is approximately isotropic (ratio of eigenvalues ≈ 1).
    """
    w_ref = np.array([0.0, 0.0])
    radii = _estimate_threshold_radii(w_ref)

    thetas = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    directions = np.column_stack([np.cos(thetas), np.sin(thetas)])
    w_comp_est = w_ref[:, None] + directions.T * radii[None, :]  # (2, N_THETA)

    _, _, params_ell, _ = fit_2d_isothreshold_contour(
        w_ref, w_comp_est, nTheta=200, flag_force_centered_ref=True
    )
    Sigma_thres = np.array(ellParamsQ_to_covMat(*params_ell[2:]))

    eigvals = np.linalg.eigvalsh(Sigma_thres)
    assert eigvals.min() > 0, "Threshold covariance matrix is not positive definite."

    ratio = eigvals.max() / eigvals.min()
    assert ratio < 2.0, (
        f"Threshold covariance is not sufficiently isotropic "
        f"(eigenvalue ratio={ratio:.3f}).\nSigma_thres:\n{Sigma_thres}"
    )
