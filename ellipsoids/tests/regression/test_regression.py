"""
Regression tests for ellipsoids_public.

Each test loads a committed baseline from tests/baselines/, runs the same
computation with the same inputs and frozen RNG, and checks that the new
output matches the baseline within the project tolerance:

    max over all elements of:
        |new - ref| / (|mean(ref)| + 1) < RTOL

To regenerate baselines (e.g. after a deliberate algorithmic change, or to
replace dev-branch baselines with reference-branch ground truth):

    python tests/regression/make_baselines.py

Then commit the updated .npz files.
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from analysis.ellipses_tools import (
    covMat_to_ellParamsQ,
    ellParamsQ_to_covMat,
    fit_2d_isothreshold_contour,
)
from core import oddity_task
from core.wishart_process import WishartProcessModel

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

RTOL = 1e-5

BASELINES_DIR = pathlib.Path(__file__).resolve().parents[1] / "baselines"


def assert_close(new, ref, name="", tol=RTOL):
    """
    Check that every element of `new` is close to the corresponding element
    of `ref` under a scale-normalised tolerance:

        |new - ref| / (|mean(ref)| + 1) < tol

    The denominator uses the mean magnitude of `ref` so that large-valued
    outputs are not held to an unreasonably tight absolute standard, while
    the +1 prevents division by near-zero for outputs that are close to 0.
    """
    new = np.asarray(new, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    assert new.shape == ref.shape, f"{name}: shape mismatch {new.shape} vs {ref.shape}"
    scale = np.abs(np.mean(ref)) + 1.0
    max_err = np.max(np.abs(new - ref)) / scale
    assert max_err < tol, (
        f"{name}: max normalised error {max_err:.2e} >= tol {tol:.2e}\n"
        f"  max |new-ref| = {np.max(np.abs(new - ref)):.6e}  scale = {scale:.6e}"
    )


def _load(name):
    path = BASELINES_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"Baseline {path.name} not found — run make_baselines.py first")
    return np.load(path)


# ---------------------------------------------------------------------------
# Shared synthetic model (must match make_baselines.py exactly)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return WishartProcessModel(
        degree=3,
        num_dims=2,
        extra_dims=1,
        variance_scale=1e-3,
        decay_rate=0.5,
        diag_term=1e-4,
    )


@pytest.fixture(scope="module")
def W():
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, shape=(3, 3, 2, 3)) * 0.1


@pytest.fixture(scope="module")
def x_grid():
    pts = np.linspace(-0.5, 0.5, 3)
    g = np.stack(np.meshgrid(pts, pts, indexing="ij"), axis=-1)
    return jnp.array(g.reshape(-1, 2))  # (9, 2)


# ---------------------------------------------------------------------------
# Regression: compute_U
# ---------------------------------------------------------------------------


def test_regression_compute_U(model, W, x_grid):
    bl = _load("compute_U")
    U = np.array(model.compute_U(W, x_grid))
    assert_close(U, bl["U"], name="compute_U")


# ---------------------------------------------------------------------------
# Regression: compute_Sigmas
# ---------------------------------------------------------------------------


def test_regression_compute_Sigmas(model, W, x_grid):
    bl = _load("compute_Sigmas")
    U = model.compute_U(W, x_grid)
    Sigma = np.array(model.compute_Sigmas(U))
    assert_close(Sigma, bl["Sigma"], name="compute_Sigmas")


# ---------------------------------------------------------------------------
# Regression: oddity_prediction
# ---------------------------------------------------------------------------


def test_regression_oddity_prediction():
    bl = _load("oddity_prediction")

    NGRID = 5
    MC = 200
    BANDWIDTH = 0.005
    DIAG_TERM = 1e-4
    NOISE_SCALE = 0.05

    w_ref = np.array([0.0, 0.0])
    w_ref_rep = np.tile(w_ref, (NGRID, 1))
    vec_lengths = np.linspace(0.05, 0.30, NGRID)
    d = np.array([1.0, 0.0])
    w_comps = w_ref[None, :] + vec_lengths[:, None] * d[None, :]

    Uref = jnp.array(np.tile(NOISE_SCALE * np.eye(2, 3), (NGRID, 1, 1)))
    U1 = Uref

    keys = jax.random.split(jax.random.PRNGKey(7), NGRID)
    pC = np.array(
        oddity_task.oddity_prediction(
            (jnp.array(w_ref_rep), jnp.array(w_comps), Uref, U1),
            keys,
            MC,
            BANDWIDTH,
            DIAG_TERM,
            oddity_task.simulate_oddity,
        )
    )
    assert_close(pC, bl["pC"], name="oddity_prediction/pC")


# ---------------------------------------------------------------------------
# Regression: ellipse fit
# ---------------------------------------------------------------------------


def test_regression_ellipse_fit():
    bl = _load("ellipse_fit")

    a, b, theta_deg = 0.15, 0.08, 30.0
    theta_rad = np.deg2rad(theta_deg)
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    x_local, y_local = a * np.cos(t), b * np.sin(t)
    w_comp = np.vstack(
        [
            cos_t * x_local - sin_t * y_local,
            sin_t * x_local + cos_t * y_local,
        ]
    )
    w_ref = np.array([0.0, 0.0])

    _, _, params_ell, _ = fit_2d_isothreshold_contour(w_ref, w_comp, nTheta=200, flag_force_centered_ref=True)
    Sigma = np.array(ellParamsQ_to_covMat(*params_ell[2:]))
    _, _, axes_lengths, theta_deg2 = covMat_to_ellParamsQ(Sigma)
    a2, b2 = axes_lengths[0], axes_lengths[1]

    assert_close(Sigma, bl["Sigma"], name="ellipse_fit/Sigma")
    assert_close(
        np.array([a2, b2, theta_deg2]),
        np.array([float(bl["a_fit"]), float(bl["b_fit"]), float(bl["theta_fit"])]),
        name="ellipse_fit/params",
    )
