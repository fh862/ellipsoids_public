"""
Generate regression baselines for the ellipsoids_public test suite.

Run this script once (from the repo root) to create/update the .npz
baseline files committed in tests/baselines/:

    python tests/regression/make_baselines.py

The same script can be run against any checkout (e.g. the reference
branch) to produce ground-truth baselines.  Commit the resulting .npz
files to fix the expected values for future test runs.
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

# Make sure the repo root is on sys.path when running as a script.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

jax.config.update("jax_enable_x64", True)

from analysis.ellipses_tools import (  # noqa: E402
    covMat_to_ellParamsQ,
    ellParamsQ_to_covMat,
    fit_2d_isothreshold_contour,
)
from core import oddity_task  # noqa: E402
from core.wishart_process import WishartProcessModel  # noqa: E402

BASELINES_DIR = REPO_ROOT / "tests" / "baselines"
BASELINES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared synthetic model  (degree-3, 2-D stimulus, 1 extra dim)
# ---------------------------------------------------------------------------

MODEL = WishartProcessModel(
    degree=3,
    num_dims=2,
    extra_dims=1,
    variance_scale=1e-3,
    decay_rate=0.5,
    diag_term=1e-4,
)

_key = jax.random.PRNGKey(0)
W = jax.random.normal(_key, shape=(3, 3, 2, 3)) * 0.1  # (degree,degree,ndims,ndims+extra)

# A small set of stimulus points: 3×3 grid in [-0.5, 0.5]^2
_pts = np.linspace(-0.5, 0.5, 3)
_g = np.stack(np.meshgrid(_pts, _pts, indexing="ij"), axis=-1)  # (3,3,2)
X_GRID = jnp.array(_g.reshape(-1, 2))  # (9, 2)


# ---------------------------------------------------------------------------
# 1. compute_U
# ---------------------------------------------------------------------------


def _baseline_compute_U():
    U = np.array(MODEL.compute_U(W, X_GRID))
    np.savez(BASELINES_DIR / "compute_U.npz", U=U)
    print(f"  compute_U: shape={U.shape}  mean={U.mean():.6f}")


# ---------------------------------------------------------------------------
# 2. compute_Sigmas
# ---------------------------------------------------------------------------


def _baseline_compute_Sigmas():
    U = MODEL.compute_U(W, X_GRID)
    Sigma = np.array(MODEL.compute_Sigmas(U))
    np.savez(BASELINES_DIR / "compute_Sigmas.npz", Sigma=Sigma)
    print(f"  compute_Sigmas: shape={Sigma.shape}  mean={Sigma.mean():.6f}")


# ---------------------------------------------------------------------------
# 3. oddity_prediction  (small batch, frozen RNG)
# ---------------------------------------------------------------------------


def _baseline_oddity_prediction():
    NGRID = 5
    MC = 200
    BANDWIDTH = 0.005
    DIAG_TERM = 1e-4

    w_ref = np.array([0.0, 0.0])
    w_ref_rep = np.tile(w_ref, (NGRID, 1))

    # Comparison points along a single direction
    vec_lengths = np.linspace(0.05, 0.30, NGRID)
    d = np.array([1.0, 0.0])
    w_comps = w_ref[None, :] + vec_lengths[:, None] * d[None, :]

    # Use fixed-scale identity U (independent of model W, so stable across refactors)
    NOISE_SCALE = 0.05
    Uref = jnp.array(np.tile(NOISE_SCALE * np.eye(2, 3), (NGRID, 1, 1)))
    U1 = Uref  # same noise at reference and comparison

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
    np.savez(
        BASELINES_DIR / "oddity_prediction.npz",
        pC=pC,
        vec_lengths=vec_lengths,
    )
    print(f"  oddity_prediction: pC={np.round(pC, 4)}")


# ---------------------------------------------------------------------------
# 4. ellipse fit round-trip
# ---------------------------------------------------------------------------


def _baseline_ellipse_fit():
    # Known ellipse: a=0.15, b=0.08, theta=30 deg
    a, b, theta_deg = 0.15, 0.08, 30.0
    theta_rad = np.deg2rad(theta_deg)
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    x_local = a * np.cos(t)
    y_local = b * np.sin(t)
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

    np.savez(
        BASELINES_DIR / "ellipse_fit.npz",
        Sigma=Sigma,
        params_ell=np.array(params_ell),
        a_fit=np.array(a2),
        b_fit=np.array(b2),
        theta_fit=np.array(theta_deg2),
    )
    print(f"  ellipse_fit: a={a2:.5f} b={b2:.5f} theta={theta_deg2:.2f}°")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Writing baselines to {BASELINES_DIR}/")
    _baseline_compute_U()
    _baseline_compute_Sigmas()
    _baseline_oddity_prediction()
    _baseline_ellipse_fit()
    print("Done.")
