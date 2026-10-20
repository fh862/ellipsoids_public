#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:06:14 2026

@author: fangfang

Evaluate the agreement between an original-data Wishart fit and its bootstrap
fits on a common fine prediction grid.

Legacy filename
---------------
This script was originally written only for normalized Bures similarity (NBS),
which is why its filename remains ``eval_NBS_org_btst.py``. The legacy name is
retained for legacy use only, so existing workflows and references do not
break, but the script now supports both NBS and the affine-invariant Riemannian
metric (AIRM). Set ``metric`` below to select which quantity is computed.

Supported metrics
-----------------
``metric = "NBS"``
    Computes normalized Bures similarity with
    ``compute_normalized_Bures_similarity_batch`` and caches it under
    ``NBS_fine_grid``. Larger dataset-level sums indicate greater similarity.

``metric = "AIRM"``
    Computes squared affine-invariant Riemannian distances with
    ``spd_affine_dist_sq_batch`` and caches them under ``AIRM_fine_grid``.
    Smaller dataset-level sums indicate a bootstrap fit closer to the original
    fit. Despite the cache-key shorthand, each grid value is a squared AIRM
    distance.

Caching and downstream use
--------------------------
The selected original-fit pickle is updated with ``grid_fine`` and
``Sigmas_noise_grid_org`` when needed. Each bootstrap pickle is updated with
the matching fine grid, ``Sigmas_noise_grid_btst``, and the selected metric's
fine-grid values. Compatible cached quantities are reused, so running the
script once for NBS and again for AIRM does not recompute covariance matrices.
A metric-specific NPZ beside the bootstrap pickles stores the shared grid and
all bootstrap metric arrays, with the bootstrap index along the first axis.

Downstream scripts such as ``visualize_CI_Wishart_grid_2d.py`` sum the selected
fine-grid values to obtain one score per bootstrap dataset. NBS must be ranked
in descending order, whereas AIRM must be ranked in ascending order.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import dill as pickled
from tqdm import trange
import numpy as np
from copy import deepcopy
import os
from analysis.utils_load import select_file_and_get_path
from analysis.model_performance import ModelPerformance


def grids_match(grid_1, grid_2, atol=1e-5):
    """Return whether two cached prediction grids have matching values."""
    grid_1 = np.asarray(grid_1)
    grid_2 = np.asarray(grid_2)
    return (
        grid_1.shape == grid_2.shape
        and np.allclose(grid_1, grid_2, rtol=0, atol=atol)
    )


# Configuration. Keep this above file selection so invalid input fails before
# any pickle can be updated.
nDatasets = 120
metric = "AIRM"
metric = metric.upper()
if metric not in {"NBS", "AIRM"}:
    raise ValueError('metric must be either "NBS" or "AIRM".')
metric_key = f"{metric}_fine_grid"

#---------------------------------------------------------------------------
# SECTION 1: load the model fits to the empirical data
# --------------------------------------------------------------------------
# Example:
#   input directory:
#   'ELPS_analysis/Experiment_DataFiles/6D_Expt/sub1/fits'
#
#   selected file:
#   'Fitted_ColorDiscrimination_6dExpt_RGBcube_sub1_decayRate0.4_nBasisDeg5.pkl'
input_fileDir_fits, file_name = select_file_and_get_path()
full_path = os.path.join(input_fileDir_fits, file_name)

# Load the necessary variables from the file
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)
model_pred = deepcopy(vars_dict['model_pred_Wishart']) #model_pred_Wishart_grid_isoluminant
ndims = model_pred.ndims    

# Reuse cached fine-grid quantities if available; otherwise compute and store.
if "grid_fine" in vars_dict.keys() and "Sigmas_noise_grid_org" in vars_dict.keys():
    grid_fine = vars_dict["grid_fine"]
    Sigmas_noise_grid_org = vars_dict["Sigmas_noise_grid_org"]
else:
    
    #for dichromats 
    # grid_fine1 = jnp.linspace(-0.6, 0.6, 73)
    # grid_fine2 = jnp.linspace(-0.85, 0.85, 103)
    # grid_fine = jnp.stack(jnp.meshgrid(grid_fine1, grid_fine2), axis = -1)
    
    # Define the fine prediction grid
    num_grid_pts_fine = 103
    grid_fine = jnp.stack(
        jnp.meshgrid(*[jnp.linspace(-0.85, 0.85, num_grid_pts_fine) for _ in range(ndims)]),
        axis=-1
    )

    # Compute covariance matrices on the fine grid for the original-data fit.
    model = model_pred.model
    W_org = model_pred.W_est
    Sigmas_noise_grid_org = model.compute_Sigmas(model.compute_U(W_org, grid_fine))

    # Cache the fine grid and covariance matrices in the original-fit pickle.
    vars_dict["grid_fine"] = grid_fine
    vars_dict["Sigmas_noise_grid_org"] = Sigmas_noise_grid_org
    with open(full_path, 'wb') as f:
        pickled.dump(vars_dict, f)

# -----------------------------------------------------------------------------
# Section 2: Load or compute the selected metric on the same fine grid
# -----------------------------------------------------------------------------
# Example:
#   input directory:
#   '/ELPS_analysis/Experiment_DataFiles/6D_Expt/sub1/fits/AEPsych_btst/decayRate0.4'
#
#   selected file:
#   'Fitted_ColorDiscrimination_6dExpt_RGBcube_sub1_decayRate0.4_nBasisDeg5_btst_AEPsych[0].pkl'
input_fileDir_fits_btst, file_name_btst = select_file_and_get_path()

# Collect cached and newly computed values in bootstrap-index order.
metric_fine_grid_all = np.empty((nDatasets,) + grid_fine.shape[:-1], dtype=float)

for r in trange(nDatasets):
    # Replace the bootstrap index in the filename template.
    input_fileDir_fits_btst_r = input_fileDir_fits_btst
    file_name_r = file_name_btst.replace('AEPsych[0]', f'AEPsych[{r}]')

    # Load bootstrap-fit pickle.
    full_path_btst_r = f"{input_fileDir_fits_btst_r}/{file_name_r}"
    
    # Load bootstrap pickle for dataset r
    with open(full_path_btst_r, 'rb') as f:
        vars_dict_btst = pickled.load(f)

    cached_grid_matches = (
        "grid_fine" in vars_dict_btst
        and grids_match(vars_dict_btst["grid_fine"], grid_fine)
    )

    # Reuse the selected metric if it is already cached on the same grid.
    if metric_key in vars_dict_btst and cached_grid_matches:
        cached_metric = np.asarray(vars_dict_btst[metric_key])
        if cached_metric.shape != grid_fine.shape[:-1]:
            raise ValueError(
                f"Cached {metric_key} has shape {cached_metric.shape}; "
                f"expected {grid_fine.shape[:-1]}."
            )
        if not np.all(np.isfinite(cached_metric)):
            raise ValueError(f"Cached {metric_key} contains non-finite values.")

        print(f"Bootstrap {r}: {metric_key} was already calculated.")
    else:
        # Metrics without a verifiably matching grid are no longer valid once
        # this pickle is updated to the current fine grid.
        if not cached_grid_matches:
            vars_dict_btst.pop("NBS_fine_grid", None)
            vars_dict_btst.pop("AIRM_fine_grid", None)

        # Reuse covariance matrices cached by a previous NBS/AIRM run when the
        # grid matches; otherwise compute them from the bootstrap model fit.
        Sigmas_noise_grid_btst = None

        reused_cached_sigmas = (
            cached_grid_matches
            and "Sigmas_noise_grid_btst" in vars_dict_btst
        )
        if reused_cached_sigmas:
            Sigmas_noise_grid_btst = vars_dict_btst["Sigmas_noise_grid_btst"]

        if Sigmas_noise_grid_btst is None:
            # Older save files stored the fitted model and weights separately;
            # newer ones store both inside `model_pred_Wishart`.
            try:
                model_btst = deepcopy(vars_dict_btst["model"])
                W_btst = vars_dict_btst["W_est"]
            except KeyError:
                try:
                    model_btst_pred = deepcopy(
                        vars_dict_btst["model_pred_Wishart"]
                    )
                    model_btst = model_btst_pred.model
                    W_btst = model_btst_pred.W_est
                except KeyError as exc:
                    raise KeyError(
                        "Bootstrap save file must contain either (`model`, "
                        "`W_est`) or `model_pred_Wishart`."
                    ) from exc

            Sigmas_noise_grid_btst = model_btst.compute_Sigmas(
                model_btst.compute_U(W_btst, grid_fine)
            )

        # Compute the selected pointwise metric.
        if metric == "NBS":
            metric_fine_grid_btst = ModelPerformance.compute_normalized_Bures_similarity_batch(
                Sigmas_noise_grid_org,
                Sigmas_noise_grid_btst,
            )
        else:
            metric_fine_grid_btst = ModelPerformance.spd_affine_dist_sq_batch(
                Sigmas_noise_grid_org,
                Sigmas_noise_grid_btst,
            )

        if not np.all(np.isfinite(metric_fine_grid_btst)):
            raise ValueError(
                f"Computed {metric_key} contains non-finite values for "
                f"bootstrap dataset {r}."
            )

        # Update only what was missing or invalid.
        if not cached_grid_matches:
            vars_dict_btst["grid_fine"] = grid_fine

        if not reused_cached_sigmas:
            vars_dict_btst["Sigmas_noise_grid_btst"] = Sigmas_noise_grid_btst

        vars_dict_btst[metric_key] = metric_fine_grid_btst

        with open(full_path_btst_r, 'wb') as f:
            pickled.dump(vars_dict_btst, f)

    metric_fine_grid_all[r] = np.asarray(vars_dict_btst[metric_key])
    del vars_dict_btst

# Save one numeric archive for this participant and selected metric.
output_npz_path = os.path.join(
    input_fileDir_fits,
    'output_data_no_classobjects',
    f"{metric}_all_bootstraps.npz",
)
np.savez(output_npz_path, grid_fine=np.asarray(grid_fine),
         **{metric_key: metric_fine_grid_all})
print(f"Saved {metric} bootstrap values: {output_npz_path}")
