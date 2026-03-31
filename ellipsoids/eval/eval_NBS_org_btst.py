#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:06:14 2026

@author: fangfang

Compute normalized Bures similarity (NBS) on a fine prediction grid between:
1. the model fit to the original dataset, and
2. each bootstrap model fit.

For efficiency, the script caches:
- the fine grid and original-fit covariance matrices in the original-fit pickle
- the fine grid, bootstrap-fit covariance matrices, and NBS values in each
  bootstrap pickle
  
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

#%%
#---------------------------------------------------------------------------
# SECTION 1: load the model fits to the empirical data
# --------------------------------------------------------------------------
# Example:
#   input directory:
#   'ELPS_analysis/Experiment_DataFiles/6D_Expt/sub1/fits'
#
#   selected file:
#   'Fitted_ColorDiscrimination_6dExpt_RGBcube_sub1_decayRate0.5_nBasisDeg5.pkl'
input_fileDir_fits, file_name = select_file_and_get_path()
full_path = os.path.join(input_fileDir_fits, file_name)

# Load the necessary variables from the file
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)
model_pred = deepcopy(vars_dict['model_pred_Wishart_grid_isoluminant'])
ndims = model_pred.ndims    

# Reuse cached fine-grid quantities if available; otherwise compute and store.
if "grid_fine" in vars_dict.keys() and "Sigmas_noise_grid_org" in vars_dict.keys():
    grid_fine = vars_dict["grid_fine"]
    Sigmas_noise_grid_org = vars_dict["Sigmas_noise_grid_org"]
    num_grid_pts_fine = grid_fine.shape[0]
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
# Section 2: For each bootstrap fit, load or compute NBS on the same fine grid
# -----------------------------------------------------------------------------
# NBS is computed pointwise between:
#   - the covariance matrices from the original-data fit
#   - the covariance matrices from a bootstrap fit
#
# The per-bootstrap NBS result is cached back into the corresponding bootstrap
# pickle to avoid repeating expensive matrix computations.

nDatasets = 120

#initialize
NBS_fine_grid_btst = np.full((nDatasets, *grid_fine.shape[:-1]), np.nan)

# Example:
#   input directory:
#   '/ELPS_analysis/Experiment_DataFiles/6D_Expt/sub1/fits/AEPsych_btst/decayRate0.5'
#
#   selected file:
#   'Fitted_ColorDiscrimination_6dExpt_RGBcube_sub1_decayRate0.5_nBasisDeg5_btst_AEPsych[0].pkl'
input_fileDir_fits_btst, file_name_btst = select_file_and_get_path()

for r in trange(nDatasets):
    # Replace the bootstrap index in the filename template.
    input_fileDir_fits_btst_r = input_fileDir_fits_btst
    file_name_r = file_name_btst.replace('AEPsych[0]', f'AEPsych[{r}]')

    # Load bootstrap-fit pickle.
    full_path_btst_r = f"{input_fileDir_fits_btst_r}/{file_name_r}"
    
    # Load bootstrap pickle for dataset r
    with open(full_path_btst_r, 'rb') as f:
        vars_dict_btst = pickled.load(f)
        
    # Reuse cached NBS if already present.
    if "NBS_fine_grid" in vars_dict_btst:
        grid_fine_btst = vars_dict_btst["grid_fine"]

        # Sanity check: cached NBS must correspond to the same fine grid.
        assert np.max(np.abs(np.asarray(grid_fine_btst) - np.asarray(grid_fine))) < 1e-10, (
            "The grid for which sigmas were computed does not match!"
        )

        NBS_fine_grid_btst[r] = vars_dict_btst["NBS_fine_grid"]

    else:
        # Compute covariance matrices on the same fine grid for bootstrap fit r.
        model_pred_btst = deepcopy(vars_dict_btst["model_pred_Wishart_grid_isoluminant"])
        model_btst = model_pred_btst.model
        W_btst = model_pred_btst.W_est

        Sigmas_noise_grid_btst = model_btst.compute_Sigmas(
            model_btst.compute_U(W_btst, grid_fine)
        )

        # Compute pointwise normalized Bures similarity (NBS) between
        # the original-fit and bootstrap-fit covariance matrices.
        NBS_fine_grid_btst[r] = ModelPerformance.compute_normalized_Bures_similarity_batch(
            Sigmas_noise_grid_org,
            Sigmas_noise_grid_btst,
        )

        # Cache the fine grid, bootstrap covariance matrices, and NBS values.
        vars_dict_btst["grid_fine"] = grid_fine
        vars_dict_btst["Sigmas_noise_grid_btst"] = Sigmas_noise_grid_btst
        vars_dict_btst["NBS_fine_grid"] = NBS_fine_grid_btst[r]

        with open(full_path_btst_r, 'wb') as f:
            pickled.dump(vars_dict_btst, f)
    
        del vars_dict_btst