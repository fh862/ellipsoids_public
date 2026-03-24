#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:37:06 2025

@author: fangfang

This script supports two related visualization workflows for Wishart model predictions.

(1) Bootstrap confidence intervals (CI) within a single subject
    - Solid contour lines: model predictions fit to the original dataset.
    - Shaded regions: model predictions fit to N bootstrap-resampled datasets.
    - Bootstrap datasets whose Normalized Bures Similarity (NBS) scores fall in
      the bottom 5% are excluded before computing the CI.

(2) Cross-subject confidence intervals (CI) across multiple subjects
    - No solid contour lines (no single “reference” subject).
    - Shaded regions: model predictions aggregated across N different subjects.

Notes on implementation:
- The current code structure is slightly inefficient for workflow (2).
- For workflow (1), the original dataset is loaded once, and bootstrap datasets
  are loaded separately, which is efficient.
- For workflow (2), the same subject’s original dataset file is currently loaded
  twice. This could be optimized in the future but is not performance-critical
  for typical use cases.

See also:
- visualize_bootstrap_CI_atMOCS.py

"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import dill as pickled
from tqdm import trange
import numpy as np
import re
import matplotlib.pyplot as plt
from dataclasses import replace
from copy import deepcopy
import sys
import os
script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from analysis.utils_load import select_file_and_get_path
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization,\
    add_CI_ellipses, Plot2DPredSettings
from core.model_predictions import rerun_model_pred_wExisting_model
from plotting.wishart_plotting import PlotSettingsBase 
from analysis.conf_interval import find_inner_outer_contours_for_gridRefs
from analysis.model_performance import ModelPerformance

#%%
#---------------------------------------------------------------------------
# SECTION 1: load the model fits to the empirical data
# --------------------------------------------------------------------------
# Select the file containing the model fits
# Navigate to the directory: ELPS_analysis/Experiment_DataFiles/sub#
#'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.5_nBasisDeg5.pkl'

# or simulated subject based on CIE1994
# 'META_analysis/ModelFitting_DataFiles/4dTask/CIE/sub1/decayRate0.5'
# 'Fitted_byWishart_Isoluminant plane_4DExpt_300_300_300_5100_AEPsychSampling_EAVC_decayRate0.5_nBasisDeg5_sub1.pkl'

# or data from the adaptation experiment with different adapting backgrounds
# 'ELPS_analysis/Experiment_DataFiles/4D_Expt_varyingBackground/sub12/fits'
# 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub12_decayRate0.5_nBasisDeg5_blue.pkl' or gray

# dichromats
# 'ELPS_analysis/Experiment_DataFiles/4D_Expt_dichromats/sub15/fits'
# 'Fitted_ColorDiscrimination_4dExpt_LSisolating plane_sub15_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl'
input_fileDir_fits, file_name = select_file_and_get_path()

# Construct the full path to the selected file
full_path = os.path.join(input_fileDir_fits, file_name)

# Load the necessary variables from the file
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)

# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict['color_thres_data']

# - Dimensionality of the color space (e.g., 2D for isoluminant planes)
ndims = color_thres_data.color_dimension

# - Experimental trial data
expt_trial = vars_dict['expt_trial']

#%%
"""
Grid-handling logic for model predictions:

Model predictions stored in the pickle files are typically computed on a fixed
grid (e.g., 5×5 or 7×7). However, for visualization or comparison purposes, we
may want predictions on a different grid resolution.

We therefore:
- Check whether predictions already exist for the desired grid size.
- If they do, reuse them directly.
- If not, recompute the model predictions on the desired grid and (optionally)
  append them to the existing pickle file for future reuse.

Assumptions:
- All grids span the same stimulus range: [-0.7, 0.7] in each dimension.

"""

num_grid_pts_desired = 5
flag_append_data = False

# Construct the key name based on the desired grid size
key_grid = f'model_pred_Wishart_grid{num_grid_pts_desired}'

if key_grid in vars_dict.keys():
    # Use precomputed results if available
    model_pred = vars_dict[key_grid]
    grid = vars_dict[f'grid{num_grid_pts_desired}']
    ndims = model_pred.model.num_dims
else:
    # - Number of grid points per dimension for model prediction computations
    model_pred = deepcopy(vars_dict['model_pred_Wishart'])
    ndims = model_pred.model.num_dims
    num_grid_pts = model_pred.fitEll_scaled.shape[0]
    
    if num_grid_pts == num_grid_pts_desired:
         model_pred = vars_dict['model_pred_Wishart']
         grid = vars_dict['grid']
    else:
        # Recompute everything if the desired grid size is not available
        grid = jnp.stack(jnp.meshgrid(*[jnp.linspace(-0.7, 0.7, num_grid_pts_desired) \
                                        for _ in range(ndims)]), axis=-1)
    
        # Use the helper function to recompute model predictions and transformed grid
        model_pred, _ = rerun_model_pred_wExisting_model(
            grid, model_pred, color_thres_data
        )
    
        # Optionally append results to the pickle
        if flag_append_data:
            vars_dict[key_grid] = model_pred
            vars_dict[f'grid{num_grid_pts_desired}'] = grid
    
            # Save the updated pickle file
            with open(full_path, 'wb') as f:
                pickled.dump(vars_dict, f)

#%% 
#---------------------------------------------------------------------------
# SECTION 2: load another 
# --------------------------------------------------------------------------
# Step 1: Define mode
##############################################################################
flag_load_other_subjects = False  # <-- Set True if loading different subjects
subject_list = [1,2,4,6,7,8,10,11]         # Only used if loading other subjects
##############################################################################

# Step 2: Select the bootstrapped data file (choose one as a reference)
# Option 1: 
# 'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits/AEPsych_btst/decayRate0.4'
# 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.4_nBasisDeg5_btst_AEPsych[0].pkl'

# Option 2 (if we want to plot CI across subjects, we need to reload sub1's data):
# 'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits'
# 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.4_nBasisDeg5.pkl'

# Option 3 (simulated subject based on CIE1994):
# 'META_analysis/ModelFitting_DataFiles/4dTask/CIE/sub1/decayRate0.4'
# 'Fitted_byWishart_Isoluminant plane_4DExpt_300_300_300_5100_AEPsychSampling_EAVC_decayRate0.4_nBasisDeg5_sub1_btst_AEPsych[0].pkl'

# Option 4
# 'ELPS_analysis/Experiment_DataFiles/4D_Expt_varyingBackground/sub12/fits'
# 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub12_decayRate0.4_nBasisDeg5_btst_AEPsych[0]_blue.pkl'
input_fileDir_fits_others, file_name_others = select_file_and_get_path()

# Step 3: Setup parameters
if flag_load_other_subjects:
    nDatasets = len(subject_list)
else:
    nDatasets = 120  # e.g., 120 bootstrap datasets

# Step 4: Pre-allocate arrays for ellipse parameters and similarity scores
#   - params_ell[i, j, r, :] stores the ellipse parameters at grid location (i, j)
#   - NBS_grid[i, j, r] stores the normalized Bures similarity between:
#       Sigma_noise(original)[i, j]  vs  Sigma_noise(dataset r)[i, j]
#     (NBS≈1 means nearly identical covariance shape/scale)
grid_shape = grid.shape[:-1]
params_ell_shape = grid_shape + (nDatasets, 5)
params_ell = np.full(params_ell_shape, np.nan)

full_path_others = []

# Step 5: Loop over datasets (bootstraps or subjects), load predictions, and compute NBS
for r in trange(nDatasets):
    # 5a) Construct file path for dataset r
    if flag_load_other_subjects:
        # Cross-subject mode: swap "subXX" in directory + filename
        subject_id = subject_list[r]

        match = re.search(r'sub\d+', file_name_others)
        if not match:
            raise ValueError('No "subXXX" pattern found in the file name!')

        old_sub = match.group()
        input_fileDir_fits_others_r = input_fileDir_fits_others.replace(old_sub, f'sub{subject_id}')
        file_name_r = file_name_others.replace(old_sub, f'sub{subject_id}')

    else:
        # Bootstrap mode: swap the bootstrap index in the filename
        input_fileDir_fits_others_r = input_fileDir_fits_others
        file_name_r = file_name_others.replace('AEPsych[0]', f'AEPsych[{r}]')

    full_path_others_r = f"{input_fileDir_fits_others_r}/{file_name_r}"
    full_path_others.append(full_path_others_r)

    # 5b) Load the pickle for dataset r
    with open(full_path_others_r, 'rb') as f:
        vars_dict_others = pickled.load(f)

    # 5c) Retrieve model predictions on the desired grid
    #     Priority:
    #       1) Use cached predictions for grid{num_grid_pts_desired} if present
    #       2) Otherwise, use existing predictions if grid matches
    #       3) Otherwise, recompute predictions on the desired grid
    if key_grid in vars_dict_others:
        model_pred_r = vars_dict_others[key_grid]
        param_ell_r = model_pred_r.params_ell
    else:
        model_pred_r = deepcopy(vars_dict_others['model_pred_Wishart'])
        num_grid_pts_others = model_pred_r.fitEll_scaled.shape[0]

        if num_grid_pts_others == num_grid_pts_desired:
            # Existing predictions already match the desired grid size
            param_ell_r = vars_dict_others['model_pred_Wishart'].params_ell
        else:
            # Recompute predictions on the desired grid
            model_pred_r, _ = rerun_model_pred_wExisting_model(grid, 
                                                               model_pred_r, 
                                                               color_thres_data)
            param_ell_r = model_pred_r.params_ell

            # Optionally cache the new results back into the pickle
            if flag_append_data:
                vars_dict_others[key_grid] = model_pred_r
                vars_dict_others[f'grid{num_grid_pts_desired}'] = grid
                with open(full_path_others_r, 'wb') as f:
                    pickled.dump(vars_dict_others, f)

    # 5d) Store ellipse params and compute per-grid-point similarity
    for idx in np.ndindex(grid_shape):
        params_ell[*idx, r] = param_ell_r[idx[0]][idx[1]]
            
#%%
#---------------------------------------------------------------------------
# SECTION 3: compute the normalized bures similarity score and 95% CI
# --------------------------------------------------------------------------
if not flag_load_other_subjects:
    # ---------------------------------------------------------------------
    # Precompute / load prediction grid + original-dataset noise ellipses
    # ---------------------------------------------------------------------
    # We evaluate noise covariance matrices on a fixed fine grid in 2D model space.
    # If these have already been saved into the original-dataset pickle, load them;
    # otherwise compute once and cache them back to disk.
    if "grid_fine" in vars_dict.keys() and "Sigmas_noise_grid_org" in vars_dict.keys():
        grid_fine = vars_dict["grid_fine"]
        Sigmas_noise_grid_org = vars_dict["Sigmas_noise_grid_org"]
        num_grid_pts_fine = grid_fine.shape[0]
    else:
        # Define the fine prediction grid
        # num_grid_pts_fine = 103
        # grid_fine = jnp.stack(
        #     jnp.meshgrid(*[jnp.linspace(-0.85, 0.85, num_grid_pts_fine) for _ in range(ndims)]),
        #     axis=-1
        # )
        
        grid_fine1 = jnp.linspace(-0.6, 0.6, 73)
        grid_fine2 = jnp.linspace(-0.85, 0.85, 103)
        
        grid_fine = jnp.stack(jnp.meshgrid(grid_fine1, grid_fine2), axis = -1)
    
        # Compute noise covariance matrices for the original dataset fit
        model = model_pred.model
        W_org = model_pred.W_est
        Sigmas_noise_grid_org = model.compute_Sigmas(model.compute_U(W_org, grid_fine))
    
        # Cache results to avoid recomputing next time
        vars_dict["grid_fine"] = grid_fine
        vars_dict["Sigmas_noise_grid_org"] = Sigmas_noise_grid_org
        with open(full_path, 'wb') as f:
            pickled.dump(vars_dict, f)
    
    # ---------------------------------------------------------------------
    # For each bootstrap dataset: compute / load NBS on the fine grid
    # ---------------------------------------------------------------------
    # NBS is computed per grid point between the original-fit covariance matrix
    # and the bootstrap-fit covariance matrix. Results are cached per bootstrap
    # pickle to avoid repeating expensive matrix operations.
    NBS_fine_grid_btst = np.full((nDatasets, num_grid_pts_fine, num_grid_pts_fine), np.nan)
    
    for r in trange(nDatasets):
        # Load bootstrap pickle for dataset r
        with open(full_path_others[r], 'rb') as f:
            vars_dict_others = pickled.load(f)
    
        # Reuse cached NBS if available
        # if "NBS_fine_grid" in vars_dict_others.keys():
        #     NBS_fine_grid_btst[r] = vars_dict_others["NBS_fine_grid"]
        # else:
        # Compute covariance matrices on the same fine grid for bootstrap fit r
        model_pred_btst = deepcopy(vars_dict_others["model_pred_Wishart"])
        model_btst = model_pred_btst.model
        W_btst = model_pred_btst.W_est
        Sigmas_noise_grid_btst = model_btst.compute_Sigmas(
            model_btst.compute_U(W_btst, grid_fine)
        )

        # Compute NBS at each grid location
        for idx in np.ndindex(grid_fine.shape[:-1]):
            NBS_fine_grid_btst[r, *idx] = \
                ModelPerformance.compute_normalized_Bures_similarity(
                Sigmas_noise_grid_org[*idx],
                Sigmas_noise_grid_btst[*idx],
            )

        # Cache grid, covariance matrices, and NBS back to the bootstrap pickle
        vars_dict_others["grid_fine"] = grid_fine
        vars_dict_others["Sigmas_noise_grid_btst"] = Sigmas_noise_grid_btst
        vars_dict_others["NBS_fine_grid"] = NBS_fine_grid_btst[r]
        with open(full_path_others[r], 'wb') as f:
            pickled.dump(vars_dict_others, f)

        del vars_dict_others
    
    # ---------------------------------------------------------------------
    # Rank bootstraps by similarity to original fit and keep top 95%
    # ---------------------------------------------------------------------
    # Aggregate NBS over grid to obtain one similarity score per bootstrap dataset
    NBS_sum = np.sum(NBS_fine_grid_btst.reshape(nDatasets, -1), axis=1)
    
    # Rank bootstrap datasets by descending similarity to original fit
    idx_NBS_sort_descending = np.argsort(NBS_sum)[::-1]
    NBS_sorted = NBS_sum[idx_NBS_sort_descending]
    
    # Keep top 95% of bootstrap datasets for confidence interval construction
    nDatasets_CI = int(nDatasets * 0.95)
    idx_keep_NBS = idx_NBS_sort_descending[:nDatasets_CI]
    
    # Retain ellipse parameters corresponding to selected bootstrap datasets
    params_ell_within_CI = params_ell[:, :, idx_keep_NBS]
else:
    #if we are loading data from different subjects, then we want to compute the full
    # range instead of 95% confidence interval
    params_ell_within_CI = params_ell

# Compute confidence interval contours at each grid point
#   - fitEll_min / fitEll_max represent the inner/outer envelopes across datasets
fitEll_min, fitEll_max = find_inner_outer_contours_for_gridRefs(params_ell_within_CI)
        
#%%           
# -------------------------------------------------------------------------
# Optional: load ground-truth predictions (simulated datasets only)
# -------------------------------------------------------------------------
# Ground truth is only defined for simulated data (e.g., CIE-based observers).
# For real experimental data, there is no “true” covariance/threshold field.
flag_load_gt = False  # toggle: load a ground-truth file for comparison
if flag_load_gt:
    # Select a ground-truth pickle file (typically contains model predictions
    # computed directly from the generative simulation / observer model)
    #
    # Example:
    #   'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/Isoluminant plane'
    #   'Fitted_isothreshold_Isoluminant plane_CIE1994_sim18000total_samplingNearContour_jitter0.3_seed0_bandwidth0.005_decay0.4_oddity.pkl'
    gt_fileDir_fits, gt_file_name = select_file_and_get_path()
    gt_full_path = os.path.join(gt_fileDir_fits, gt_file_name)

    # Load ground-truth variables
    with open(gt_full_path, 'rb') as f:
        vars_dict_gt = pickled.load(f)

    # Retrieve ground-truth model predictions evaluated on the desired grid
    # Priority:
    #   1) Use cached predictions for grid{num_grid_pts_desired} if present
    #   2) Otherwise fall back to the default predictions, but require that
    #      they already match the desired grid size (no recomputation here)
    if key_grid in vars_dict_gt:
        gt_model_pred = deepcopy(vars_dict_gt[key_grid])
    else:
        gt_model_pred = deepcopy(vars_dict_gt['model_pred_Wishart'])

        num_grid_pts = gt_model_pred.fitEll_scaled.shape[0]
        if num_grid_pts != num_grid_pts_desired:
            raise ValueError(
                "Ground-truth model predictions were computed on a "
                f"{num_grid_pts}×{num_grid_pts} grid, but "
                f"{num_grid_pts_desired}×{num_grid_pts_desired} was requested."
            )

    # Use GT ellipses as the reference overlay in the visualization
    reference_ell_vis = gt_model_pred.fitEll_unscaled
    fig_name_ext = '_wGroundTruth'

else:
    # Simulated data, but user chose not to load ground truth
    reference_ell_vis = model_pred.fitEll_unscaled
    fig_name_ext = ''

#%% 
#---------------------------------------------------------------------------
# SECTION 5: visualize the model predictions with confidence intervals
# --------------------------------------------------------------------------
output_figDir_fits = os.path.join(os.path.dirname(input_fileDir_fits.replace('DataFiles', 'FigFiles')), 'AEPsych_btst')
if flag_load_other_subjects:
    output_figDir_fits = re.sub(r'sub\d+', 'groupData', output_figDir_fits)
    fig_name = re.sub(r'sub\d+', 'groupData', file_name[:-4])
else:
    fig_name = f"{file_name[:-4]}_wBtstCI{fig_name_ext}.pdf"
os.makedirs(output_figDir_fits, exist_ok=True)

# Create a base plotting settings instance (shared across plots)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 8)

# Initialize 2D prediction settings by copying from base and overriding method-specific parameters
pred2D_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
pred2D_settings = replace(pred2D_settings, 
                          visualize_samples= False,
                          visualize_gt = False, #if flag_load_other_subjects else True,
                          visualize_model_estimatedCov = False,
                          flag_rescale_axes_label = False,
                          visualize_model_pred = True,
                          modelpred_alpha = 1,
                          ticks = np.linspace(-0.7, 0.7,5),
                          modelpred_lw = 0.75,
                          modelpred_ls = '-',
                          gt_lw = 0.5, 
                          gt_lc = 'k',
                          gt_label = 'Model predictions (original dataset)' \
                              if not flag_load_gt else 'Ground truths',
                          gt_ls = '-',  
                          title =f'decay rate = {model_pred.model.decay_rate}',
                          fig_name = fig_name) 
# Initialize Visualization Class for Wishart Predictions
wishart_pred_vis_wCI = WishartPredictionsVisualization(expt_trial,
                                                       model_pred.model, 
                                                       model_pred, 
                                                       color_thres_data,
                                                       settings = pltSettings_base,
                                                       save_fig = True)
# Create figure and axes for plotting
fig, ax = plt.subplots(1, 1, figsize=pred2D_settings.fig_size, dpi=pred2D_settings.dpi)

# plot the confidence interval
for idx in np.ndindex(grid.shape[:-1]):
    cm = color_thres_data.W2D_to_rgb(grid[*idx])
    if idx == (0,0):
        if flag_load_other_subjects: lbl = f'100% across subjects CI ({nDatasets} datasets)' 
        else: lbl = f'95% bootstrap CI ({nDatasets} datasets)' 
    else:
        lbl = None
    #adapting_bg_2DW = np.array([-0.218, 0.5461, 0.9652])  #np.array([-0.0021, 0.0023, 0.9652]) np.array([-0.218, 0.5461, 0.9652])
    #adapting_bg_rgb = color_thres_data.M_2DWToRGB @ adapting_bg_2DW
    #ax.scatter(*adapting_bg_2DW[:2], marker = '*', color = adapting_bg_rgb, 
    #           edgecolor = 'k',lw = 0.2, s = 30,
    #           label = 'Adapting background' if idx == (0,0) else None)
    add_CI_ellipses(fitEll_min[*idx], fitEll_max[*idx],
                    ax=ax, cm=cm, label=lbl, lw_outer = 0,
                    alpha = 0.5)

# Overlay model predictions (joint fits) onto the same axes
wishart_pred_vis_wCI.plot_2D(grid, gt_ellipses= reference_ell_vis, 
                             ax=ax, settings=pred2D_settings)
