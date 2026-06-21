#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:36:21 2025

@author: fangfang

Export: Model-derived data files

This script loads a fitted Wishart Process Psychophysical Model (WPPM) from a
.pkl file and exports a collection of datasets that include both model-derived
quantities and trial-level behavioral data. In addition to model predictions,
the exported files also include validation (MOCS) trials and pre-generated
Sobol trials, with trials organized by type.

1) Thres_ellipses_sub#.csv
   Threshold covariance ellipses evaluated on a coarse reference grid.
   (If we want to export grid of 5 x 5, change 
        grid = vars_dict["grid"] -> vars_dict['grid_MacAdam']
        model_pred = deepcopy(vars_dict["model_pred_Wishart"]) -> deepcopy(vars_dict['model_pred_MacAdam'])
    )

2) Noise_ellipses_sub#.csv
   Noise covariance ellipses evaluated on a fine reference grid.

3) Bestfit_W_sub#.csv
   Best-fit Wishart model weight tensor, unrolled into a long-form table with
   explicit indices.

4) trial_data_pooled_by_type.csv
   Reference–comparison stimulus pairs and binary responses, pooled across
   different trial types (AEPsych, MOCS, and pre-generated Sobol trials).
   
5) Thres_at_validation_conditions_sub#.csv
   This file contains estimated discrimination thresholds at the validation 
   conditions. For each validation condition, we fitted a Weibull psychometric 
   function with the guess rate fixed at 1/3. The threshold was defined as the 
   stimulus level yielding 66.7% correct performance. 

6) Thres_at_validation_conditions_WPPM_sub#.csv
   This file contains estimated discrimination thresholds predicted by the WPPM.
   
"""

from copy import deepcopy
import re
import dill as pickled
import os
import numpy as np
import pandas as pd
from tqdm import trange
from dataclasses import replace
import matplotlib.pyplot as plt
from analysis.utils_load import select_file_and_get_path, extract_sub_number
from analysis.data_validation import DataExport
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization, \
    Plot2DSamplingSettings
from plotting.wishart_plotting import PlotSettingsBase 
from core import viz
from plotting.visualize_MOCS import MOCSTrialsVisualization, PlotThresCompSettings
from plotting.visualize_MOCS import PlotThresCompSettings_bds as plt_st

#%%
# -----------------------------------------------------------
# SECTION 1: Load WPPM fit
# -----------------------------------------------------------
# Select a fitted model: ELPS_analysis/Experiment_DataFiles/sub#
# Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl
input_fileDir_fits, file_name = select_file_and_get_path()
subN = extract_sub_number(file_name)

full_path = os.path.join(input_fileDir_fits, file_name)
with open(full_path, "rb") as f:
    vars_dict = pickled.load(f)

# Color-space transforms (DKL/RGB/W, etc.) and dimensionality
color_thres_data = vars_dict["color_thres_data"]

# Trial-level experimental data used for fitting (AEPsych-chosen trials + inserted Sobol trials)
# Contains xref (reference), x1 (comparison), and y (binary/choice response)
expt_trial = vars_dict["expt_trial"]

# Model evaluation grids:
#   - grid:      coarse set of reference locations (e.g., 7x7x2)
#   - grid_fine: dense grid for smooth visualization/export (e.g., 103x103x2)
grid = vars_dict["grid"]
grid_fine = vars_dict["grid_fine"]

# Fitted Wishart-process model object (parameters + derived predictions on grids)
model_pred = deepcopy(vars_dict["model_pred_Wishart"])

# Best-fit basis weights (W) defining the spatially varying covariance field
W_org = model_pred.W_est # e.g., (deg, deg, ndims, ndims+extra_dims)

# Model-predicted threshold covariance matrices on the coarse grid (one 2x2 Sigma per grid point)
Sigmas_thres_grid_org = model_pred.Sigmas_thres_grid        # (7, 7, 2, 2)
Sigmas_noise_grid_org = vars_dict["Sigmas_noise_grid_org"]  # (103, 103, 2, 2)

# Output directory for CSV exports (no class objects, just flat numeric tables)
output_dir = os.path.join(input_fileDir_fits, "output_data_no_classobjects")
os.makedirs(output_dir, exist_ok=True)

#%%
# -----------------------------------------------------------
# SECTION 2: Export original (non-bootstrap) results to CSV
# -----------------------------------------------------------
# Export threshold ellipses on the *coarse* grid:
# Each row corresponds to one reference location in `grid`, with its 2x2 Sigma flattened.
path_csv_thres = os.path.join(output_dir, f"Thres_ellipses_sub{subN}.csv")
_, grid_flat, Sigmas_thres_flat_org = DataExport.export_ellipses_csv(grid=grid,
                               Sigmas=Sigmas_thres_grid_org,
                               grid_col="grid_ref",
                               sigma_col="Sigmas_thres_grid_org",
                               out_path=path_csv_thres
                               )

# Export noise ellipses on the *fine* grid:
# Each row corresponds to one reference location in `grid_fine`, with its 2x2 Sigma flattened.
path_csv_noise = os.path.join(output_dir, f"Noise_ellipses_sub{subN}.csv")
_, grid_fine_flat, Sigmas_noise_flat_org = DataExport.export_ellipses_csv(grid=grid_fine,
                               Sigmas=Sigmas_noise_grid_org,
                               grid_col="grid_ref_fine",
                               sigma_col="Sigmas_noise_grid_org",
                               out_path=path_csv_noise
                               )

# Export best-fit weight tensor W (basis coefficients) to CSV for reproducibility/inspection
path_csv_weights = os.path.join(output_dir, f"Bestfit_W_sub{subN}.csv")
_, _, _ = DataExport.export_weights_csv(W_org, 
                                        "i,j,k,l", 
                                        "W_org",
                                        out_path=path_csv_weights
                                        )

#%% 
# -----------------------------------------------------------
# SECTION 3: Load WPPM/Wishart fits for bootstrap datasets
# -----------------------------------------------------------
#ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits/AEPsych_btst/decayRate0.5'
# e.g. 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.4_varScaler0.0003_nBasisDeg5_btst_AEPsych[0].pkl'
btst_fileDir_fits, btst_file_name = select_file_and_get_path()
nBtst = 120

# Allocate arrays to hold bootstrap outputs in a shape-aligned way:
#   - Sigmas_thres_grid_btst: (nBtst, *Sigmas_thres_grid_org.shape)
#   - Sigmas_noise_grid_btst: (nBtst, *Sigmas_noise_grid_org.shape)
#   - W_btst:                (nBtst, *W_org.shape)
Sigmas_thres_grid_btst = np.full((nBtst,) + Sigmas_thres_grid_org.shape, np.nan)
Sigmas_noise_grid_btst = np.full((nBtst,) + Sigmas_noise_grid_org.shape, np.nan)
W_btst = np.full((nBtst,) + W_org.shape, np.nan)

# For validation metrics stored per bootstrap replicate:
#   - L2norm_thres_Wishart_btst_list: predicted threshold vector lengths at MOCS validation conditions
#   - NBS_sum: scalar similarity score used to rank replicates (higher = more similar)
L2norm_thres_Wishart_btst_list = []
NBS_sum = np.full((nBtst,), np.nan)

#default is the expt did include MOCS trials, but in some future versions of
#the expts, we didn't include MOCS trials
flag_no_MOCS = False 

for n in trange(nBtst, desc="Loading bootstraps"):
    # Swap out the bootstrap index inside "AEPsych[...]" while keeping the rest of the filename intact
    fname_n = re.sub(r"AEPsych\[\d+\]", f"AEPsych[{n}]", btst_file_name)
    full_btst_file_n = os.path.join(btst_fileDir_fits, fname_n)

    with open(full_btst_file_n, "rb") as f:
        vars_dict_n = pickled.load(f)

    # Fitted model object for this bootstrap replicate
    model_pred_n = deepcopy(vars_dict_n["model_pred_Wishart"])

    # Save best-fit W and covariance predictions for this bootstrap
    W_btst[n] = np.asarray(model_pred_n.W_est)
    Sigmas_thres_grid_btst[n] = np.asarray(model_pred_n.Sigmas_thres_grid)
    Sigmas_noise_grid_btst[n] = np.asarray(vars_dict_n["Sigmas_noise_grid_btst"])
    
    # Save predicted threshold magnitudes at the MOCS validation conditions
    try:
        L2norm_thres_Wishart_btst_list.append(vars_dict_n['thres_Wishart_based_atMOCS']['vecLen_at_targetPC_Wishart'])
    except:
        flag_no_MOCS = True
        if n == 0: print('Cannot find MOCS data.')

    # Compute a scalar NBS similarity score for ranking (sum over the fine-grid NBS map)
    NBS_grid = np.asarray(vars_dict_n["NBS_fine_grid"])
    NBS_sum[n] = float(NBS_grid.sum())

# Rank bootstrap replicates by descending NBS similarity (best match first)
idx_desc = np.argsort(NBS_sum)[::-1]  

# Reorder bootstrap outputs according to this ranking
Sigmas_thres_sorted = Sigmas_thres_grid_btst[idx_desc]
Sigmas_noise_sorted = Sigmas_noise_grid_btst[idx_desc]
W_sorted = W_btst[idx_desc]

# Convert list -> array and reorder predicted thresholds consistently
if not flag_no_MOCS:
    L2norm_thres_Wishart_btst = np.asarray(L2norm_thres_Wishart_btst_list)

#----------------------------------------------------------------------------------------------
# Append bootstrap columns to the existing CSVs (original columns stay; new columns are added)
# Note: we pass both the sorted arrays and the idx_desc mapping so column names can encode
#       the bootstrap index and/or rank consistently.
DataExport.append_bootstrap_cov_columns(Sigmas_thres_sorted, 
                                        idx_desc,
                                        path_csv_thres,
                                        prefix="thres", 
                                        )

# Noise ellipses (fine grid: 103x103)
DataExport.append_bootstrap_cov_columns(Sigmas_noise_sorted,
                                        idx_desc,
                                        path_csv_noise, 
                                        prefix="noise", 
                                        )

#weight matrix
DataExport.append_bootstrap_weights_columns(W_sorted, 
                                            idx_desc,
                                            path_csv_weights
                                            )

#%% 
# -----------------------------------------------------------
# SECTION 4: Load MOCS data
# -----------------------------------------------------------
# Load the fitted psychometric-function (PMF) results for the MOCS validation trials
#   ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits/
#   Fitted_weibull_psychometric_func_Isoluminant plane_6000totalTrials_25refs_MOCS_sub1.pkl
if not flag_no_MOCS:
    MOCS_fileDir, MOCS_file_name = select_file_and_get_path()
    full_path_MOCS = os.path.join(MOCS_fileDir, MOCS_file_name)
    
    with open(full_path_MOCS, "rb") as f:
        vars_dict_MOCS = pickled.load(f)
    
    # Unique references used for validation (one per condition)
    xref_unique_MOCS = vars_dict_MOCS["xref_unique"]   # (nRefs_MOCS, 2), usually (25, 2)
    nTrials_MOCS = vars_dict_MOCS["nTrials"]           # total scheduled trials across all refs
    nRefs_MOCS   = vars_dict_MOCS["nRefs"]             # number of validation/reference conditions
    nLevels_MOCS = vars_dict_MOCS["nLevels"]           # number of comparison levels per reference
    nTrials_perRef   = nTrials_MOCS // nRefs_MOCS      # trials per reference condition
    nTrials_perLevel = nTrials_perRef // nLevels_MOCS  # trials per comparison level
    
    # We rebuild trial-level (xref, x1, y) arrays in the same coordinate space as the
    # main experiment data, by converting PMF “delta” stimuli back into absolute stimuli:
    #   x1 = xref + stim_delta
    xref_MOCS_list = []
    x1_MOCS_list   = []
    y_MOCS_list    = []
    cmap_allref    = []
    
    for n in range(nRefs_MOCS):
        # PMF-fit container for reference condition n
        fit_PMF_n = deepcopy(vars_dict_MOCS["fit_PMF_MOCS"][n])
    
        # Stimulus coordinates used for PMF fitting.
        # In this pipeline, these are typically deltas relative to the reference:
        #   stim_delta ≈ (comp - ref), shape: (nTrials_perRef, 2)
        stim_delta = fit_PMF_n.stim_org 
        
        # Repeat the reference coordinate to match the per-trial length for this condition
        xref_rep = np.tile(xref_unique_MOCS[n][None], (nTrials_perRef,1)) #(240, 2)
    
        # During PMF fitting, a synthetic [0, 0] stimulus delta may have been appended
        # (commonly to anchor the guess rate or stabilize the fit). That row is not a real
        # trial, so we drop it when reconstructing trial-level arrays.
        keep = np.any(stim_delta != 0, axis=1)  
    
        # Convert delta stimuli back to absolute comparison stimuli in the original space
        x1_keep = xref_rep + stim_delta[keep]
    
        # Binary responses aligned with stim_org (drop the synthetic row consistently)
        y_keep = fit_PMF_n.resp_org[keep]
    
        # Accumulate per-condition trial arrays
        xref_MOCS_list.append(xref_rep)
        x1_MOCS_list.append(x1_keep)
        y_MOCS_list.append(y_keep)
        
        # Store an RGB color for plotting this reference condition (for consistent colormaps)
        cm = color_thres_data.W2D_to_rgb(vars_dict_MOCS['xref_unique'][n])
        cmap_allref.append(cm)
        
        # Optional debug: visualize psychometric sampling for this reference
        # plt.scatter(np.linalg.norm(stim_delta[keep], axis=1), y_keep, alpha=0.05)
        # plt.show()
    
    # Concatenate trials across all reference conditions into flat arrays
    xref_MOCS = np.concatenate(xref_MOCS_list)
    x1_MOCS = np.concatenate(x1_MOCS_list)
    y_MOCS = np.concatenate(y_MOCS_list)
    
    # Sanity checks
    assert xref_MOCS.shape[0] == x1_MOCS.shape[0] == y_MOCS.shape[0], "Length mismatch across fields."
    assert xref_MOCS.shape[0] == nTrials_MOCS, (
        f"Unexpected trial count after removing injected [0,0] rows: "
        f"got {xref_MOCS.shape[0]}, expected {nTrials_MOCS}."
    )

#%%
# -----------------------------------------------------------
# SECTION 5: Export pooled trial-level data with labeled trial types
# -----------------------------------------------------------
# Planned AEPsych design: [Sobol_small, Sobol_medium, Sobol_large, Adaptive]
nTrials_strat = [300, 300, 300, 6600] #5100

# Intended number of AEPsych trials (what the design asked for)
nTrials_AEPsych = sum(nTrials_strat)

# Intended number of Sobol trials within the AEPsych design (excluding adaptive block)
nTrials_AEPsych_sobol = sum(nTrials_strat[:-1])

# Actual number of trials recorded in the main experiment run (AEPsych + any inserted trials)
nTrials_actual = expt_trial.xref_all.shape[0]

# Extra pre-generated Sobol trials inserted beyond the intended AEPsych design
nTrials_pregenSobol = nTrials_actual - nTrials_AEPsych

# Build TrialType labels (one label per row in the pooled arrays)
# AEPsych (+ inserted pregen Sobol) labels are built to align with the *main-run* trial order.
# The helper returns:
#   - trial_type_ae: list[str] of length nTrials_actual
#   - (possibly updated) counts for bookkeeping / printing
trial_type_ae, nTrials_AEPsych, nTrials_AEPsych_sobol, nTrials_pregenSobol = \
    DataExport.build_trial_type_ae(nTrials_strat, nTrials_actual)


# Combine main-run trials with MOCS validation trials
# Stack references/comparisons and append responses to form one pooled dataset.
if not flag_no_MOCS:
    xref = np.vstack((expt_trial.xref_all, xref_MOCS))
    x1   = np.vstack((expt_trial.x1_all,   x1_MOCS))
    y    = np.concatenate((expt_trial.y_all, y_MOCS), axis=0)
    
    # MOCS labels align with the reconstructed MOCS trial order (after dropping synthetic rows)
    trial_type_mocs = DataExport.build_trial_type_mocs(
        nRefs_MOCS, nLevels_MOCS, nTrials_perLevel
    )
    
    # Concatenate TrialType strings to match the pooled (main-run + MOCS) arrays
    trial_type_str = trial_type_ae + trial_type_mocs
else:
    xref = expt_trial.xref_all
    x1 = expt_trial.x1_all
    y = expt_trial.y_all
    
    trial_type_str = trial_type_ae

# Sanity checks: all pooled arrays must have identical length
N = xref.shape[0]
if not (x1.shape[0] == N and y.shape[0] == N and len(trial_type_str) == N):
    raise ValueError(
        f"Length mismatch: xref={N}, x1={x1.shape[0]}, y={y.shape[0]}, "
        f"TrialType={len(trial_type_str)}"
    )

# Serialize vectors into single CSV columns
# Store xref and x1 as stringified vectors so each trial occupies one CSV row.
# (This avoids expanding into multiple coordinate columns and keeps the export format consistent.)
xref_str = DataExport.vec_to_str(xref)
x1_str   = DataExport.vec_to_str(x1)

# Export pooled trial table
df = pd.DataFrame({
    "TrialType": trial_type_str,   # categorical label describing trial provenance
    "xref": xref_str,              # reference stimulus coordinate (string)
    "x1": x1_str,                  # comparison stimulus coordinate (string)
    "y": y.astype(int),            # binary response
})

out_path = os.path.join(output_dir, f"trial_data_pooled_by_type_sub{subN}.csv")
df.to_csv(out_path, index=False)

#%%
# -----------------------------------------------------------
# SECTION 6: Export threshold summaries at the 25 MOCS validation references
# -----------------------------------------------------------
# We export two kinds of threshold readouts at the same set of validation references:
#   (A) Validation (MOCS) thresholds:
#       Thresholds estimated by fitting a psychometric function (PMF) to the MOCS
#       binary responses at each reference condition.
#   (B) WPPM/Wishart-predicted thresholds:
#       Thresholds read out directly from the fitted Wishart-process model at the
#       same references (model prediction, not refit PMF).

if not flag_no_MOCS:
    # Model-predicted threshold comparison stimuli at the 25 validation refs, shape: (25, 2)
    x1_thres_Wishart_org = vars_dict_MOCS["stim_at_targetPC_Wishart"]
    
    # Euclidean threshold magnitude per ref (distance from xref to predicted threshold comp), shape: (25,)
    L2norm_thres_Wishart_org = vars_dict_MOCS["vecLen_at_targetPC_Wishart"]
    
    # PMF-derived threshold comparison stimuli at each ref, shape: (25, 2)
    x1_thres_validation_org = vars_dict_MOCS["stim_at_targetPC_MOCS"]
    
    # PMF-derived Euclidean threshold magnitude per ref, shape: (25,)
    L2norm_thres_validation_org = vars_dict_MOCS["vecLen_at_targetPC_MOCS"]
    
    # Bootstrap PMF thresholds:
    # Stored as (nBtst, nRefs) or (nRefs, nBtst) depending on upstream code.
    # Here we transpose to get shape (nBtst, nRefs) so each bootstrap gives a length-25 vector.
    L2norm_thres_validation_btst = vars_dict_MOCS["vecLen_at_targetPC_MOCS_btst"].T
    
    # Validation-threshold table (PMF-based)
    df_vthres = pd.DataFrame({
        "xref": DataExport.vec_to_str(xref_unique_MOCS),
        "x1_thres_org": DataExport.vec_to_str(x1_thres_validation_org),
        "L2norm_thres_org": np.round(L2norm_thres_validation_org, 8),
    })
    
    # WPPM/Wishart-threshold table (model-predicted)
    df_wthres = pd.DataFrame({
        "xref": DataExport.vec_to_str(xref_unique_MOCS),
        "x1_thres_org": DataExport.vec_to_str(x1_thres_Wishart_org),
        "L2norm_thres_org": np.round(L2norm_thres_Wishart_org, 8),
    })
    
    # Validation bootstrap columns: one column per bootstrap replicate.
    # Column names encode the bootstrap id (0..nBtst-1).
    v_cols = {
        f"L2norm_thres_btst{b}": L2norm_thres_validation_btst[b]
        for b in range(nBtst)
    }
    df_vthres = pd.concat([df_vthres, pd.DataFrame(v_cols)], axis=1)
    
    # Wishart bootstrap columns: we also encode the *rank* of each bootstrap replicate
    # based on NBS similarity (computed earlier).
    #
    # idx_desc is an ordering of bootstrap ids from best -> worst:
    #   idx_desc[0] = best bootstrap id, idx_desc[1] = second best, ...
    #
    # Build an inverse mapping so we can label each bootstrap id with its rank.
    rank_of = np.empty(nBtst, dtype=int)
    rank_of[idx_desc] = np.arange(nBtst)  # rank_of[btst_id] = rank (0 = best)
    
    # L2norm_thres_Wishart_btst is assumed to be shape (nBtst, nRefs) or (nBtst, 25),
    # where each entry is the model-predicted threshold magnitude for that bootstrap.
    if not flag_no_MOCS:
        w_cols = {
            f"L2norm_thres_btst{b}_rank{rank_of[b]}": L2norm_thres_Wishart_btst[b]
            for b in range(nBtst)
        }
        df_wthres = pd.concat([df_wthres, pd.DataFrame(w_cols)], axis=1)
        
        # Write to disk
        out_path_v = os.path.join(output_dir, f"thres_at_validation_conditions_sub{subN}.csv")
        out_path_w = os.path.join(output_dir, f"thres_at_validation_conditions_WPPM_sub{subN}.csv")
        
        df_vthres.to_csv(out_path_v, index=False, float_format="%.8f")
        df_wthres.to_csv(out_path_w, index=False, float_format="%.8f")

#%% 
# --------------------------------------------------------------------------
# SECTION 7: Debugging plot to make sure the data were exported properly
# --------------------------------------------------------------------------
flag_debug_plot = True
if flag_debug_plot:
    # ---------------------------- Different trial types ----------------------------
    # Create settings instance with custom fig_dir
    pltSettings_base = PlotSettingsBase(fig_dir=out_path, fontsize = 8)
    pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
    sampling_vis = SamplingRefCompPairVisualization(2,
                                                    color_thres_data,
                                                    settings = pltSettings_tp,
                                                    save_fig = False
                                                    )
    
    # This array defines the opacity of markers in the plots, decreasing with more trials.
    marker_alpha = [0.5, 0.3, 0.5, 0.3]
    slc_datapoints_to_show_lb = [0, 
                                 nTrials_AEPsych_sobol, 
                                 nTrials_AEPsych]
    slc_datapoints_to_show_ub = [nTrials_AEPsych_sobol, 
                                 nTrials_AEPsych, 
                                 nTrials_AEPsych + nTrials_pregenSobol]
    if not flag_no_MOCS:
        slc_datapoints_to_show_lb.append(nTrials_AEPsych + nTrials_pregenSobol)
        slc_datapoints_to_show_ub.append(nTrials_AEPsych + nTrials_MOCS + nTrials_pregenSobol)
    
    # Loop over the selected data points to generate and visualize each corresponding figure.
    for i, (lb_i, ub_i) in enumerate(zip(slc_datapoints_to_show_lb, slc_datapoints_to_show_ub)):
        pltSettings_tp = replace(pltSettings_tp,
                                 ref_markeralpha = marker_alpha[i],
                                 comp_markeralpha = marker_alpha[i],
                                 linealpha = marker_alpha[i], 
                                 ticks = np.linspace(-0.7, 0.7, 5),
                                 bounds = 0.75 * np.array([-1,1]),
                                 )
    
        fig, ax = plt.subplots(1, 1, figsize = (3,3.5), dpi= pltSettings_tp.dpi)
        # Visualize the trials up to the nth data point with specified marker transparency.
        sampling_vis.plot_sampling(xref[lb_i:ub_i],  # Reference points up to the nth data point
                                   x1[lb_i:ub_i],    # Comparison points up to the nth data point
                                   settings = pltSettings_tp,
                                   ax = ax
                                   )            
        ax.set_title(color_thres_data.plane_2D)
        plt.show()
        
    # ---------------------------- noise / thres ellipses ----------------------------
    fig2, ax2 = plt.subplots(1, 1, figsize = (3,3.5), dpi= pltSettings_tp.dpi)
    pltSettings_tp = replace(pltSettings_tp, visualize_bounds = False)
    sampling_vis.plot_sampling(np.empty((1,2)),  # Reference points up to the nth data point
                               np.empty((1,2)),    # Comparison points up to the nth data point
                               settings = pltSettings_tp,
                               ax = ax2
                               ) 
    for n in range(grid_flat.shape[0]):
        cmap_n = color_thres_data.W2D_to_rgb(grid_flat[n])
        viz.plot_ellipse(ax2, grid_flat[n], Sigmas_thres_flat_org[n],
                         color = cmap_n, ls = ':', lw = 0.75)
        
        idx_match = np.where(np.all(np.abs(grid_fine_flat - grid_flat[n]) < 1e-5, axis=1))[0]
        viz.plot_ellipse(ax2, grid_flat[n], Sigmas_noise_flat_org[idx_match.item()],
                         color = cmap_n, ls = '-', lw = 1)
    
    # ---------------------------- linear regresshion ----------------------------
    # here is hard-coded. We have 120 bootstrap cond
    idx_desc_keep = idx_desc[:int(nBtst * 0.95)]
    #Euclidean distan between the ref and comp at the thres predicted by the Wishart model
    L2norm_thres_Wishart_btst_within_CI = L2norm_thres_Wishart_btst[idx_desc_keep]
    
    #sort Euclidean distance and take the CI bounds by index
    L2norm_thres_Wishart_sorted = np.sort(L2norm_thres_Wishart_btst_within_CI, axis = 0)
    L2norm_thres_Wishart_CI_bds = L2norm_thres_Wishart_sorted[[0,-1]]
    
    # Convert CI bounds into asymmetric error bars relative to the point estimate
    # (matplotlib expects [lower_err, upper_err]).
    # Shape: (nCond, 2)
    L2norm_thres_Wishart_CI_err = np.vstack((
        vars_dict_MOCS['vecLen_at_targetPC_Wishart'] - L2norm_thres_Wishart_CI_bds[0],
        L2norm_thres_Wishart_CI_bds[1] - vars_dict_MOCS['vecLen_at_targetPC_Wishart']
    )).T
    
    slope_corr_dict = vars_dict_MOCS['slope_corr_analysis_matched_btst']

    #plotting
    if not flag_no_MOCS:
        vis_MOCS = MOCSTrialsVisualization(vars_dict_MOCS['fit_PMF_MOCS'], 
                                           settings = pltSettings_tp,
                                           save_fig= False)
        plt_st_n = plt_st[f'sub{subN}']
        predComp_settings = replace(PlotThresCompSettings(), **pltSettings_base.__dict__)
        predComp_settings = replace(predComp_settings,
                                    fontsize = 9.5,
                                    ms = 6,
                                    fig_size = (4.8, 5), 
                                    alpha = 0.8,
                                    lw = 1.5,
                                    bds = plt_st_n['bds'], 
                                    xlabel = 'Threshold distance (validation)',
                                    ylabel = 'Threshold distance (WPPM)',
                                    cmap = cmap_allref,
                                    )
        # plot the comparison of thresholds between AEPsych predictions and MOCS predictions
        vis_MOCS.plot_comparison_thres(thres_Wishart = L2norm_thres_Wishart_org,
                                       slope_org = vars_dict_MOCS['slope_corr_analysis_matched_btst']['slope_modelPred_org'].item(),
                                       slope_CI= slope_corr_dict['slope_btst_CI'],
                                       xref_unique = xref_unique_MOCS,
                                       thres_Wishart_95btstErr = L2norm_thres_Wishart_CI_err,
                                       settings = predComp_settings)
