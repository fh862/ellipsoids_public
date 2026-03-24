#!/usr/bin/env python3
"""
Created on Wed Jan 15 11:37:06 2025

@author: fangfang

This script is organized into five main sections:

1. Load the .pkl file containing psychometric function (PMF) fits to the
   validation MOCS trials.

2. Load the .pkl file containing WPPM-predicted threshold ellipses at the
   25 validation reference conditions, based on the fit to the original
   (non-bootstrapped) dataset.
   If these predictions have already been computed, they are loaded directly;
   otherwise, they are computed and appended to the file.

3. Load the .pkl file containing WPPM-predicted threshold ellipses at the same
   validation conditions, but based on fits to bootstrap datasets (N = 120).
   As above, predictions are loaded if available; otherwise, they are computed
   and appended to the corresponding file.

4. Compute 95% bootstrap confidence intervals for:
   - The WPPM-predicted threshold ellipses, and
   - The Euclidean distance between the reference and comparison stimulus
     at threshold along each validation chromatic direction.

5. Visualization:
   5.1. Plot WPPM-predicted threshold ellipses at the validation reference
        locations, overlaid with the validation stimulus directions.
   5.2. Plot the fitted PMFs for the validation data together with the
        WPPM-predicted percent-correct curves along the same chromatic
        directions.
   5.3. Plot and analyze the linear relationship between WPPM-predicted
        thresholds and empirically measured validation thresholds.

If this is run on hpc, use runPython_wPytorch.sbatch

#!/bin/bash
#SBATCH --job-name=cross_validate_find_opt_decay_rate
#SBATCH --output=slurm_scripts/slurm%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=fh862@sas.upenn.edu
#SBATCH -p gpu -N1 -G1 --constraint=h100 --cpus-per-task=4 --mem-per-cpu=20G
#SBATCH --time=00:30:00

"""

# Toggle between HPC batch mode and local/interactive mode.
flag_running_on_hpc = False
import jax

jax.config.update("jax_enable_x64", True)
import os
import re
import sys
from copy import deepcopy
from dataclasses import replace

import dill as pickled
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from analysis.conf_interval import (
    find_btst_dataset_within_CI,
    find_inner_outer_contours_for_gridRefs,
    intervals_overlap,
)
from analysis.MOCS_thresholds import compute_Wishart_based_pCorrect_atMOCS
from core.model_predictions import rerun_model_pred_wExisting_model
from plotting.visualize_MOCS import (
    MOCSConditionsVisualization,
    MOCSTrialsVisualization,
    PlotCondSettings,
    PlotPMFSettings,
    PlotThresCompSettings,
)
from plotting.visualize_MOCS import PlotThresCompSettings_bds as plt_st
from plotting.wishart_plotting import PlotSettingsBase
from plotting.wishart_predictions_plotting import (
    Plot2DPredSettings,
    WishartPredictionsVisualization,
    add_CI_ellipses,
)

# %%
# ---------------------------------------------------------------------------
# SECTION 1: load the fitted psychometric functions to validation trials
# --------------------------------------------------------------------------
# Base directory where data lives. On HPC, prefer paths relative to the script.
base_dir = (
    os.path.dirname(__file__) if flag_running_on_hpc else "/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/"
)

# load best-fit Weibull Psychometric functions
#'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits'
#'Fitted_weibull_psychometric_func_Isoluminant plane_6000totalTrials_25refs_MOCS_sub#.pkl'

# OR
#'ELPS_analysis/Simulation_DataFiles/MOCS/gt_CIE'
#'Fitted_weibull_psychometric_func_Isoluminant plane_240totalTrials_25refs_MOCS_subCIE1994.pkl'
flag_load_data = False
subN = 1

if flag_load_data:
    input_fileDir_fits_MOCS = os.path.join(
        base_dir,
        "ELPS_analysis",
        "Experiment_DataFiles",
        "pilot2",
        f"sub{subN}",
        "fits",
    )
    file_name_MOCS = (
        "Fitted_weibull_psychometric_func_Isoluminant plane_"
        + f"6000totalTrials_25refs_MOCS_sub{subN}_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl"
    )
else:
    input_fileDir_fits_MOCS = os.path.join(base_dir, "ELPS_analysis", "Simulation_DataFiles", "MOCS", "gt_CIE")
    file_name_MOCS = (
        "Fitted_weibull_psychometric_func_Isoluminant plane_"
        + "240totalTrials_25refs_MOCS_subCIE1994_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl"
    )

# Construct the full path to the selected file
full_path_MOCS = os.path.join(input_fileDir_fits_MOCS, file_name_MOCS)

# Load the necessary variables from the file
with open(full_path_MOCS, "rb") as f:
    MOCS = pickled.load(f)

# %%
# ---------------------------------------------------------------------------
# SECTION 2: load the Wishart model predictions
# --------------------------------------------------------------------------
# Navigate to the directory: ELPS_analysis/Experiment_DataFiles/sub#
# 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.4_nBasisDeg5.pkl'

# OR
#'META_analysis/ModelFitting_DataFiles/4dTask/CIE/sub1/decayRate0.5'
#'Fitted_byWishart_Isoluminant plane_4DExpt_300_300_300_5100_AEPsychSampling_EAVC_decayRate0.5_nBasisDeg5_sub1.pkl'

if flag_load_data:
    input_fileDir_fits = os.path.join(
        base_dir,
        "ELPS_analysis",
        "Experiment_DataFiles",
        "pilot2",
        f"sub{subN}",
        "fits",
    )
    file_name = (
        "Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_"
        + f"sub{subN}_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl"
    )
else:
    input_fileDir_fits = os.path.join(
        base_dir,
        "META_analysis",
        "ModelFitting_DataFiles",
        "4dTask",
        "CIE",
        f"sub{subN}",
    )
    file_name = (
        "Fitted_Sim4dTask_colorDiscrimination_EAVC_6000Trials_"
        + f"300_300_300_5100_sub{subN}_gtCIE1994_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl"
    )
decay_rate = float(re.search(r"decayRate([0-9.]+)", file_name).group(1))

# Construct the full path to the selected file
full_path = os.path.join(input_fileDir_fits, file_name)

# Load the necessary variables from the file
with open(full_path, "rb") as f:
    vars_dict = pickled.load(f)

# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict["color_thres_data"]
# - Experimental trial data
expt_trial = vars_dict["expt_trial"]

key_gridMOCS = "model_pred_Wishart_MOCS"
flag_append_data = True

if key_gridMOCS in vars_dict:
    model_pred_Wishart_MOCS = vars_dict[key_gridMOCS]
else:
    # Retrieve the variables of interest from the loaded dictionary
    # - Model predictions using the Wishart process
    model_pred_existing = deepcopy(vars_dict["model_pred_Wishart"])
    grid_MOCS = MOCS["xref_unique"]

    # Use the helper function to recompute model predictions and transformed grid
    model_pred_Wishart_MOCS, _ = rerun_model_pred_wExisting_model(
        grid_MOCS[None], model_pred_existing, color_thres_data
    )
    # Optionally append results to the pickle
    if flag_append_data:
        vars_dict[key_gridMOCS] = model_pred_Wishart_MOCS
        vars_dict["grid_MOCS"] = grid_MOCS

        # Save the updated pickle file
        with open(full_path, "wb") as f:
            pickled.dump(vars_dict, f)

# %%
# ---------------------------------------------------------------------------
# SECTION 3: Retrieve model predictions from each bootstrapped dataset
# --------------------------------------------------------------------------
# Step 1: Select the bootstrapped data file (choose one as a reference)
#'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits/AEPsych_btst/decayRate0.4'
#'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.4_nBasisDeg5_btst_AEPsych[0].pkl'

# OR
#'META_analysis/ModelFitting_DataFiles/4dTask/CIE/sub1/decayRate0.4'
#'Fitted_byWishart_Isoluminant plane_4DExpt_300_300_300_5100_AEPsychSampling_EAVC_decayRate0.4_nBasisDeg5_sub1_btst_AEPsych[0].pkl'  # noqa: E501

input_fileDir_fits_btst = os.path.join(input_fileDir_fits, "AEPsych_btst", f"decayRate{decay_rate}")
file_name_btst = f"{file_name[:-4]}_btst_AEPsych[0].pkl"

# number of bootstrapped datasets: AEPsych[0] ... AEPsych[119]
nBtst = 120

# Storage
# params_all: ellipse parameters at each MOCS reference, for each bootstrap
#   params = [x_center, y_center, major_axis, minor_axis, rotation_deg]
params_all = np.full((MOCS["nRefs"], nBtst, 5), np.nan)

# pChoosingX1_Wishart_list: one entry per bootstrap (each entry is typically length nRefs)
# stores WPPM-predicted p(correct) evaluated at MOCS validation conditions
pChoosingX1_Wishart_list = []
vecLen_at_targetPC_Wishart_list = []

# NBS_sum: dataset-level similarity score between each bootstrap fit and the original fit
# computed by summing the fine-grid NBS values (higher = more similar)
# used to rank bootstraps and to drop the bottom 5% when forming a 95% CI
NBS_sum = np.full((nBtst,), np.nan)

# --------------------------------------------------------------------------
# Loop over bootstraps
# --------------------------------------------------------------------------
for r in trange(nBtst, desc="Loading bootstraps"):
    # File for bootstrap r (swap AEPsych[0] -> AEPsych[r])
    file_name_btst_r = file_name_btst.replace("AEPsych[0]", f"AEPsych[{r}]")
    full_path_btst_r = f"{input_fileDir_fits_btst}/{file_name_btst_r}"

    with open(full_path_btst_r, "rb") as f:
        vars_dict_btst = pickled.load(f)

    # If we already cached the MOCS-grid Wishart predictions in this pickle, reuse them
    if key_gridMOCS in vars_dict_btst:
        model_pred_Wishart_MOCS_r = deepcopy(vars_dict_btst[key_gridMOCS])
        thres_Wishart_based_atMOCS_r = vars_dict_btst["thres_Wishart_based_atMOCS"]

    else:
        # Otherwise, recompute Wishart predictions at the MOCS validation references
        model_pred_r = vars_dict_btst["model_pred_Wishart"]
        grid_MOCS_r = MOCS["xref_unique"]  # unique MOCS reference locations

        # Re-run prediction using the existing fitted model, evaluated at MOCS grid
        # (also returns transformed grid in other spaces if needed; ignored here)
        model_pred_Wishart_MOCS_r, _ = rerun_model_pred_wExisting_model(
            grid_MOCS_r[None], model_pred_r, color_thres_data
        )

        # Monte Carlo simulation: generate WPPM-predicted psychometric curves at MOCS conditions
        # Output includes (1) `pChoosingX1_Wishart`
        #                 (2) `vecLen_at_targetPC_Wishart`
        #                 (3) `stim_at_targetPC_Wishart`
        thres_Wishart_based_atMOCS_r = compute_Wishart_based_pCorrect_atMOCS(
            nBtst,
            MOCS["nLevels"],
            MOCS["fit_PMF_MOCS"],
            MOCS["xref_unique"],
            deepcopy(model_pred_Wishart_MOCS_r),
            color_thres_data,
        )

        # Optionally cache computed results back into this bootstrap pickle (speeds up reruns)
        if flag_append_data:
            vars_dict_btst[key_gridMOCS] = model_pred_Wishart_MOCS_r
            vars_dict_btst["grid_MOCS"] = grid_MOCS_r
            vars_dict_btst["thres_Wishart_based_atMOCS"] = thres_Wishart_based_atMOCS_r

            with open(full_path_btst_r, "wb") as f:
                pickled.dump(vars_dict_btst, f)

    # Collect WPPM-predicted percent-correct at validation conditions for this bootstrap
    # (used later to form CI across bootstraps)
    pChoosingX1_Wishart_list.append(thres_Wishart_based_atMOCS_r["pChoosingX1_Wishart"])
    vecLen_at_targetPC_Wishart_list.append(thres_Wishart_based_atMOCS_r["vecLen_at_targetPC_Wishart"])

    # Aggregate similarity score (higher means closer to original dataset fit)
    # vars_dict_btst["NBS_fine_grid"] is NBS per fine-grid point; sum -> one score per dataset
    NBS_sum[r] = np.sum(vars_dict_btst["NBS_fine_grid"])

    # Extract ellipse parameters at each MOCS reference for this bootstrap
    for j in range(MOCS["nRefs"]):
        params_all[j, r] = model_pred_Wishart_MOCS_r.params_ell[0][j]

# %%
if not flag_running_on_hpc:
    # --------------------------------------------------------------------------
    # SECTION 4a: Rank bootstraps by similarity (NBS) and keep the top 95% for CI
    # --------------------------------------------------------------------------
    # fit a linear regression to the original dataset
    slope_modelPred_org, *_ = np.linalg.lstsq(
        MOCS["vecLen_at_targetPC_MOCS"][:, None],
        MOCS["vecLen_at_targetPC_Wishart"],
        rcond=None,
    )
    corr_coef_modelPred_org = np.corrcoef(MOCS["vecLen_at_targetPC_MOCS"], MOCS["vecLen_at_targetPC_Wishart"])[0, 1]

    # the confidence interval is the central 95%
    percent_CI = 0.95

    # Ellipse parameters restricted to the retained bootstrap set
    # Expected output shape should be (nDatasets_CI, nRefs, 5)
    params_ell_within_CI, idx_keep_NBS, _ = find_btst_dataset_within_CI(
        NBS_sum, np.moveaxis(params_all, 0, 1), CI_percent=percent_CI
    )

    # For visualization: compute inner/outer envelope contours across bootstrap ellipses
    # aggregated over all reference locations.
    fitEll_min, fitEll_max = find_inner_outer_contours_for_gridRefs(np.moveaxis(params_ell_within_CI, 0, 1))

    # Same filtering for pChoosingX1_Wishart (align bootstraps with idx_keep_NBS)
    pChoosingX1_Wishart_btst = np.asarray(pChoosingX1_Wishart_list)  # shape (nBtst, nRefs, 1200)
    pChoosingX1_Wishart_within_CI = pChoosingX1_Wishart_btst[idx_keep_NBS]  # shape (nDatasets_CI, nRefs, 1200)

    # sort across bootstrap fits, then take the CI bounds by index.
    pChoosingX1_Wishart_sorted = np.sort(pChoosingX1_Wishart_within_CI, axis=0)
    # take the first and the last element as the CI bounds since this is already trimmed datasets
    pChoosingX1_Wishart_CI_bds = pChoosingX1_Wishart_sorted[[0, -1]]

    # Euclidean distan between the ref and comp at the thres predicted by the Wishart model
    vecLen_at_targetPC_Wishart_btst = np.asarray(vecLen_at_targetPC_Wishart_list)
    vecLen_at_targetPC_Wishart_within_CI = vecLen_at_targetPC_Wishart_btst[idx_keep_NBS]

    # sort Euclidean distance and take the CI bounds by index
    vecLen_at_targetPC_Wishart_sorted = np.sort(vecLen_at_targetPC_Wishart_within_CI, axis=0)
    vecLen_at_targetPC_Wishart_CI_bds = vecLen_at_targetPC_Wishart_sorted[[0, -1]]

    # Convert CI bounds into asymmetric error bars relative to the point estimate
    # (matplotlib expects [lower_err, upper_err]).
    # Shape: (nCond, 2)
    vecLen_at_targetPC_Wishart_err = np.vstack(
        (
            MOCS["vecLen_at_targetPC_Wishart"] - vecLen_at_targetPC_Wishart_CI_bds[0],
            vecLen_at_targetPC_Wishart_CI_bds[1] - MOCS["vecLen_at_targetPC_Wishart"],
        )
    ).T

    # -----------------------------------------------------------------------------
    # SECTION 4b: compute the linear regression by pairing up bootstrap model fit
    # between validation and AEPsych trials
    # -----------------------------------------------------------------------------
    # Here we pair each empirical bootstrap sample (MOCS threshold estimates) with the
    # corresponding WPPM bootstrap fit (ellipse-based predictions) and compute:
    #   - slope (no intercept) of predicted vs. empirical
    #   - Pearson correlation coefficient
    # Then we form 95% CIs across matched bootstrap samples.

    key_matched_btst_analysis = "slope_corr_analysis_matched_btst"

    # Variables to cache inside MOCS so we don’t recompute on reruns
    var_names = [
        "slope_modelPred_org",
        "corr_coef_modelPred_org",
        "slope_btst",
        "corr_coef_btst",
        "slope_btst_sorted",
        "slope_btst_CI",
        "corr_coef_btst_sorted",
        "corr_coef_btst_CI",
    ]

    # If analysis already exists in MOCS, load all saved variables
    if key_matched_btst_analysis in MOCS:
        for name in var_names:
            globals()[name] = MOCS[key_matched_btst_analysis][name]
    else:
        # Initialize arrays to store the slope and correlation for each matched bootstrap sample
        slope_btst = np.full((nBtst,), np.nan)
        corr_coef_btst = np.full((nBtst,), np.nan)

        # Note that we should loop through `nBtst` instead of `nDatasets_CI` datasets!
        for n in range(nBtst):
            # Get the n-th MOCS bootstrap sample (empirical thresholds)
            data_MOCS_n = MOCS["vecLen_at_targetPC_MOCS_btst"][:, n]

            # Get the corresponding AEPsych-predicted thresholds for this bootstrap sample
            data_model_pred = vecLen_at_targetPC_Wishart_btst[n]

            # Perform linear regression (without intercept) to compute slope of predicted vs. empirical
            slope_modelPred_n, *_ = np.linalg.lstsq(data_MOCS_n.reshape(-1, 1), data_model_pred, rcond=None)
            slope_btst[n] = slope_modelPred_n.item()

            # Compute Pearson correlation coefficient between predicted and empirical thresholds
            corr_coef_btst[n] = np.corrcoef(data_MOCS_n, data_model_pred)[0, 1]

        # Aggregate slope results
        nDatasets_CI = int(nBtst * percent_CI)  # 114
        idx_bds_CI = [
            int(nBtst * 0.025),
            int(nBtst * 0.975) - 1,
        ]  # # Upper bound index (convert to 0-based indexing)

        slope_btst_sorted = np.sort(slope_btst)
        slope_btst_CI = slope_btst_sorted[idx_bds_CI]

        # Aggregate correlation coefficient results
        corr_coef_btst_sorted = np.sort(corr_coef_btst)
        corr_coef_btst_CI = corr_coef_btst_sorted[idx_bds_CI]

        if flag_append_data:
            # Package all computed variables into a dictionary for saving
            slope_corr_analysis_matched_btst = {name: eval(name) for name in var_names}
            MOCS[key_matched_btst_analysis] = slope_corr_analysis_matched_btst

            # Save updated MOCS dictionary back to file
            with open(full_path_MOCS, "wb") as f:
                pickled.dump(MOCS, f)

# %%
if not flag_running_on_hpc:
    # Set up output directory for saving figures
    output_figDir_fits = os.path.join(
        input_fileDir_fits.replace("ModelFitting_DataFiles", "ModelFitting_FigFiles"),
        "comp_validation",
    )
    os.makedirs(output_figDir_fits, exist_ok=True)

    # Create a base plotting settings instance (shared across plots)
    pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize=8)

    # Initialize 2D prediction settings based on the base, with method-specific overrides
    pred2D_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
    pred2D_settings = replace(
        pred2D_settings,
        visualize_samples=False,
        visualize_gt=False,
        visualize_model_estimatedCov=False,
        modelpred_alpha=1,
        modelpred_lw=0.7,
        modelpred_ls="-",
        ticks=np.linspace(-0.7, 0.7, 5),
        legend_off=False,
        flag_rescale_axes_label=False,
    )

    # Initialize visualization object for the Wishart model predictions
    wishart_pred_vis_MOCS = WishartPredictionsVisualization(
        expt_trial,
        model_pred_Wishart_MOCS.model,
        model_pred_Wishart_MOCS,
        color_thres_data,
        settings=pltSettings_base,
        save_fig=False,
    )

    # ---------------------------------------------------------------------------------
    # SECTION 5a: Visualize threshold ellipses at validation conditions with 95% CIs
    # ---------------------------------------------------------------------------------
    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=pred2D_settings.fig_size, dpi=pred2D_settings.dpi)

    # Plot bootstrapped confidence intervals (CI) from Wishart predictions
    cmap_allref = []
    for j in range(MOCS["nRefs"]):
        lbl = f"{percent_CI * 100}% bootstrap CI ({nBtst} AEPsych datasets)" if j == 0 else None

        # Convert 2D chromatic coordinate to RGB for colormap
        cm = color_thres_data.W2D_to_rgb(MOCS["xref_unique"][j])
        cmap_allref.append(cm)

        # Plot the CI region between inner and outer ellipse contours
        add_CI_ellipses(fitEll_min[j], fitEll_max[j], ax=ax, cm=cm, label=lbl, alpha=0.75)

    # Plot model-predicted threshold ellipses from AEPsych data
    wishart_pred_vis_MOCS.plot_2D(MOCS["xref_unique"][None], ax=ax, settings=pred2D_settings)

    # Plot MOCS trial conditions as direction vectors from reference stimuli
    plotCond_Settings = replace(PlotCondSettings(), **pltSettings_base.__dict__)
    plotCond_Settings = replace(
        plotCond_Settings,
        ref_ms=5,
        ref_lw=1,
        ref_label=None,
        ticks=np.linspace(-0.7, 0.7, 5),
        comp_ms=0.01,
        comp_lw=1.5,
        flag_show_comp_marker=False,
        easyTrials_highlight=False,
    )

    MOCS_cond_vis = MOCSConditionsVisualization(settings=pltSettings_base, save_fig=False)
    MOCS_cond_vis.plot_MOCS_conditions(
        ndims=2,
        xref_unique=MOCS["xref_unique"],
        comp_unique=MOCS["stim_at_targetPC_MOCS"][:, None],
        color_thres_data=color_thres_data,
        ax=ax,
        settings=plotCond_Settings,
    )

    # Plot confidence intervals for threshold magnitude along MOCS directions
    MOCS_chromDir = MOCS["stim_at_targetPC_MOCS"] - MOCS["xref_unique"]
    MOCS_chromDir_norm = MOCS_chromDir / np.linalg.norm(MOCS_chromDir, axis=-1)[:, None]
    for i in range(MOCS["nRefs"]):
        CI_i = MOCS["fit_PMF_MOCS"][i].stim_at_targetPC_95btstCI
        lb_i = MOCS["xref_unique"][i] + CI_i[0] * MOCS_chromDir_norm[i]
        ub_i = MOCS["xref_unique"][i] + CI_i[1] * MOCS_chromDir_norm[i]
        lbl = "95% bootstrap CI (120 MOCS datasets)" if i == 0 else None
        # Draw a line from lower to upper bound of threshold CI
        ax.plot(
            [lb_i[0], ub_i[0]],
            [lb_i[1], ub_i[1]],
            c="k",
            lw=1.5,
            solid_capstyle="butt",
            label=lbl,
        )

    ax.set_title("Isoluminant plane")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.45),
        fontsize=pred2D_settings.fontsize - 1,
    )
    # Save figure
    fig_name = f"{file_name[:-4]}_comparison_btw_MOCS_WishartPredictions_wBtstCI.pdf"
    fig.savefig(os.path.join(output_figDir_fits, fig_name), bbox_inches="tight")
    plt.show()

    # %%
    # ---------------------------------------------------------------------------------
    # SECTION 5b: Visualize PMFs along with WPPM predicted percent correct with 95% CI
    # ---------------------------------------------------------------------------------
    # Create a base plotting settings instance (shared across plots)
    pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize=11)
    # Initialize 2D prediction settings based on the base, with method-specific overrides
    predPMF_settings = replace(PlotPMFSettings(), **pltSettings_base.__dict__)

    # visualization object
    vis_MOCS = MOCSTrialsVisualization(MOCS["fit_PMF_MOCS"], settings=pltSettings_base, save_fig=False)

    # initialize color map
    cmap_allref = []
    for n in range(MOCS["nRefs"]):
        fig_name_part1 = file_name[:-4]
        fig_name_n = (
            f"{fig_name_part1}Ref{n}_Wdim1_{np.round(MOCS['xref_unique'][n][0], 2)}_"
            + f"Wdim2_{np.round(MOCS['xref_unique'][n][1], 2)}.pdf"
        )

        # define color map for each reference
        cmap_n = color_thres_data.W2D_to_rgb(MOCS["xref_unique"][n])
        # append colormap so we can reuse it for the next plot
        cmap_allref.append(cmap_n)

        predPMF_settings = replace(
            predPMF_settings,
            fig_size=(3.3, 6.3),
            filler_pts=[0, 1 / 3],
            cmap_PMF="k",
            cmap_dots="k",
            CI_area_alpha=0.3,
            CI_thres_errorbar_lw=5,
            Wishart_pred_lc=cmap_n,
            Wishart_pred_lw=1,
        )

        fig_n, ax_n = plt.subplots(1, 1, figsize=predPMF_settings.fig_size, dpi=predPMF_settings.dpi)
        vis_MOCS.plot_PMF(
            slc_idx=n,
            pX1_Wishart_slc=MOCS["pChoosingX1_Wishart"][n],
            xref=MOCS["xref_unique"][n],
            ax=ax_n,
            settings=predPMF_settings,
        )
        # add CI of the PMF based on the Wishart fit
        slc_PMF_MOCS = MOCS["fit_PMF_MOCS"][n]
        ax_n.fill_between(
            slc_PMF_MOCS.fineVal,
            *pChoosingX1_Wishart_CI_bds[:, n],
            lw=2,
            color=cmap_n,
            alpha=0.5,
            edgecolor="none",
        )
        # Add error bars for estimated threshold
        ax_n.errorbar(
            MOCS["vecLen_at_targetPC_Wishart"][n],
            slc_PMF_MOCS.target_pC,
            xerr=vecLen_at_targetPC_Wishart_err[n][:, None],
            c=cmap_n,
            lw=3,
            capsize=4,
        )
        fig_n.savefig(
            os.path.join(output_figDir_fits, f"{fig_name_n[:-4]}_v2.pdf"),
            bbox_inches="tight",
        )
        plt.show()

    # %%
    # ---------------------------------------------------------------------------------
    # SECTION 5c: Visualize the comparison of thresholds in a scatter plot
    # ---------------------------------------------------------------------------------
    # thresholds CI predicted by validation trials
    Validation_thres_CI_bds = np.vstack(
        [
            MOCS["vecLen_at_targetPC_MOCS"][i] + np.array([-1, 1]) * MOCS["fit_PMF_MOCS"][i].stim_at_targetPC_95btstErr
            for i in range(MOCS["nRefs"])
        ]
    )
    # compute how many validation conditions have overlapped confidence intervals
    _, num_overlaps, _ = intervals_overlap(Validation_thres_CI_bds, vecLen_at_targetPC_Wishart_CI_bds.T)

    # visualization object
    pltSettings_base2 = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize=9.5)
    vis_MOCS = MOCSTrialsVisualization(MOCS["fit_PMF_MOCS"], settings=pltSettings_base2, save_fig=True)
    subN = 7
    plt_st_n = plt_st[f"sub{subN}"]
    predComp_settings = replace(PlotThresCompSettings(), **pltSettings_base.__dict__)
    predComp_settings = replace(
        predComp_settings,
        fontsize=9.5,
        ms=6,
        fig_size=(4.8, 5),
        alpha=0.8,
        lw=1.5,
        bds=plt_st_n["bds"],
        corr_text_loc=plt_st_n["corr_text_loc"],
        slope_text_loc=plt_st_n["slope_text_loc"],
        numOverlaps_text_loc=plt_st_n["numOverlaps_text_loc"],
        xlabel="Threshold distance (validation)",
        ylabel="Threshold distance (WPPM)",
        cmap=cmap_allref,
        fig_name=f"{fig_name[:-4]}_v2.pdf",
    )
    # plot the comparison of thresholds between AEPsych predictions and MOCS predictions
    vis_MOCS.plot_comparison_thres(
        thres_Wishart=MOCS["vecLen_at_targetPC_Wishart"],
        slope_org=MOCS["slope_modelPred_org"].item(),
        slope_CI=slope_btst_CI,
        xref_unique=MOCS["xref_unique"],
        thres_Wishart_95btstErr=vecLen_at_targetPC_Wishart_err,
        corr_coef_org=MOCS["corr_coef_modelPred_org"],
        corr_coef_CI=corr_coef_btst_CI,
        num_overlaps=num_overlaps,
        settings=predComp_settings,
    )
