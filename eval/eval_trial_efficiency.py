#!/usr/bin/env python3
"""
Created on Sat Jul 26 14:50:58 2025

@author: fangfang
"""

import jax

jax.config.update("jax_enable_x64", True)
import copy
import os
import re
import sys
from dataclasses import replace

import dill as pickled
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from analysis.color_thres import CustomUnpickler
from analysis.conf_interval import find_inner_outer_contours_for_gridRefs
from analysis.cross_validation import TrialDistribution
from analysis.ellipses_tools import ellParamsQ_to_covMat
from analysis.model_performance import ModelPerformance
from analysis.utils_load import extract_between_patterns, load_expt_data
from core import oddity_task, optim
from core.model_predictions import wishart_model_pred
from core.wishart_process import WishartProcessModel
from plotting.adaptive_sampling_plotting import Plot2DSamplingSettings, SamplingRefCompPairVisualization
from plotting.modelperf_plotting import EvaluateTrialEfficiencyVisualization, PltVaryingTrialsNumbersSettings
from plotting.wishart_plotting import PlotSettingsBase
from plotting.wishart_predictions_plotting import Plot2DPredSettings, WishartPredictionsVisualization, add_CI_ellipses

# %%
# ---------------------------------------------------------------------------
# SECTION 1: load the model fits to the simulated dataset
# --------------------------------------------------------------------------
# Define base directory (adjusted for local access; comment out if running on HPC)
###### FLAG THAT WE CAN CHANGE #####
flag_running_on_hpc = False
ndims = 2
####################################

base_dir = (
    os.path.dirname(__file__) if flag_running_on_hpc else "/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/"
)

# simulated subject based on CIE1994
input_fileDir_fits = os.path.join(
    base_dir,
    "META_analysis",
    "ModelFitting_DataFiles",
    f"{ndims * 2}dTask",
    "CIE",
    "sub1",
    "decayRate0.5" if ndims == 2 else "decayRate0.4",
)

if ndims == 2:
    sim_file_name = (
        "Fitted_byWishart_Isoluminant plane_4DExpt_300_300_300_5100_"
        + "AEPsychSampling_EAVC_decayRate0.5_nBasisDeg5_sub1.pkl"
    )
elif ndims == 3:
    sim_file_name = (
        "Fitted_byWishart_ellipsoids_6DExpt_1500_1500_1500_25500_"
        + "AEPsychSampling_EAVC_decayRate0.4_nBasisDeg5_sub1.pkl"
    )

# Construct the full path to the selected file
sim_full_path = os.path.join(input_fileDir_fits, sim_file_name)

# Try loading the pickle file normally; if a module compatibility error occurs,
# reopen the file and fall back to a custom unpickler to bypass incompatible modules.
with open(sim_full_path, "rb") as f:
    vars_dict = pickled.load(f)

# Remove problematic JAX sharding-dependent entries
vars_dict.pop("model_pred_Wishart", None)
vars_dict.pop("model", None)  # If this key exists and is also problematic

# Save a lightweight version
sim_path_safe = sim_full_path.replace(".pkl", "_safe.pkl")
with open(sim_path_safe, "wb") as f:
    pickled.dump(vars_dict, f)

# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict["color_thres_data"]

# - Experimental trial data sampled by AEPsych
data = vars_dict["data_AEPsych"]

# number of trials for each strategy [300, 300, 300, 5100]
NTRIALS_STRAT = list(vars_dict["strat_dict"].values())
NTRIALS_SOBOL = sum(NTRIALS_STRAT[:3])
NTRIALS_ADAPTIVE = NTRIALS_STRAT[-1]

# %%
# -------------------------------------------------------------------------
# Section 2: load ground truth data
# -------------------------------------------------------------------------
# Select the ground truth file
if ndims == 2:
    gt_fileDir_fits = os.path.join(
        base_dir,
        "ELPS_analysis",
        "ModelFitting_DataFiles",
        f"{ndims}D_oddity_task",
        "Isoluminant_plane" if flag_running_on_hpc else "Isoluminant plane",
    )
    # in hpc: Isoluminant_plane
    # local: Isoluminant plane
    gt_file_name = (
        "Fitted_isothreshold_Isoluminant plane_CIE1994_sim18000total_"
        + "samplingNearContour_jitter0.3_seed0_bandwidth0.005_decay0.4_oddity.pkl"
    )
elif ndims == 3:
    gt_fileDir_fits = os.path.join(
        base_dir, "ELPS_analysis", "ModelFitting_DataFiles", f"{ndims}D_oddity_task", "CIE1994"
    )
    gt_file_name = (
        "Fitted_isothreshold_ellipsoids_sim240perCond_samplingNearContour_"
        + "jitter0.3_seed0_CIE1994_bandwidth0.005_oddity.pkl"
    )

# Build the full path to the selected file
gt_full_path = os.path.join(gt_fileDir_fits, gt_file_name)

# Try loading the pickle file normally; if a module compatibility error occurs
# (e.g., due to missing JAX internals like 'shard_arg'), fall back to a custom unpickler.
try:
    with open(gt_full_path, "rb") as f:
        vars_dict_gt = pickled.load(f)
except Exception:
    print("Retrying with CustomUnpickler...")
    vars_dict_gt = vars_dict = CustomUnpickler.safe_load_pickle(gt_full_path)

gt_model_pred = vars_dict_gt["model_pred_Wishart"]
gt_ell_vis = gt_model_pred.fitEll_unscaled
num_grid_pts_desired = gt_ell_vis.shape[0]

# %%
# -----------------------------------------------------------
# Optional: Truncate adaptive trials to achieve a desired edge-to-central ratio
# -----------------------------------------------------------
# Determine the bounds of the reference stimuli (xref)
xref_bounds = [np.min(data[1]).item(), np.max(data[1]).item()]

# Split the full dataset into two parts:
#   - Sobol-sampled trials (used to initialize AEPsych, typically not model-informed)
#   - Adaptively sampled trials (based on AEPsych model predictions)
data_sobol = tuple(arr[:NTRIALS_SOBOL] for arr in data)
data_adaptive = tuple(arr[NTRIALS_SOBOL:] for arr in data)

# Estimate the current proportion of edge trials in the adaptive portion
# (Edge vs. central is determined using xref_bounds)
_, _, nTrials_edge, nTrials_central = TrialDistribution.separate_edge_vs_central_trials_2Dplane(
    data_adaptive, xref_bounds
)
org_edge_percentage = nTrials_edge / (nTrials_edge + nTrials_central)
print(f"The original edge-to-total proportion is {org_edge_percentage:.3f}.")

# Specify the desired edge-to-total ratio for the *adaptive* trial subset.
# Set to None to skip this step and use all adaptive trials.
desired_edge_percentage = 0.1  # e.g., try values like 0.9, 0.7, 0.5, 0.3, 0.1, or None
if desired_edge_percentage is None:
    print("No trial truncation will be applied (using full dataset).")
else:
    print(
        f"The edge-to-total proportion we would like to include in the subset of data is {desired_edge_percentage:.3f}."
    )  # noqa: E501

# Minimum number of trials to retain (including the 900 Sobol trials)
nTrials_min = 1000  # must be ≥ NTRIALS_SOBOL
nSets_nTrials_inclusion = 11  # Number of progressively truncated subsets to evaluate

if desired_edge_percentage is not None:
    # Truncate the adaptive trials to match the desired edge-to-total ratio.
    # Returns trial indices for each subset level.
    dict_split_trials = TrialDistribution.truncate_edge_vs_central_trials_given_percentage(
        data_adaptive,
        xref_bounds,
        desired_edge_percentage,
        nTrials_min=nTrials_min - NTRIALS_SOBOL,  # minimum adaptive trials
        nLevels=nSets_nTrials_inclusion,
    )

    # Total number of trials at each level (including Sobol trials)
    nTrials_total = NTRIALS_SOBOL + dict_split_trials["nTrials_total_keep"]

    # Trial counts to include at each evaluation level
    nTrials_inclusion = NTRIALS_SOBOL + dict_split_trials["nTrials_levels"]

else:
    # Use full trial range and generate evenly spaced trial counts
    nTrials_total = sum(NTRIALS_STRAT)
    nTrials_inclusion = np.linspace(nTrials_min, nTrials_total, nSets_nTrials_inclusion).astype(int)

# %%
# -----------------------------------------------------------------------
# SECTION 3: Define constant variables for model fitting and evaluation
# -----------------------------------------------------------------------
# Number of Monte Carlo samples used to approximate the likelihood function
MC_SAMPLES = 2000

# Bandwidth used in the logistic density function (affects the smoothness of the likelihood surface)
BANDWIDTH = 5e-3

# Optimization parameters for fitting the model
opt_params = {
    "learning_rate": 1e-4,  # Step size for gradient-based optimization
    "momentum": 0.2,  # Momentum factor for the optimizer
    "mc_samples": MC_SAMPLES,
    "bandwidth": BANDWIDTH,
}

# extract the decay rate from the file
decay_rate = 0.5

# Define the Wishart model
model = WishartProcessModel(
    5,  # Degree of the polynomial basis functions
    ndims,  # Number of stimulus dimensions
    1,  # Number of extra inner dimensions in `U`.
    3e-4,  # Scale parameter for prior on `W`.
    decay_rate,  # Geometric decay rate on `W`
    0,  # Diagonal term setting minimum variance for the ellipsoids.
)

# we want to fit the Wishart model to the original dataset as well as bootstrapped
# dataset with specified seed
btst_seed = [None]  # + list(range(10))
flag_btst = [False]  # + [True]*10

# Generate a multidimensional grid spanning the model's 2D subspace (COLOR_DIMENSION / 2)
# The range [-0.7, 0.7] covers the normalized model space
grid = jnp.stack(jnp.meshgrid(*[jnp.linspace(-0.7, 0.7, num_grid_pts_desired) for _ in range(ndims)]), axis=-1)

# Target percent correct value corresponding to the discrimination threshold
target_pC = 0.667  # 3AFC midpoint between chance (0.33) and perfect (1.0)

# %% output file
# List of variable names to be saved
variable_names = [
    "sim_file_name",
    "data",
    "gt_file_name",
    "gt_model_pred",
    "desired_edge_percentage",
    "xref_bounds",
    "data_sobol",
    "data_adaptive",
    "dict_split_trials",
    "nTrials_total",
    "nTrials_inclusion",
    "opt_params",
    "model",
    "btst_seed",
    "flag_btst",
    "grid",
    "target_pC",
]

# Dictionary to store variable names and their corresponding values
vars_dict_init = {}
for var_name in variable_names:
    try:
        # Check if the variable exists in the global scope
        vars_dict_init[var_name] = eval(var_name)
    except NameError:
        # If the variable does not exist, assign None and print a message
        vars_dict_init[var_name] = None
        print(f"Variable '{var_name}' does not exist. Assigned as None.")

# %% loop through each trial number
if flag_running_on_hpc:
    for idx, nTrials in enumerate(nTrials_inclusion):
        # Set output directory for saving model fitting results or plots
        output_fileDir_fits_idx = os.path.join(
            base_dir,
            "hpc_sweeps",
            "trial_efficiency",
            f"{ndims}D_gtCIE",
            f"edge_percentage_{desired_edge_percentage}",
            f"{nTrials}trials",
        )

        # Uncomment the following line to create the output directory if it doesn't exist
        os.makedirs(output_fileDir_fits_idx, exist_ok=True)

        # select a subset of the original data
        # Unpack concatenated data
        if desired_edge_percentage is None:
            y_jnp, xref_jnp, x1_jnp = data
            data_subset, data_vis = TrialDistribution.select_proportion_data(
                [data_sobol, data_adaptive], [NTRIALS_SOBOL, nTrials - NTRIALS_SOBOL]
            )
            # If no truncation is needed, use the full set of adaptive trials
            split_trial_idx = [NTRIALS_SOBOL]  # marks start of adaptive trials
        else:
            # Build an array representing the number of trials to include from:
            # [Sobol trials, edge trials, central trials] for this level
            split_trial = np.array(
                [
                    NTRIALS_SOBOL,  # Fixed number of Sobol trials
                    dict_split_trials["nTrials_edge_levels"][idx],  # Edge trials to keep
                    dict_split_trials["nTrials_central_levels"][idx],  # Central trials to keep
                ]
            )

            # Pass the corresponding datasets and desired trial counts to the function.
            # The function returns:
            #   - data_subset: the full concatenated dataset
            #   - data_vis: a list of individually truncated datasets for [Sobol, Edge, Central]
            data_subset, data_vis = TrialDistribution.select_proportion_data(
                [data_sobol, dict_split_trials["data_edge_trunc"], dict_split_trials["data_central_trunc"]], split_trial
            )

            # Compute cumulative indices to split the concatenated dataset into segments
            split_trial_idx = np.cumsum(split_trial)

        for flag_btst_AEPsych, ll in zip(flag_btst, btst_seed):  # noqa: B905
            if flag_btst_AEPsych:
                str_ext = f"_btst_AEPsych[{ll}]"  # noqa: E701
            else:
                str_ext = ""  # noqa: E701

            # Construct output filename based on number of folds and subject/session identifier
            output_file_ll = f"TrialEfficiency_{nTrials}trials_{sim_file_name[:-4]}{str_ext}.pkl"
            full_path_ll = os.path.join(output_fileDir_fits_idx, output_file_ll)

            # Write the list of dictionaries to a file using pickle/dill
            with open(full_path_ll, "wb") as f:
                pickled.dump(vars_dict_init, f)

            # when i == 0, fit the original subset of the data; do not bootstrap
            if flag_btst_AEPsych:
                # bootstrap AEPsych trials
                xref_btst, x1_btst, y_btst, btst_indices = load_expt_data.bootstrap_AEPsych_data(
                    data_subset[1], data_subset[2], data_subset[0], trials_split=split_trial_idx, seed=ll
                )
            else:
                y_btst, xref_btst, x1_btst = data_subset
                btst_indices = None
            data_btst = (y_btst, xref_btst, x1_btst)

            # -----------------------------------------------------------------------
            # SECTION 4: Fit the Wishart model
            # -----------------------------------------------------------------------
            # Generate a matrix of random seeds for each initialization
            random_seeds = np.random.randint(0, 2**32, size=(2,))

            # Generate random keys for initializing parameters, data, and optimizer
            W_INIT_KEY = jax.random.PRNGKey(random_seeds[0])  # Key to initialize `W_est`.
            OPT_KEY = jax.random.PRNGKey(random_seeds[1])  # Key passed to optimizer.

            # Fit model, initialized at a random W sampled from the prior distribution
            W_init = 1e-1 * model.sample_W_prior(W_INIT_KEY)

            W_est, iters, objhist = optim.optimize_posterior(
                W_init,
                data_btst,
                model,
                OPT_KEY,
                copy.deepcopy(opt_params),
                oddity_task.simulate_oddity,
                total_steps=10,
                save_every=1,
                show_progress=True,
            )

            # fig, ax = plt.subplots(1, 1)
            # ax.plot(iters, objhist)
            # fig.tight_layout(); plt.show()

            # -------------------------------------------------------
            # SECTION 5: Compute model predictions (66.7% correct )
            # -------------------------------------------------------
            # Compute the covariance matrices ('Sigmas') at each point in the grid using
            # the model's compute_U function.
            Sigmas_noise_grid = model.compute_Sigmas(model.compute_U(W_est, grid))

            # Initialize the Wishart model prediction using various parameters.
            model_pred_Wishart = wishart_model_pred(
                model,
                opt_params,
                W_INIT_KEY,
                OPT_KEY,
                W_init,
                W_est,
                Sigmas_noise_grid,
                color_thres_data,
                target_pC=target_pC,
                ngrid_bruteforce=1000,
                bds_bruteforce=[0.0005, 0.25],
            )

            # batch compute 66.7% threshold contour based on estimated weight matrix
            model_pred_Wishart.convert_Sig_Threshold_oddity_batch(grid)

            # -------------------------------------------------------------------------
            # Compute Bures-Wasserstein Distance (BWD) and major axis discrepancies
            # between WPPM-predicted and ground-truth ellipses
            # -------------------------------------------------------------------------
            # Initialize arrays to store outputs
            base_shape = (num_grid_pts_desired, num_grid_pts_desired)
            major_gt = np.full(base_shape, np.nan)  # Ground-truth major axis lengths
            minor_gt = np.full(base_shape, np.nan)  # Ground-truth minor axis lengths
            BWD = np.full(base_shape, np.nan)  # Bures-Wasserstein distances

            # Loop over all reference color grid locations
            for n in range(num_grid_pts_desired):
                for m in range(num_grid_pts_desired):
                    # Extract WPPM-predicted ellipse parameters
                    _, _, a, b, theta = model_pred_Wishart.params_ell[n][m]

                    # Extract ground-truth ellipse parameters
                    _, _, major_gt, minor_gt, theta_gt = gt_model_pred.params_ell[n][m]

                    # Convert ellipse parameters to covariance matrices
                    cov_nm = ellParamsQ_to_covMat(a, b, theta)
                    cov_nm_gt = ellParamsQ_to_covMat(major_gt, minor_gt, theta_gt)

                    # Compute Bures-Wasserstein distance between predicted and ground-truth ellipses
                    BWD[n, m] = ModelPerformance.compute_Bures_Wasserstein_distance(cov_nm, cov_nm_gt)
            BWD_sum = np.sum(BWD)
            BWD_avg = np.mean(BWD)
            BWD_min = np.min(BWD)
            BWD_max = np.max(BWD)

            # print the results
            print(
                f"Percent of edge trials: {desired_edge_percentage}; "
                + f"Included trials: {nTrials}; bootstrap: {ll}; BWD: {BWD_avg:.4f}; "
                + f"sum: {BWD_sum:.4f}; range: [{BWD_min:.4f}, {BWD_max:.4f}]"
            )

            # % append data
            data_subset_copy = data_subset
            append_variable_names = [
                "data_vis",
                "split_trial_idx",
                "data_subset_copy",
                "data_btst",
                "btst_indices",
                "random_seeds",
                "W_INIT_KEY",
                "OPT_KEY",
                "W_init",
                "W_est",
                "iters",
                "objhist",
                "Sigmas_noise_grid",
                "model_pred_Wishart",
                "BWD",
                "BWD_sum",
                "BWD_avg",
                "BWD_min",
                "BWD_max",
            ]

            append_vars_dict = {}
            for var_name in append_variable_names:
                append_vars_dict[var_name] = eval(var_name)

            # Load the existing pickle file
            with open(full_path_ll, "rb") as file:
                vars_dict_load = pickled.load(file)

            # Update the loaded dictionary with new variables
            vars_dict_load.update(append_vars_dict)

            # Save the updated dictionary back to the same file
            with open(full_path_ll, "wb") as file:
                pickled.dump(vars_dict_load, file)

            # -----------------------------------------------------------------------
            # Clean up: delete all temporary variables
            # -----------------------------------------------------------------------
            # Delete the append_vars_dict itself
            del append_vars_dict, output_file_ll, full_path_ll

            # Delete each variable listed in append_variable_names, if it exists
            for var_name in append_variable_names:
                try:
                    del globals()[var_name]
                except KeyError:
                    pass  # Variable wasn't defined or already deleted

        # clean up the file name and path
        del output_fileDir_fits_idx, data_subset

# %%
# -----------------------------------------------------------------------
# We can only run the following two sections if we get the sbatch jobs back from the hpc
# -----------------------------------------------------------------------
if not flag_running_on_hpc:
    # Continuously load trial efficiency result files selected by the user
    # This allows loading multiple datasets in sequence for comparison
    # We avoid relying on fixed full paths

    # navigate to: 'META_analysis/ModelFitting_DataFiles/4dTask/CIE/sub1/trial_efficiency/edge_percentage_#'
    # select all the files in that folder
    BWD_results_list, params_ell_list, varying_nTrials, loaded_percentages, modelpred_list, vars_dict_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    while True:
        try:
            # Prompt user to select one or more trial efficiency result files
            # and load corresponding BWD summary and model predictions
            BWD_results, input_fileDir_fits_set, vars_dict_set = TrialDistribution.load_trial_efficiency_results()

            # Store the loaded results
            BWD_results_list.append(BWD_results)
            vars_dict_list.append(vars_dict_set)

            # Extract number of trials from each file name
            varying_nTrials_temp = extract_between_patterns(
                input_fileDir_fits_set, str_prefix="TrialEfficiency_", str_suffix="trials"
            )
            varying_nTrials.append(varying_nTrials_temp)

            # Extract the edge percentage value from the directory name of the last file
            loaded_percentages_temp = extract_between_patterns(
                [input_fileDir_fits_set[-1]], str_prefix="percentage_", str_suffix="/TrialEfficiency"
            )
            loaded_percentages.append(loaded_percentages_temp[0])

        except:  # noqa: E722
            print("Cancel loading more data.")
            break

    # -----------------------------------------------------------------------
    # Visualize nLL as a function of trial number
    # -----------------------------------------------------------------------
    input_fileDir_fits_last, file_name_last = os.path.split(input_fileDir_fits_set[-1])

    # create a fig directory for the summary figure
    output_sumFig_dir = os.path.dirname(
        input_fileDir_fits_last.replace("ModelFitting_DataFiles", "ModelFitting_FigFiles")
    )
    os.makedirs(output_sumFig_dir, exist_ok=True)

    # figure name
    fig_name = re.sub(r"\d+trials_", "", file_name_last[:-4])

    # Configure plotting settings
    # Base plotting settings (e.g., figure directory and font size)
    pltSettings_base = PlotSettingsBase(fig_dir=output_sumFig_dir, fontsize=9)

    # Initialize figure-specific settings by merging with base settings
    BWD_vis_settings = replace(PltVaryingTrialsNumbersSettings(), **pltSettings_base.__dict__)
    BWD_vis_settings = replace(BWD_vis_settings, fig_name=fig_name)

    # Instantiate plotter
    BWD_vis = EvaluateTrialEfficiencyVisualization(pltSettings_base, save_fig=False)

    # Generate plot
    fig, ax = plt.subplots(figsize=(5, 3.25), dpi=1024)

    # Get the 'turbo' colormap and discretize to 6 colors
    cmap = plt.get_cmap("turbo")
    colors_rgb = cmap(np.arange(cmap.N))[::40, :3]

    for i, bwd in enumerate(BWD_results_list):
        label = (
            f"{loaded_percentages[i]}" if loaded_percentages[i] != "None" else f"Original: {org_edge_percentage:.2f}"
        )

        BWD_vis_settings = replace(BWD_vis_settings, lc=colors_rgb[i], label_line=label, ylim=[1, 5])

        BWD_vis.plot_BWD_varying_nTrials(
            bwd["nTrials_inclusion"], bwd["BWD_sum_all_org"], BWD_vis_settings, BWD_CI=bwd["BWD_CI"], ax=ax
        )
    ax.legend(title="Edge-to-total proportion", ncol=3)
    # Save the figure as a PDF
    fig.savefig(os.path.join(output_sumFig_dir, f"{fig_name}.pdf"), bbox_inches="tight")

# %% plot the selected trials
if not flag_running_on_hpc:
    # loop through each edge-to-total condition
    for i, vars_dict_i in enumerate(vars_dict_list):
        # create subfolder
        output_sumFig_dir_i = os.path.join(output_sumFig_dir, f"edge_percentage_{loaded_percentages[i]}")
        os.makedirs(output_sumFig_dir_i, exist_ok=True)

        # plot settings
        pltSettings_base = PlotSettingsBase(fig_dir=output_sumFig_dir_i, fontsize=8)
        pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
        pltSettings_tp = replace(
            pltSettings_tp,
            ref_markeralpha=0.5,
            comp_markeralpha=0.2,
            linealpha=0.2,
            ticks=np.linspace(-0.7, 0.7, 5),
            bounds=xref_bounds,
        )
        sampling_vis = SamplingRefCompPairVisualization(
            ndims, color_thres_data, settings=pltSettings_tp, save_fig=False
        )

        # retrieve the saved dictionaries
        vars_dict_i_full = vars_dict_i[-1][0]

        # data_vis_i is a tuple with 3 elements, corresponding to Sobol, edge and central data
        data_vis_i = vars_dict_i_full["data_vis"]

        # Loop over the selected data points to generate and visualize each corresponding figure.
        for i, (data_i, ttl_i) in enumerate(zip(data_vis_i, ["Sobol", "Edge", "Central"])):  # noqa: B007, B905
            fig, ax = plt.subplots(1, 1, figsize=(3, 3.5), dpi=pltSettings_tp.dpi)
            nTrials_i = data_i[0].shape[0]
            pltSettings_tp = replace(pltSettings_tp, title=f"{ttl_i} trials: {nTrials_i}")
            # Visualize the trials up to the nth data point with specified marker transparency.
            sampling_vis.plot_sampling(
                data_i[1],  # Reference points up to the nth data point
                data_i[2],  # Comparison points up to the nth data point
                settings=pltSettings_tp,
                ax=ax,
            )

            # Save the figure as a PDF
            fig.savefig(
                os.path.join(output_sumFig_dir_i, f"TrialEfficiency_{ttl_i}{nTrials_i}.pdf"), bbox_inches="tight"
            )
            plt.show()

# %% plot the confidence interval
if not flag_running_on_hpc:
    # loop through each edge-to-total condition
    for i, vars_dict_i in enumerate(vars_dict_list):
        # create subfolder
        output_sumFig_dir_i = os.path.join(output_sumFig_dir, f"edge_percentage_{loaded_percentages[i]}")
        # Create a base plotting settings instance (shared across plots)
        pltSettings_base = PlotSettingsBase(fig_dir=output_sumFig_dir_i, fontsize=8)

        # Initialize 2D prediction settings by copying from base and overriding method-specific parameters
        pred2D_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
        pred2D_settings = replace(
            pred2D_settings,
            visualize_samples=False,
            visualize_gt=True,
            visualize_model_estimatedCov=False,
            flag_rescale_axes_label=False,
            visualize_model_pred=True,
            ticks=np.linspace(-0.7, 0.7, 5),
            modelpred_alpha=1,
            gt_lw=0.5,
            gt_lc="k",
            gt_label="Ground truths",
            gt_ls="--",
        )

        for d in range(nSets_nTrials_inclusion):
            modelpred_id = vars_dict_i[d][0]["model_pred_Wishart"]

            # Create figure and axes for plotting
            fig_d, ax_d = plt.subplots(1, 1, figsize=pred2D_settings.fig_size, dpi=pred2D_settings.dpi)
            pred2D_settings = replace(
                pred2D_settings,
                title=f"Edge-to-total ratio: {loaded_percentages[i]}; trials: {varying_nTrials[i][d]}",  # noqa: E501
                fig_name=f"ModelPredictions_EdgeToTotalRatio{loaded_percentages[i]}_{varying_nTrials[i][d]}trials.pdf",
            )  # noqa: E501

            # Initialize Visualization Class for Wishart Predictions
            wishart_pred_vis_wCI = WishartPredictionsVisualization(
                None, modelpred_id, modelpred_id, color_thres_data, settings=pltSettings_base, save_fig=True
            )

            if np.sum(flag_btst) >= 2:
                # Infer grid size and extract ellipse parameters
                params_ell_grid = TrialDistribution.extract_params_ell_grid(modelpred_id)

                # Computes the confidence intervals for the model-predicted ellipses at each grid point.
                fitEll_min, fitEll_max = find_inner_outer_contours_for_gridRefs(params_ell_grid)
                # plot the confidence interval
                for i in range(num_grid_pts_desired):
                    for j in range(num_grid_pts_desired):
                        cm = color_thres_data.W2D_to_rgb(grid[i, j])
                        if i == 0 and j == 0:
                            lbl = f"100% CI computed from the original \ndataset and {np.sum(flag_btst)} bootstrap"
                        else:
                            lbl = None
                        add_CI_ellipses(
                            fitEll_min[i, j], fitEll_max[i, j], ax=ax_d, cm=cm, label=lbl, lw_outer=0, alpha=0.8
                        )

            # Overlay model predictions (joint fits) onto the same axes
            wishart_pred_vis_wCI.plot_2D(
                vars_dict_i[d][0]["grid"], gt_ellipses=gt_ell_vis, ax=ax_d, settings=pred2D_settings
            )
