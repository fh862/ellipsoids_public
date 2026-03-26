#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 22:48:14 2025

@author: fangfang

This script analyzes the effects of two hyperparameters in the Wishart Process
Psychophysical Model (WPPM):

    1. ε (epsilon): decay rate
    2. γ (gamma): variance scale

Together, these parameters determine the variance of the model weights:
    η_{i+j} = γ · ε^{(i+j)},
where (i + j) denotes the order of the 2D Chebyshev polynomial basis functions.

In the original analysis, the hyperparameters were hand-selected
(ε = 0.4, γ = 0.0003). Here, we perform a systematic evaluation of these choices.

----------------------------------------------------------------------
Hyperparameter sweep
----------------------------------------------------------------------

We sweep a range of values for each hyperparameter:
    • ε ∈ [0.15, 1]
    • γ ∈ [1e-6, 3e-3]

Model performance is evaluated using 5-fold cross-validation, with predictive
accuracy quantified by the negative log-likelihood (nLL) on held-out test data.

----------------------------------------------------------------------
Cross-validation procedure
----------------------------------------------------------------------

For each hyperparameter value:

1. The dataset is split into 5 folds.
2. The WPPM is fit to 4 folds (training set).
3. The fitted model is used to compute the nLL on the held-out fold (test set).
4. Steps 2–3 are repeated until each fold has served as the test set.
5. Training and test nLLs are averaged across the 5 folds.

To reduce sensitivity to local minima, each model fit is initialized from
three random starting points, and the solution with the lowest negative log
posterior is retained.

----------------------------------------------------------------------
Main findings
----------------------------------------------------------------------

The two hyperparameters influence predictive accuracy in a similar manner:

• With γ fixed at 0.0003, predictive accuracy is highest for ε ≈ 0.3–0.5.
  Performance drops rapidly for ε < 0.3 due to oversmoothing, and decreases
  more gradually for ε > 0.5, reflecting undersmoothing and increased uncertainty
  in model predictions.

• With ε fixed at 0.4, predictive accuracy is highest for γ ≈ 1e-5–1e-3.
  Performance declines rapidly for smaller γ (oversmoothing) and more slowly
  for larger γ as undersmoothing increases prediction uncertainty.

Based on this analysis, we selected ε = 0.4 (slightly smaller than the original
value of 0.5) while retaining γ = 0.0003.

----------------------------------------------------------------------
Figure outputs
----------------------------------------------------------------------

This script produces the following figures:

1. Training and test nLL as a function of the swept hyperparameter.
2. The full range of model predictions across a grid of reference colors,
   aggregated across the 5 cross-validation folds for each hyperparameter value.
3. Variance of model weights as a function of Chebyshev basis order at varying
   hyperparameter values.

----------------------------------------------------------------------
HPC execution notes
----------------------------------------------------------------------

When running on the HPC, the script is launched using the following SLURM setup:

    #!/bin/bash
    #SBATCH --job-name=cross_validate_find_opt_decay_rate
    #SBATCH --output=slurm_scripts/slurm%j.out
    #SBATCH --mail-type=END
    #SBATCH --mail-user=fh862@sas.upenn.edu
    #SBATCH -p gpu -N1 -G1 --constraint=h100
    #SBATCH --cpus-per-task=4 --mem-per-cpu=20G
    #SBATCH --time=06:00:00

Note that hyperparameter sweeps may need to be split across multiple jobs.
In particular, for ε > 0.8 (weaker smoothness constraints), convergence is slower,
and we increased the total number of optimization steps to 5000 (vs. 3000 for
other settings).

"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import dill as pickled
import numpy as np
import copy
import re
from dataclasses import replace
import os
from core import optim, oddity_task
from core.wishart_process import WishartProcessModel
from core.model_predictions import wishart_model_pred
from analysis.color_thres import color_thresholds
from analysis.utils_load import load_expt_data
from analysis.cross_validation import CrossValidation
from analysis.utils_load import select_file_and_get_path, extract_sub_number
from analysis.conf_interval import find_inner_outer_contours_for_gridRefs
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization,\
    Plot2DPredSettings, add_CI_ellipses
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.modelperf_plotting import NFoldsCrossValidationVisualization, \
    PltVaryingHyperParamSettings

#%%
# -----------------------------------------------------------
# SECTION 1: Set up directories and load model predictions
# -----------------------------------------------------------
# Define base directory (adjusted for local access; comment out if running on HPC)
flag_running_on_hpc = False
base_dir = os.path.dirname(__file__) if flag_running_on_hpc else \
    '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

# Specify participant and experiment details
stim_dims = 2
psyfield_dims = 4   # Dimensionality of the psychometric field (2 dims for the ref; 2 dims for the delta)
################## THINGS NEEDED TO CHANGE ##################
subN = 8           # Subject number
nSessions = 12     # Total number of experimental sessions
hyper_param = 'DecayRate' # 'DecayRate' or 'VarianceScale'
#############################################################

# Specify color plane for analysis and load relevant transformation matrices
color_thres_data = color_thresholds(stim_dims, base_dir, plane_2D = 'Isoluminant plane')
color_thres_data.load_transformation_matrix(file_date = "02242025")

# Set output directory for saving model fitting results or plots
output_fileDir_fits = os.path.join(base_dir, 'hpc_sweeps', f'cv_{hyper_param}', f'sub{subN}')
os.makedirs(output_fileDir_fits, exist_ok=True)

#%%
# -----------------------------------------------------------
# SECTION 2: Organize and prepare data for cross-validation
# -----------------------------------------------------------
# Set the number of folds for cross-validation
total_folds = 5

# Set a participant-specific seed for reproducibility
shuffle_seed = subN
    
# Retrieve file paths and metadata for all sessions of the given subject
path_str = os.path.join(base_dir, 'ELPS_analysis', 'Experiment_DataFiles',
                        'pilot2', f'sub{subN}')
session_files, session_file_name_part1 = \
    load_expt_data.get_all_sessions_file_names(subN, nSessions, path_str)

# Load data from all session files into a list
data_allSessions = load_expt_data.load_data_all_sessions(session_files)

# Extract and concatenate AEPsych trials and pregenerated Sobol trials
aepsych_data, sobol_data, combined_data = \
    load_expt_data.load_combine_AEPsych_pregSobol(data_allSessions)

# Unpack combined data into individual arrays
xref_combined, x1_combined, y_combined = combined_data
            
# Shuffle the data (preserving correspondence across xref, x1, and y)
data_shuffled = CrossValidation.shuffle_data(
    (y_combined, xref_combined, x1_combined),
    seed = shuffle_seed
)

# Split the shuffled dataset into N folds for cross-validation
# Returns a dictionary where each key corresponds to a fold index (0 to total_folds-1)
# For each fold, the value is a tuple:
#   (training_data, validation_data, training_indices, validation_indices)
# Each training/validation data is a tuple of (y, xref, x1)
data_split_NFold = CrossValidation.select_NFold_data_noFixedRef(
    data_shuffled, total_folds
)

#%% cross validation
# -----------------------------------------------------------------------
# SECTION 3: Define constant variables for model fitting and evaluation
# -----------------------------------------------------------------------
# Optimization parameters for fitting the model
opt_params = {
    "learning_rate": 1e-4,   # Step size for gradient-based optimization
    "momentum": 0.2,         # Momentum factor for the optimizer
    "mc_samples": 2000,      # Number of Monte Carlo samples used to approximate the likelihood function
    "bandwidth": 5e-3,       # Bandwidth used in the logistic density function (affects the smoothness of the likelihood surface)
}

# Repeat optimization from multiple random initializations to reduce the risk of
# converging to a poor local minimum
nRepeats = 3 

# Number of grid points along each axis for evaluating model predictions
NUM_GRID_PTS = 7

# Generate a multidimensional grid spanning the model's 2D subspace 
# The range [-0.7, 0.7] covers the normalized model space
grid = jnp.stack(
    jnp.meshgrid(*[jnp.linspace(-0.7, 0.7, NUM_GRID_PTS) for _ in range(stim_dims)]),
    axis=-1
)

# Target percent correct value corresponding to the discrimination threshold
target_pC = 2/3  # 2AFC midpoint between chance (0.5) and perfect (1.0)

# Decide the sweep values
if hyper_param == "DecayRate":
    hyper_param_arr = np.concatenate([
        np.array([0.15]),
        np.arange(0.2, 1.05, 0.05)
    ])
elif hyper_param == "VarianceScale":
    hyper_param_arr = np.concatenate([
        np.array([1e-6, 5e-6, 1e-5, 5e-5]),
        np.linspace(1e-4, 1e-3, 10),
        np.array([0.0015, 0.002, 0.003])
    ])
else:
    raise ValueError(f"Unknown hyper_param: {hyper_param}")

# Total number of decay rates to sweep over
n_hyper_param = len(hyper_param_arr)

# Map the sweep to the correct model argument name
sweep_arg = {
    "DecayRate": "decay_rate",
    "VarianceScale": "variance_scale",
}[hyper_param]

#kept decimals
decimals = {"DecayRate": 4, "VarianceScale": 6}[hyper_param]

#greek symbol
hp_symbol = {"DecayRate": r"\epsilon", "VarianceScale": r"\gamma"}[hyper_param]

# Fixed model kwargs
base_kwargs = dict(
    degree=5,
    num_dims = stim_dims,
    extra_dims = 1,
    variance_scale=3e-4,# default (will be overridden if sweeping VarianceScale)
    decay_rate=0.4,     # default (will be overridden if sweeping DecayRate)
    diag_term=0,
)

# output file
# List of variable names to be saved
variable_names = ['subN','stim_dims','psyfield_dims','nSessions', 'color_thres_data',
                  'total_folds', 'shuffle_seed', 'data_allSessions', 'aepsych_data', 
                  'sobol_data', 'combined_data', 'data_shuffled', 'data_split_NFold', 
                  'NUM_GRID_PTS', 'opt_params', 'nRepeats', 'grid', 'target_pC',
                  'hyper_param','hyper_param_arr', 'n_hyper_param']

# Dictionary to store variable names and their corresponding values
vars_dict = {}
for var_name in variable_names:
    try:
        # Check if the variable exists in the global scope
        vars_dict[var_name] = eval(var_name)
    except NameError:
        # If the variable does not exist, assign None and print a message
        vars_dict[var_name] = None
        print(f"Variable '{var_name}' does not exist. Assigned as None.")

#%% Start hyperparameter sweeps
if flag_running_on_hpc:
    for d, val in enumerate(hyper_param_arr):
        # Construct output filename based on number of folds and subject/session identifier
        output_file_d = f"CrossValidation{total_folds}folds_{nRepeats}reptitions_"+\
                      f"varying{hyper_param}{val:.{decimals}f}_{session_file_name_part1}.pkl"
        full_path_d = os.path.join(output_fileDir_fits, output_file_d)
        
        # Write the list of dictionaries to a file using pickle/dill
        with open(full_path_d, 'wb') as file:
            pickled.dump(vars_dict, file)
        
        for f in range(total_folds):
            #select the data for fitting
            data_keep = data_split_NFold[f][0]
            data_heldout = data_split_NFold[f][1]
            
            nSuccess = 0
            attempt = 0
            while nSuccess < nRepeats:
                attempt += 1
                try:
                    #print out counters
                    print(f"{hyper_param}: {val:.{decimals}f}; fold: {f+1}; "+\
                          f"repetition(success): {nSuccess}; attempt: {attempt}")
                    
                    # -----------------------------------------------------------------------
                    # SECTION 4: Define the Wishart model with different decay rate
                    # -----------------------------------------------------------------------
                    # Override just the parameter being swept
                    model = WishartProcessModel(**(base_kwargs | {sweep_arg: float(val)}))
                
                    #Generate a matrix of random seeds for each initialization
                    random_seeds = np.random.randint(0, 2**32, size = (2,))
                    
                    # Generate random keys for initializing parameters, data, and optimizer
                    W_INIT_KEY   = jax.random.PRNGKey(random_seeds[0])  # Key to initialize `W_est`. 
                    OPT_KEY = jax.random.PRNGKey(random_seeds[1])  # Key passed to optimizer.
                    
                    # Fit model, initialized at a random W sampled from the prior distribution
                    W_init = model.sample_W_prior(W_INIT_KEY) 
                    
                    W_est, iters, objhist = optim.optimize_posterior(
                        W_init, data_keep, model, OPT_KEY,
                        copy.deepcopy(opt_params),
                        oddity_task.simulate_oddity, 
                        total_steps=1500, #NOTE that we might have to increase this value for some hyperparameter values
                        save_every=1,
                        show_progress=True
                    )
                        
                    #fig, ax = plt.subplots(1, 1)
                    #ax.plot(iters, objhist)
                    #fig.tight_layout(); plt.show()
                    
                    # -------------------------------------------------------
                    # SECTION 5: Compute model predictions (66.7% correct )
                    # -------------------------------------------------------
                    # Compute the covariance matrices ('Sigmas') at each point in the grid using 
                    # the model's compute_U function. 
                    Sigmas_noise_grid = model.compute_Sigmas(model.compute_U(W_est, grid))
                    
                    # Initialize the Wishart model prediction using various parameters.
                    model_pred_Wishart = wishart_model_pred(model, opt_params, W_INIT_KEY,
                                                            OPT_KEY, W_init,
                                                            W_est, Sigmas_noise_grid,
                                                            color_thres_data, 
                                                            target_pC = target_pC,
                                                            ngrid_bruteforce = 1000,
                                                            bds_bruteforce = [0.0005, 0.25])
                    
                    # batch compute 66.7% threshold contour based on estimated weight matrix
                    model_pred_Wishart.convert_Sig_Threshold_oddity_batch(grid)
                except Exception as e:
                    print(f"Failed attempt (will retry): {e}")
                    continue  # skip everything below and try again

                # If we reach here, the try block succeeded.
                nSuccess += 1

                # -----------------------------------------------------------------
                # SECTION 6: compute the likelihood of the model given heldout data
                # -----------------------------------------------------------------     
                DATA_KEY = jax.random.PRNGKey(np.random.randint(0, 2**32)) 
                #negative log likelihood of W given held out data
                nLL_heldout_data = -oddity_task.estimate_loglikelihood(
                        W_est, model, data_heldout, DATA_KEY, opt_params['mc_samples'], 
                        opt_params['bandwidth'], oddity_task.simulate_oddity
                    )
                
                #negative log likelihood of W given the training data
                nLL_training_data = -oddity_task.estimate_loglikelihood(
                        W_est, model, data_keep, DATA_KEY, opt_params['mc_samples'], 
                        opt_params['bandwidth'], oddity_task.simulate_oddity
                    )
                
                #negative log posterior of W given the training data
                nPL_training_data = objhist[-1]
                
                #print the results
                print(f"nPL (training data): {nPL_training_data:.4f}; "+\
                      f"nLL (training data): {nLL_training_data:.4f}; "+\
                      f"nLL (test data): {nLL_heldout_data:.4f}")
                
                # Define the new key under which the dictionary will be nested
                new_key_name = f'{hyper_param}{val:.{decimals}f}_CVfold{f}_rep{nSuccess}'
                
                #% append data        
                append_variable_names = ['model', 'random_seeds', 'W_INIT_KEY','W_init',
                                         'W_est', 'iters', 'objhist', 'Sigmas_noise_grid', 
                                         'model_pred_Wishart', 'DATA_KEY', 
                                         'nLL_heldout_data','nPL_training_data',
                                         'nLL_training_data']
        
                append_vars_dict = {}
                for var_name in append_variable_names:
                    append_vars_dict[var_name] = eval(var_name)
                    
                # Load the existing pickle file
                with open(full_path_d, 'rb') as file:
                    vars_dict_f = pickled.load(file)
                
                # Add the new nested dictionary under the specified key
                vars_dict_f[new_key_name] = append_vars_dict
                
                # Save the updated dictionary back to the same file
                with open(full_path_d, 'wb') as file:
                    pickled.dump(vars_dict_f, file)
                    
                # -----------------------------------------------------------------------
                # Clean up: delete all temporary variables
                # -----------------------------------------------------------------------      
                # Delete the append_vars_dict itself
                del append_vars_dict
                
                # Delete each variable listed in append_variable_names, if it exists
                for var_name in append_variable_names:
                    try:
                        del globals()[var_name]
                    except KeyError:
                        pass  # Variable wasn't defined or already deleted
                        
        # clean up the file name and path
        del output_file_d, full_path_d
                    
#%%
# --------------------------------------------------------------------------------------
# We can only run the following two sections if we get the sbatch jobs back from the hpc
# -------------------------------------------------------------------------------------- 
if not flag_running_on_hpc:
    # Prompt user to select a cross-validation results file.
    # Navigate to ELPS_analysis/Experiment_DataFiles/sub#/fits/sweep_variance_scale
    # 'CrossValidation5folds_3reptitions_varyingVarianceScale0.000001_ColorDiscrimination_4dExpt_Isoluminant plane_sub8.pkl'
    # or
    # ELPS_analysis/Experiment_DataFiles/sub#/fits/sweep_decay_rate
    # 'CrossValidation5folds_3reptitions_varyingDecayRate0.1500_ColorDiscrimination_4dExpt_Isoluminant plane_sub8.pkl'
    input_fileDir_fits_set, file_name_set = select_file_and_get_path()
    
    sub_loaded = extract_sub_number(file_name_set)
    if sub_loaded != subN:
        raise ValueError(f"Subject mismatch: expected sub{subN}, but loaded sub{sub_loaded} from '{file_name_set}'.")

    # Construct the full path to the selected file
    full_path_set = os.path.join(input_fileDir_fits_set, file_name_set)

    # Preallocate arrays to store metrics across decay rates and folds
    base_shape = (n_hyper_param, total_folds)
    nLP_training = np.full(base_shape, np.nan)   # Negative log posterior
    nLL_training = np.full(base_shape, np.nan)   # Negative log likelihood (training)
    nLL_test = np.full(base_shape, np.nan)       # Negative log likelihood (test)
    params_ell = np.full(base_shape + (NUM_GRID_PTS, NUM_GRID_PTS, 5), np.nan)

    # Iterate over all decay rates
    for d, val in enumerate(hyper_param_arr):
        # Replace the decay rate part of the file name dynamically
        full_path_set_d = full_path_set.replace(f'varying{hyper_param}{hyper_param_arr[0]:.{decimals}f}',
                                                f'varying{hyper_param}{val:.{decimals}f}')
        
        # Load the data dictionary from file
        with open(full_path_set_d, 'rb') as f:
            vars_dict_set_d = pickled.load(f)
        
        # Iterate through cross-validation folds
        for f in range(total_folds):
            print(f"{hyper_param}: {val:.{decimals}f}; fold: {f+1}")
            
            # Temporary storage for each repetition
            nLP_training_allreps = np.full((nRepeats,), np.nan)
            nLL_training_allreps = np.full((nRepeats,), np.nan)
            nLL_test_allreps = np.full((nRepeats,), np.nan)

            for i in range(nRepeats):
                try:
                    key_names = f"{hyper_param}{val:.{decimals}f}_CVfold{f}_rep{i+1}"
                    var_dfi = vars_dict_set_d[key_names]
    
                    # Extract relevant metrics
                    nLP_training_allreps[i] = var_dfi['nPL_training_data'].item()
                    nLL_training_allreps[i] = var_dfi['nLL_training_data'].item()
                    nLL_test_allreps[i] = var_dfi['nLL_heldout_data'].item()
                except:
                    print("Not found.")

            # Use the repetition with the lowest nLP on training data
            idx_i = np.argmin(nLP_training_allreps)

            # Store the best metrics for this decay rate and fold
            nLP_training[d, f] = nLP_training_allreps[idx_i]
            nLL_training[d, f] = nLL_training_allreps[idx_i]
            nLL_test[d, f] = nLL_test_allreps[idx_i]
            
            # ell parameters
            var_dfi_min_nLP = vars_dict_set_d[f"{hyper_param}{val:.{decimals}f}_CVfold{f}_rep{idx_i+1}"]
            # Extract ellipse parameters at each grid point
            for k in range(NUM_GRID_PTS):
                for l in range(NUM_GRID_PTS):
                    # Format: (x0, y0, a, b, theta)
                    params_ell[d, f, k, l] = var_dfi['model_pred_Wishart'].params_ell[k][l]

    # Average and std dev across folds (for plotting or model selection)
    nLL_test_avg = np.nanmean(nLL_test, axis=1)
    nLL_training_avg = np.nanmean(nLL_training, axis=1)
    nLL_test_std = np.nanstd(nLL_test, axis=1)
    nLL_training_std = np.nanstd(nLL_training, axis=1)
    
    # -----------------------------------------------------------------------
    # Visualize nLL as a function of decay rate
    # -----------------------------------------------------------------------  
    # Replace 'Experiment_DataFiles' with 'Experiment_FigFiles' in the file path
    output_fileDir_figs_set = input_fileDir_fits_set.replace('Experiment_DataFiles', 
                                                             'Experiment_FigFiles')
    os.makedirs(output_fileDir_figs_set, exist_ok=True)
    
    # Standardize figure file name by removing numeric value from 'varyingDecayRate'
    fig_name = re.sub(fr'varying{hyper_param}[\d.]+', f'varying{hyper_param}', file_name_set[:-4])
    
    # Configure plotting settings
    # Base plotting settings (e.g., figure directory and font size)
    pltSettings_base = PlotSettingsBase(fig_dir=output_fileDir_figs_set, fontsize=9)
    
    # Initialize figure-specific settings by merging with base settings
    nLL_vis_settings = replace(PltVaryingHyperParamSettings(), **pltSettings_base.__dict__)
    nLL_vis_settings = replace(nLL_vis_settings,
                               fig_size = (5, 3),
                               xticks = [1e-6, 3e-4, 1e-3, 3e-3] \
                                   if hyper_param == 'VarianceScale' else None,
                               xticklabels = [r"$10^{-6}$", "0.0003", "0.001", "0.003"] \
                                   if hyper_param == 'VarianceScale' else None,
                               xlabel = fr"Hyperparameter ${hp_symbol}$",
                               fig_name=fig_name)
    
    # Instantiate plotter
    nLL_vis = NFoldsCrossValidationVisualization(pltSettings_base, save_fig=True)
    
    # Prepare confidence intervals
    nLL_training_CI = [np.nanmin(nLL_training, axis=1), np.nanmax(nLL_training, axis=1)]
    nLL_test_CI = [np.nanmin(nLL_test, axis=1), np.nanmax(nLL_test, axis=1)]
    
    #Generate plot
    nLL_vis.plot_nLL_varying_hyper_param(hyper_param_arr, nLL_training_avg, nLL_test_avg, 
                                        nLL_training_CI, nLL_test_CI, total_folds, nLL_vis_settings)
                        
#%% 
# -----------------------------------------------------------------------------
# Visualize how the model predictions change as a function of the decay rate
# -----------------------------------------------------------------------------  
if not flag_running_on_hpc:
    # New shape: (n_hyper_param, ref_dim1, ref_dim2, n_folds, ellipse_params)
    params_ell_trans = np.transpose(params_ell, (0, 2, 3, 1, 4))
    
    # Create base settings shared across all plots
    pltSettings_base = PlotSettingsBase(fig_dir=output_fileDir_figs_set, fontsize=8)

    # Customize settings for 2D prediction plots
    pred2D_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
    pred2D_settings = replace(pred2D_settings,
                              visualize_samples=False,
                              visualize_gt=False,
                              visualize_model_estimatedCov=False,
                              flag_rescale_axes_label=False,
                              visualize_model_pred=False,
                              ticks=np.linspace(-0.7, 0.7, 5))

    for d, val in enumerate(hyper_param_arr):
        # Compute inner and outer contours (confidence bounds) from fitted ellipses
        fitEll_min, fitEll_max = find_inner_outer_contours_for_gridRefs(params_ell_trans[d])

        # Create figure and axes for current decay rate
        fig_d, ax_d = plt.subplots(1, 1, figsize=pred2D_settings.fig_size, dpi=pred2D_settings.dpi)
        pred2D_settings = replace(pred2D_settings,
                                  title=fr"Hyperparameter ${hp_symbol} = {val:.{decimals}f}$",
                                  fig_name=f'{fig_name}_{hyper_param}{val:.{decimals}f}.pdf')

        # Initialize visualization class with model predictions and plotting settings
        wishart_pred_vis_wCI = WishartPredictionsVisualization(
            None, var_dfi['model'], var_dfi['model_pred_Wishart'], color_thres_data,
            settings=pltSettings_base, save_fig=True
        )

        # Plot confidence ellipses across the 2D grid
        for i in range(NUM_GRID_PTS):
            for j in range(NUM_GRID_PTS):
                cm = color_thres_data.W2D_to_rgb(grid[i,j])
                lbl = f'Full range of model predictions evaluated \nusing {total_folds}'+\
                    '-fold cross-validation' if (i == 0 and j == 0) else None
                add_CI_ellipses(
                    fitEll_min[i, j], fitEll_max[i, j], ax=ax_d,
                    cm=cm, label=lbl, lw_inner = 0, lw_outer=1, alpha=0.9
                )

        # Overlay joint model predictions on top of CI ellipses
        wishart_pred_vis_wCI.plot_2D(grid, ax=ax_d, settings=pred2D_settings)
    
    # --------------------------------------------------------------------------------
    # Visualize how the variance of weight change with different hyperparameter values
    # --------------------------------------------------------------------------------  
    #add the prior to the plot
    model = var_dfi['model']
    basis_orders = np.arange(0,9,1)
    
    if hyper_param == 'DecayRate':
        gamma = base_kwargs['variance_scale']
        eta_all = [gamma *  p ** basis_orders for p in hyper_param_arr]
    else:
        epsilon = base_kwargs['decay_rate']
        eta_all = [p *  epsilon ** basis_orders for p in hyper_param_arr]
    
    # --- colors: sample extra + drop brightest end ---
    cmap_full = plt.get_cmap("bone", n_hyper_param + 2)
    colors = cmap_full(np.arange(n_hyper_param))  # keep first n_hyper_param colors (avoid whitest)
    
    cmap = mpl.colors.ListedColormap(colors, name="bone_trunc_discrete")
    
    # --- map actual hyperparameter values to [0,1] for the colorbar ---
    norm = mpl.colors.Normalize(vmin=float(np.min(hyper_param_arr)),
                                vmax=float(np.max(hyper_param_arr)))
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 2), dpi=1024)
    
    for n, val in enumerate(hyper_param_arr):
        ax.plot(basis_orders, eta_all[n], color=cmap(norm(val)), lw=1.5)
        if np.abs(val - base_kwargs[sweep_arg]) < 1e-10:
            ax.plot(basis_orders, eta_all[n], color='yellow', lw=1, 
                    ls = '--', label = fr'Our choice: ${hp_symbol}$={val:.{decimals}f}')
    ax.legend(loc = 'upper right')
    ax.set_xlabel(r'The order of 2D Chebyshev basis functions $(i + j)$')
    ax.set_ylabel('Variance of weights\n' + r'$(\eta_{i+j} = \gamma \cdot \epsilon^{i+j})$')
    
    # colorbar whose scale is in *hyper_param values*
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(fr"Hyperparameter ${hp_symbol}$")
    
    # --- ticks: use hyper_param_arr (but subsample to avoid clutter) ---
    tick_idx = [0, n_hyper_param-1]
    tick_vals = hyper_param_arr[tick_idx]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{v:.3g}" for v in tick_vals])
    fig.tight_layout()
    fig.savefig(os.path.join(output_fileDir_figs_set,
                             f"prior_variance_vs_basis_order_varying{hyper_param}.pdf"),
                bbox_inches="tight")

