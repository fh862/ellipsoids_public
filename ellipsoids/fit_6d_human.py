#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 21:55:19 2025

@author: fangfang

This script fits the Wishart model to 3D / 6D experimental data. 
Because fitting can be computationally intensive, different parts of the workflow 
are split across HPC (for fitting and prediction) and local machines (for visualization). 

The code may look unconventional at first glance, but the structure is designed 
to work around hardware constraints on the cluster.

There are three main usage modes, controlled by flags:

1. flag_running_on_hpc = True; flag_using_gpu = True
   - Task: Fit the Wishart model (estimate W_est).
   - Hardware: Cluster GPUs.
   - Job script: runPython_wPytorch.sbatch
   - Typical runtime: ~1h.

2. flag_running_on_hpc = True; flag_using_gpu = False
   - Task: Compute model predictions using the best W_est.
   - Hardware: Cluster CPUs (preferred, since this step exceeds GPU memory limits).
   - Job script: runPython_cpu.sbatch
   - Typical runtime: ~2h.

3. flag_running_on_hpc = False
   - Task: Load .pkl results and visualize model predictions locally.
   - Note: 'flag_using_gpu' is ignored in this mode.

Additional options:
- The script can fit both the original dataset and bootstrapped datasets.
- Use `flag_btst` to toggle bootstrapping, and `btst_seed` to control the random seed.

"""

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import dill as pickled
from dataclasses import replace
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import copy
import os
from core import optim, oddity_task
from core.wishart_process import WishartProcessModel
from core.model_predictions import wishart_model_pred
from analysis.color_thres import color_thresholds
from analysis.utils_load import load_expt_data
from analysis.utils_load import select_file_and_get_path
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization, \
    Plot2DSamplingSettings
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization,\
    WishartPredictionsVisualization_html, Plot3DPredSettings, Plot3DPredHTMLSettings
from plotting.adaptive_sampling_plotting import Plot3DSamplingHTMLSettings,\
    SamplingRefCompPairVisualization_html

#%%
# -----------------------------------------------------------
# SECTION 1: set directories
# -----------------------------------------------------------
# flag for using the cluster and using gpus
flag_running_on_hpc = False
flag_using_gpu = True

# base directory
base_dir = os.path.dirname(__file__) if flag_running_on_hpc else \
    '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

#specify subject information
subN = 1
stim_dims = 3      # the stimulus is 3D
psyfield_dims = 6  # 6D psychometric field 
nSessions = 40     #total number of sessions
path_str  = os.path.join(base_dir,'ELPS_analysis','Experiment_DataFiles', 
                         f'{psyfield_dims}D_Expt', f'sub{subN}')
                         #'pilot3',f'sub{subN}')

# Define the color plane and load for transformation matrices
color_thres_data = color_thresholds(stim_dims, os.path.join(base_dir))
color_thres_data.load_transformation_matrix(file_date= '02242025')

# Set output directory for saving model fitting results or plots
output_fileDir_fits = os.path.join(base_dir, 'hpc_sweeps', 'model_fitting',
                                   f'sub{subN}', 
                                   f'{stim_dims}D{psyfield_dims}D')
os.makedirs(output_fileDir_fits, exist_ok=True)

#%%
# -----------------------------------------------------------------------
# SECTION 2: Define constant variables for model fitting and evaluation
# -----------------------------------------------------------------------
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions
    stim_dims, # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    3e-4,  # Scale parameter for prior on `W`.
    0.5,   # Geometric decay rate on `W`. default = 0.4
    0,     # Diagonal term setting minimum variance for the ellipsoids.
    num_dims_cov= stim_dims
)

# Number of grid points over stimulus space.
NUM_GRID_PTS = 5      

opt_params = {
    "learning_rate": 1e-4,
    "momentum": 0.2,
    "mc_samples": 2000, # Number of simulated trials to compute likelihood.
    "bandwidth": 5e-3,  # Bandwidth for logistic density function.
}

# Search for the best-fit parameters from three random initializations to avoid
# getting stuck at local minimum
nRepeats = 1

# Generate a multidimensional grid based on the number of color dimensions
max_val = 0.7
grid = jnp.stack(jnp.meshgrid(*[jnp.linspace(-max_val, max_val,
                    NUM_GRID_PTS) for _ in range(model.num_dims)]), axis=-1)

# probability that corresponds to threshold
target_pC = 0.667

#we want to fit the Wishart model to the original dataset as well as bootstrapped
#dataset with specified seed
flag_btst_list = [True]*10
btst_seed_list = list(range(10)) 

#%%
# ----------------------------------------------------------------
# SECTION 3: Load, organize and fit the model to the pilot data 
# ----------------------------------------------------------------
if flag_running_on_hpc:
    # Get file paths for all session data of the subject
    session_files, session_file_name_part1 = \
        load_expt_data.get_all_sessions_file_names(subN, nSessions, path_str,
                                                      exptCond = '_6dExpt_RGBcube')
        
    # set path
    for flag_btst, btst_seed in zip(flag_btst_list, btst_seed_list):
        str_ext = f'_btst_AEPsych[{btst_seed}]' if flag_btst else ''
        output_file = f"Fitted_{session_file_name_part1}_decayRate{model.decay_rate}"+\
            f"_nBasisDeg{model.degree}{str_ext}.pkl" 
        full_path = os.path.join(output_fileDir_fits, output_file)
        
        # we run model fitting on the GPUs
        if flag_using_gpu:        
            # Load session data from the files
            data_allSessions = load_expt_data.load_data_all_sessions(session_files)
                
            # Extract and concatenate AEPsych data across all sessions
            aepsych_data, sobol_data, combined_data = load_expt_data.load_combine_AEPsych_pregSobol(data_allSessions)
            xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list,\
                    xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed = aepsych_data
                    
            # Get the total number of AEPsych trials and for each strategy
            nTrials_AEPsych = y_AEPsych.shape[0]
            NTRIALS_STRAT = data_allSessions[0]['NTRIALS_STRAT']
            
            # Extract pregenerated Sobol data across all sessions
            xref_combined, x1_combined, y_combined = combined_data
                        
            # Pack the processed data into a tuple for further use.
            if flag_btst:
                #bootstrap AEPsych trials
                xref_jnp, x1_jnp, y_jnp, btst_indices = load_expt_data.bootstrap_AEPsych_data(\
                    xref_combined, x1_combined, y_combined,
                    trials_split=[sum(NTRIALS_STRAT[:-1]), nTrials_AEPsych],
                    seed=btst_seed)
            else:
                xref_jnp = xref_combined
                x1_jnp = x1_combined
                y_jnp = y_combined
            data = (y_jnp, xref_jnp, x1_jnp)
            nTrials_forFitting = xref_jnp.shape[0]
                
            # -----------------------------------------------------------------------
            # SECTION 3a: Fit the Wishart model
            # -----------------------------------------------------------------------
            #Generate a matrix of random seeds for each initialization
            random_seeds = np.random.randint(0, 2**12, size = (nRepeats, 2))
            # Initialize a high upper bound for negative log-likelihood (nLL) to track the best fit
            objhist_end = 1e3  # Start with a large number to ensure any valid fit is better
            
            # Loop through each random initialization
            for i in range(nRepeats):
                # Generate random keys for initializing parameters, data, and optimizer
                W_INIT_KEY_i   = jax.random.PRNGKey(random_seeds[i,0])  # Key to initialize `W_est`. 
                OPT_KEY_i      = jax.random.PRNGKey(random_seeds[i,1])  # Key passed to optimizer.
                
                # Fit model, initialized at a random W sampled from the prior distribution
                W_init_i = 1e-1*model.sample_W_prior(W_INIT_KEY_i) 
                
                W_recover_i, iters_i, objhist_i = optim.optimize_posterior(
                    W_init_i, data, model, OPT_KEY_i,
                    copy.deepcopy(opt_params),
                    oddity_task.simulate_oddity, 
                    total_steps=4000,
                    save_every=1,
                    show_progress=True
                )
                
                # Update the best-fit model if the current fit improves the objective (lower nLL)
                if objhist_i[-1] < objhist_end:
                    objhist_end = objhist_i[-1]
                    W_init, W_est, iters, objhist = W_init_i, W_recover_i, iters_i, objhist_i
                    W_INIT_KEY, OPT_KEY = W_INIT_KEY_i, OPT_KEY_i
                    bestfit_seed = random_seeds[i]
                    
            # -----------------------------------------------------------------------
            # SECTION 3b: Save the data
            # -----------------------------------------------------------------------            
            variable_names = ['subN','stim_dims','psyfield_dims','nSessions', 'color_thres_data',
                              'model', 'NUM_GRID_PTS', 'opt_params', 'nRepeats', 'grid',
                              'target_pC', 'btst_seed', 'flag_btst', 'data_allSessions', 
                              'nTrials_AEPsych', 'NTRIALS_STRAT', 'xref_jnp', 'x1_jnp',
                              'y_jnp', 'data', 'nTrials_forFitting', 'random_seeds', 
                              'objhist_end', 'W_init', 'W_est', 'objhist', 'W_INIT_KEY',
                              'OPT_KEY', 'bestfit_seed']
            vars_dict = {}
            for var_name in variable_names:
                try:
                    # Check if the variable exists in the global scope
                    vars_dict[var_name] = eval(var_name)
                except NameError:
                    # If the variable does not exist, assign None and print a message
                    vars_dict[var_name] = None
                    print(f"Variable '{var_name}' does not exist. Assigned as None.")
                    
            # Write the list of dictionaries to a file using pickle
            with open(full_path, 'wb') as f:
                pickled.dump(vars_dict, f)
            
        # we run model predictions on CPUs
        else:
            # Load the data dictionary from file
            with open(full_path, 'rb') as f:
                vars_dict = pickled.load(f)
            for k, v in vars_dict.items():
                globals()[k] = v
    
            # -----------------------------------------------------------------------
            # SECTION 3a: Compute the model predictions for a grid of reference
            # -----------------------------------------------------------------------
            # Compute the covariance matrices ('Sigmas') at each point in the grid using 
            # the model's compute_U function. 
            Sigmas_noise_grid = model.compute_Sigmas(model.compute_U(W_est, grid))
            
            # Initialize the Wishart model prediction using various parameters.
            model_pred_Wishart = wishart_model_pred(model, opt_params, W_INIT_KEY,
                                                    OPT_KEY, W_init,
                                                    W_est, Sigmas_noise_grid,
                                                    color_thres_data, 
                                                    target_pC= target_pC,
                                                    ngrid_bruteforce = 1000,
                                                    bds_bruteforce = [0.0005, 0.3])
            
            # batch compute 66.7% threshold contour based on estimated weight matrix
            model_pred_Wishart.convert_Sig_Threshold_oddity_batch(grid)
            
            # -----------------------------------------------------------------------
            # SECTION 3b: Compute the model predictions on an isolulminant plane
            # -----------------------------------------------------------------------
            # Compute model predictions for a grid of reference on the isoluminant plane      
            # extract points on the isoluminant plane
            NUM_GRID_PTS_2D = 7
            grid_1d = jnp.linspace(-max_val, max_val, NUM_GRID_PTS_2D)
            grid_isoluminant_2DW = jnp.stack(jnp.meshgrid(*[grid_1d for _ in range(2)]), axis=-1)
            
            # we need to extract the coordinates of the grid of reference stimuli on
            # the isoluminant plane
            # 2D model space -> RGB in 3D -> 3D model space
            grid_isoluminant_RGB = color_thres_data.W2D_to_rgb(grid_isoluminant_2DW.reshape(-1, 2))[None, None]
            grid_isoluminant_3DW = color_thres_data.N_unit_to_W_unit(grid_isoluminant_RGB)
            
            #covariance matrices
            Sigmas_noise_grid_isoluminant = model.compute_Sigmas(model.compute_U(W_est, grid_isoluminant_3DW))
            
            #model predictions
            model_pred_Wishart_grid_isoluminant = \
                wishart_model_pred(model, opt_params, W_INIT_KEY, OPT_KEY, W_init,
                                   W_est, Sigmas_noise_grid_isoluminant, color_thres_data, 
                                   target_pC = target_pC, ngrid_bruteforce = 1000,
                                   bds_bruteforce = [0.0005, 0.3])
            model_pred_Wishart_grid_isoluminant.convert_Sig_Threshold_oddity_batch(grid_isoluminant_3DW)
            
            # Append new variables to the dictionary
            vars_dict.update({
                "Sigmas_noise_grid": Sigmas_noise_grid,
                "model_pred_Wishart": model_pred_Wishart,
                "NUM_GRID_PTS_2D": NUM_GRID_PTS_2D,
                "Sigmas_noise_grid_isoluminant": Sigmas_noise_grid_isoluminant,
                "model_pred_Wishart_grid_isoluminant": model_pred_Wishart_grid_isoluminant,
                "grid_isoluminant_RGB": grid_isoluminant_RGB,
                "grid_isoluminant_3DW": grid_isoluminant_3DW,
                "grid_isoluminant_2DW": grid_isoluminant_2DW,
            })
            
            # Save updated dictionary back to the same file
            with open(full_path, 'wb') as f:
                pickled.dump(vars_dict, f)
else:
    # -----------------------------------------------------------------------
    # SECTION 4: Visualize model predictions
    # -----------------------------------------------------------------------
    # Prompt user to select a cross-validation results file.
    # Navigate to ELPS_analysis/Experiment_DataFiles/sub#
    # 'CrossValidation5folds_varyingDecayRate_ColorDiscrimination_4dExpt_Isoluminant plane_sub11.pkl'
    input_fileDir_fits, file_name = select_file_and_get_path()

    # Construct the full path to the selected file
    full_path = os.path.join(input_fileDir_fits, file_name)

    # Load the data dictionary from file
    with open(full_path, 'rb') as f:
        vars_dict = pickled.load(f)
    for k, v in vars_dict.items():
        globals()[k] = v
            
    # Create the output directory if it doesn't exist
    output_figDir_fits = input_fileDir_fits.replace('DataFiles', 'FigFiles')
    os.makedirs(output_figDir_fits, exist_ok=True)
    
    # Create a base plotting settings instance (shared across plots)
    pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 11)
    
    # Initialize 2D prediction settings by copying from base and overriding method-specific parameters
    predM_settings = replace(Plot3DPredSettings(), **pltSettings_base.__dict__)
    predM_settings = replace(predM_settings,
                             visualize_gt = False,
                             visualize_samples = False,
                             visualize_model_pred = True,
                             samples_s = 1,
                             samples_alpha = 0.2, 
                             modelpred_alpha = 0.55,
                             modelpred_lc = None,
                             contour_3D_label = '3D predictions',
                             contour_2D_label = '2D predictions',
                             fig_name = f"{file_name[:-4]}") 
            
    # Initialize Visualization Class for Wishart Predictions
    wishart_pred_vis = WishartPredictionsVisualization(None,
                                                       model_pred_Wishart.model, 
                                                       model_pred_Wishart, 
                                                       color_thres_data,
                                                       settings = pltSettings_base,
                                                       save_fig = False)
    # Create figure and axes for plotting
    wishart_pred_vis.plot_3D(grid, settings = predM_settings)

#%%
# -----------------------------------------------------------------------
# SECTION 5: Visualize trial placement (separately for Sobol and EAVC)
# -----------------------------------------------------------------------
if not flag_running_on_hpc and not flag_btst:
    # Create settings instance with custom fig_dir
    pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
    
    # This array defines the opacity of markers in the plots, decreasing with more trials.
    marker_alpha = [0.2, 0.2, 0.2]
    # Define specific slices of data points to be visualized, ranging from very few to many.
    slc_datapoints_to_show_lb = [0, sum(NTRIALS_STRAT[:-1]), nTrials_AEPsych]
    slc_datapoints_to_show_ub = [sum(NTRIALS_STRAT[:-1]), nTrials_AEPsych, nTrials_forFitting]
    
    # Loop over the selected data points to generate and visualize each corresponding figure.
    for i, (lb_i, ub_i) in enumerate(zip(slc_datapoints_to_show_lb, slc_datapoints_to_show_ub)):
        # Construct a filename for each figure based on the plane and number of experiments.
        str_idx = f'{ub_i:05}total' if lb_i == 0 else f'{ub_i:05}total_from{lb_i:05}'
        fig_name = f"TrialPlacement_isothreshold_{psyfield_dims}DExpt_{str_idx}_sub{subN}"
        pltSampSettings = replace(pltSettings_tp, 
                                  fontsize = 11,
                                  fig_size = (6,6),
                                  linealpha = marker_alpha[i],  # Line transparency for this subset of data
                                  bounds = 0.75 *np.array([-1,1]),
                                  flag_rescale_axes_label = False,
                                  ticks = np.linspace(-0.7,0.7,5),
                                  ref_markeralpha = 0.5,
                                  comp_markeralpha = marker_alpha[i], # Marker transparency for this subset of data
                                  fig_name = fig_name)

        fig = plt.figure(figsize = pltSampSettings.fig_size, dpi= pltSampSettings.dpi, constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        sampling_vis_az = SamplingRefCompPairVisualization(stim_dims,
                                                           color_thres_data,
                                                           settings = pltSampSettings,
                                                           save_fig = False)
        
        # Visualize the trials up to the nth data point with specified marker transparency.
        sampling_vis_az.plot_sampling(xref_jnp[lb_i:ub_i],  # Reference points up to the nth data point
                                      x1_jnp[lb_i:ub_i],    # Comparison points up to the nth data point
                                      ax = ax,
                                      settings = pltSampSettings) 
        # Save the figure as a PDF
        ax.view_init(30, -60)
        fig.savefig(os.path.join(output_figDir_fits, f"{fig_name}.pdf"))
        plt.pause(0.01)
        plt.show()
            
    #create a figure
    fig2 = plt.figure(figsize = pltSampSettings.fig_size, dpi= pltSampSettings.dpi, constrained_layout=True)
    ax2 = fig2.add_subplot(111, projection='3d')       
    eps = 1e-8
    non_edge_mask = jnp.all(jnp.abs(xref_jnp) < 0.85 - eps, axis=1)
    non_edge_idx = jnp.where(non_edge_mask)[0]
    non_edge_non_sobol_idx = non_edge_idx[(non_edge_idx > sum(NTRIALS_STRAT[:-1])) & (non_edge_idx < nTrials_AEPsych)]
    nTrials_non_edge_non_sobol = non_edge_non_sobol_idx.shape[0]
    # Visualize the trials up to the nth data point with specified marker transparency.
    sampling_vis_az.plot_sampling(xref_jnp[non_edge_non_sobol_idx],  # Reference points up to the nth data point
                                  x1_jnp[non_edge_non_sobol_idx],    # Comparison points up to the nth data point
                                  ax = ax2,
                                  settings = pltSampSettings) 
    # Save the figure as a PDF
    fig2.savefig(os.path.join(output_figDir_fits, 
                              f"TrialPlacement_isothreshold_{psyfield_dims}DExpt_nonedgeTrials_sub{subN}.pdf"))    
    plt.show()

    #%% Visualize
    # Visualization helper with HTML settings
    vis_pred_html = WishartPredictionsVisualization_html(settings=Plot3DPredHTMLSettings())

    fig2 = go.Figure()
    # Render 3D ellipsoids (mesh surfaces) evaluated on the isoluminant plane
    vis_pred_html.plot_ellipsoids_mesh(fig2, model_pred_Wishart)
    # Apply consistent 3D layout (camera, axes, lighting, hover behavior)
    vis_pred_html.apply_3d_layout(fig2)
    # Save interactive HTML
    html_dir = os.path.join(os.path.dirname(output_figDir_fits), 'html')
    os.makedirs(html_dir, exist_ok=True)
    out_html = os.path.join(html_dir, "Ellipsoids_Fitted_ColorDiscrimination_"+\
                            f"{psyfield_dims}dExpt_RGBcube_sub{subN}_decayRate"+\
                            f"{model.decay_rate}_nBasisDeg{model.degree}.html")
    fig2.write_html(out_html, include_plotlyjs=True)
    
    #%% Visualize adaptively sampled trials
    
    lb_i = sum(NTRIALS_STRAT[:-1])
    ub_i = nTrials_AEPsych    
    xref_adaptive = np.asarray(xref_jnp[lb_i:ub_i], float)
    x1_adaptive   = np.asarray(x1_jnp[lb_i:ub_i],   float)
    
    vis_sample_html = SamplingRefCompPairVisualization_html(settings=Plot3DSamplingHTMLSettings())

    fig3 = vis_sample_html.plot_sampling(xref_adaptive, x1_adaptive)
    
    out_html = os.path.join(html_dir, f"TrialPlacement_{psyfield_dims}DExpt_sub{subN}.html")
    fig3.write_html(out_html, include_plotlyjs=True)
    