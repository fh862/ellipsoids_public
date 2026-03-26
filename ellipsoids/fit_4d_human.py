#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 00:03:09 2025

@author: fangfang

This script loads session-level raw data from the 2D stimulus / 4D psychometric
experiment and fits the Wishart Process Psychophysical Model (WPPM) to
comprehensively characterize color discrimination thresholds across the
stimulus space.

In addition to model fitting, the script visualizes:
    (1) Trial placement selected by AEPsych during adaptive sampling
    (2) Model-predicted discrimination thresholds evaluated on a grid of
        reference locations
        
If this is run on hpc, use runPython_wPytorch.sbatch 

#!/bin/bash
#SBATCH --job-name=cross_validate_find_opt_decay_rate
#SBATCH --output=slurm_scripts/slurm%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=fh862@sas.upenn.edu
#SBATCH -p gpu -N1 -G1 --constraint=h100 --cpus-per-task=4 --mem-per-cpu=20G
#SBATCH --time=08:00:00

(for 120 bootstrap datasets)

"""

# Toggle between HPC batch mode and local/interactive mode.
flag_running_on_hpc = False
import matplotlib
import matplotlib.pyplot as plt
if flag_running_on_hpc: 
    matplotlib.use('Agg')   # must come before pyplot import
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import dill as pickled
from dataclasses import replace
import numpy as np
np.random.seed(None)
import os
from core import optim, oddity_task
from core.wishart_process import WishartProcessModel
from core.model_predictions import wishart_model_pred
from analysis.color_thres import color_thresholds
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization,\
    Plot2DPredSettings
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization, \
    Plot2DSamplingSettings
from plotting.wishart_plotting import PlotSettingsBase 
from analysis.utils_load import load_expt_data
from analysis.cross_validation import expt_data
from dconfig.config_4Ddata import DatasetConfig_4D

#%%
# -----------------------------------------------------------
# SECTION 1: set directories
# -----------------------------------------------------------
# Base directory where data lives. On HPC, prefer paths relative to the script.
base_dir = os.path.dirname(__file__) if flag_running_on_hpc else \
    '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

stim_dims = 2
psyfield_dims = 4
subN = 15

# choose one dataset
dcfg = DatasetConfig_4D.human_ls_isolating(base_dir, subN)
# dcfg = DatasetConfig_4D.human_isoluminant(base_dir, subN)
# dcfg = DatasetConfig_4D.human_varying_background(base_dir, subN)
# dcfg = DatasetConfig_4D.simulated_isoluminant(base_dir, subN)

dcfg.print_summary()
nSession = dcfg.nSession

# modify anything if needed
# dcfg.nSession = 7
# dcfg.__post_init__()   # rerun only if you changed something that affects derived fields

# Define the color plane and load for transformation matrices
color_thres_data = color_thresholds(2, base_dir, plane_2D = dcfg.plane_2D) 
color_thres_data.load_transformation_matrix(file_date = dcfg.file_date)  

#specify figure name and path
if flag_running_on_hpc:
    output_fileDir_fits = os.path.join(base_dir, 'hpc_sweeps', 'model_fitting',
                                       dcfg.coloralg if not dcfg.flag_load_datafile else '',
                                       f'sub{subN}', f'{stim_dims}D{psyfield_dims}D')
    output_figDir_fits = os.path.join(output_fileDir_fits, 'figs')    
else:
    output_fileDir_fits = os.path.join(dcfg.path_str,'fits')
    output_figDir_fits = dcfg.path_str.replace('DataFiles', 'FigFiles')
    
# Create directories if they don't exist
os.makedirs(output_figDir_fits, exist_ok=True)
os.makedirs(output_fileDir_fits, exist_ok=True)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 8)

#%%
# -----------------------------------------------------------
# SECTION 2: Load and organize the pilot data 
# -----------------------------------------------------------
#we want to fit the Wishart model to the original dataset as well as bootstrapped
#dataset with specified seed
btst_seed = [None] #+ list(range(10)) 
flag_btst = [False] #+ [True]*10 

for flag_btst_AEPsych, ll in zip(flag_btst, btst_seed):
    str_ext = dcfg.str_ext_s
    if flag_btst_AEPsych: str_ext += f'_btst_AEPsych[{ll}]'
    
    if dcfg.flag_load_datafile:
        # Get file paths for all session data of the subject
        session_files, session_file_name_part1 = \
            load_expt_data.get_all_sessions_file_names(subN, 
                                                       nSession, 
                                                       dcfg.path_str, 
                                                       exptCond = dcfg.exptCond, 
                                                       str_ext = f'{dcfg.adaptation_cond_str}_copy')
        
        # Load session data from the files
        data_allSessions = load_expt_data.load_data_all_sessions(session_files)
        
        # In the original 4D experiment on the isoluminant plane, we interleaved validation
        # trials. In the following adaptation experiment, we removed the validation MOCS
        # trials as there is no need to validate our approach again.
        try:
            # Extract and concatenate MOCS data across all sessions
            xref_MOCS_list, x1_MOCS_list, y_MOCS_list, xref_MOCS, x1_MOCS, y_MOCS = \
                load_expt_data.load_MOCS_data(data_allSessions)
            
            # Organize MOCS trials by unique reference stimulus conditions
            xref_unique_MOCS, nRefs_MOCS, refStimulus_MOCS, compStimulus_MOCS,\
                    responses_MOCS, nLevels_MOCS, nTrials_MOCS, _, _, _ = \
                        load_expt_data.org_MOCS_by_condition(xref_MOCS, x1_MOCS, y_MOCS)    
        except:
            print('MOCS trials are not found in this dataset.')
        
        # Extract and concatenate AEPsych data across all sessions
        aepsych_data, sobol_data, combined_data = \
            load_expt_data.load_combine_AEPsych_pregSobol(data_allSessions)
        xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list,\
                xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed = aepsych_data
        # Get the total number of AEPsych trials and for each strategy
        nTrials_AEPsych = y_AEPsych.shape[0]
        nTrials_strat = data_allSessions[0]['NTRIALS_STRAT']
        
        # Extract pregenerated Sobol data across all sessions
        xref_combined, x1_combined, y_combined = combined_data
    else:
        session_file_name_part1 = dcfg.file_name[:-4]
        full_path = os.path.join(dcfg.path_str, dcfg.file_name)

        with open(full_path, 'rb') as f:
            vars_dict = pickled.load(f)
            
        nTrials_strat = list(vars_dict['strat_dict'].values())
        nTrials_AEPsych = vars_dict['NTRIALS']
        
        xref_combined = jnp.array(vars_dict['AEPsych_trial_given_Wishart_gt'].xref_all)
        x1_combined = jnp.array(vars_dict['AEPsych_trial_given_Wishart_gt'].x1_all)
        y_combined = jnp.array(vars_dict['AEPsych_trial_given_Wishart_gt'].binaryResp_all)
                
    # Pack the processed data into a tuple for further use.
    if flag_btst_AEPsych:
        #bootstrap AEPsych trials
        xref_jnp, x1_jnp, y_jnp, btst_indices = load_expt_data.bootstrap_AEPsych_data(\
            xref_combined, x1_combined, y_combined, 
            trials_split=[sum(nTrials_strat[:-1]), nTrials_AEPsych],seed=ll)
    else:
        xref_jnp = xref_combined
        x1_jnp = x1_combined
        y_jnp = y_combined
    data = (y_jnp, xref_jnp, x1_jnp)
    nTrials_forFitting = xref_jnp.shape[0]
    # for sanity check, make the plot below. The dots should be within [-0.3, 0.3] for both axes
    # plt.scatter(x1_jnp[:,0] - xref_jnp[:,0], x1_jnp[:,1] - xref_jnp[:,1])
        
    #%
    # -----------------------------------------------------------------------
    # SECTION 3: Visualize trial placement (separately for Sobol and EAVC)
    # -----------------------------------------------------------------------
    # Create settings instance with custom fig_dir
    pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
    sampling_vis = SamplingRefCompPairVisualization(stim_dims,
                                                    color_thres_data,
                                                    settings = pltSettings_tp,
                                                    save_fig = False
                                                    )
    
    # This array defines the opacity of markers in the plots, decreasing with more trials.
    marker_alpha = [0.3, 0.2, 0.2] #[0.5, 0.3, 0.7]
    # Define specific slices of data points to be visualized, ranging from very few to many.
    slc_datapoints_to_show_lb = [0, sum(nTrials_strat[:-1]), nTrials_AEPsych]
    slc_datapoints_to_show_ub = [sum(nTrials_strat[:-1]), nTrials_AEPsych, nTrials_forFitting]
    
    # Loop over the selected data points to generate and visualize each corresponding figure.
    for i, (lb_i, ub_i) in enumerate(zip(slc_datapoints_to_show_lb, slc_datapoints_to_show_ub)):
        # Construct a filename for each figure based on the plane and number of experiments.
        str_idx = f'{ub_i:05}total' if lb_i == 0 else f'{ub_i:05}total_from{lb_i:05}'
        fig_name = f"TrialPlacement_isothreshold_{color_thres_data.plane_2D}_"+\
                    f"{psyfield_dims}DExpt_{str_idx}_sub{subN}{str_ext}{dcfg.adaptation_cond_str}"
        pltSettings_tp = replace(pltSettings_tp,
                                 ref_markeralpha = marker_alpha[i],
                                 comp_markeralpha = marker_alpha[i],
                                 linealpha = marker_alpha[i], 
                                 ticks = np.linspace(-0.7, 0.7, 5),
                                 bounds = 0.75 * np.array([-1,1]),
                                 fig_name = fig_name
                                 )
    
        fig, ax = plt.subplots(1, 1, figsize = (3,3.5), dpi= pltSettings_tp.dpi)
        # Visualize the trials up to the nth data point with specified marker transparency.
        sampling_vis.plot_sampling(xref_jnp[lb_i:ub_i],  # Reference points up to the nth data point
                                   x1_jnp[lb_i:ub_i],    # Comparison points up to the nth data point
                                   settings = pltSettings_tp,
                                   ax = ax
                                   )            
        ax.set_title(color_thres_data.plane_2D)
        # Save the figure as a PDF
        fig.savefig(os.path.join(output_figDir_fits, f"{fig_name}.pdf"), bbox_inches='tight')    
        plt.show()
    
    #%
    # -----------------------------------------------------------------------
    # SECTION 4: Fit the Wishart model
    # -----------------------------------------------------------------------
    model = WishartProcessModel(
        5,         # Degree of the polynomial basis functions
        stim_dims, # Number of stimulus dimensions
        1,         # Number of extra inner dimensions in `U`.
        3e-4,      # Scale parameter for prior on `W`.
        0.5,       # Geometric decay rate on `W`. default = 0.4
        0,         # Diagonal term setting minimum variance for the ellipsoids.
    )

    opt_params = {
        "learning_rate": 1e-4,
        "momentum": 0.2,
        "mc_samples": 2000,   # Number of simulated trials to compute likelihood.
        "bandwidth": 5e-3,    # Bandwidth for logistic density function.
    } 
    
    # Search for the best-fit parameters from three random initializations to avoid
    # getting stuck at local minimum
    nRepeats = 3
    #Generate a matrix of random seeds for each initialization
    random_seeds = np.random.randint(0, 2**32, size = (nRepeats, 2))
    # Initialize a high upper bound for negative log-likelihood (nLL) to track the best fit
    objhist_end = 1e3  # Start with a large number to ensure any valid fit is better
    # Loop through each random initialization
    for i in range(nRepeats):
        # Generate random keys for initializing parameters, data, and optimizer
        W_INIT_KEY_i   = jax.random.PRNGKey(random_seeds[i,0])  # Key to initialize `W_est`. 
        OPT_KEY_i      = jax.random.PRNGKey(random_seeds[i,1])  # Key passed to optimizer.
        
        # Fit model, initialized at a random W sampled from the prior distribution
        W_init_i = model.sample_W_prior(W_INIT_KEY_i) 
        
        W_est_i, iters_i, objhist_i = optim.optimize_posterior(
            W_init_i, data, model, OPT_KEY_i,
            opt_params,
            oddity_task.simulate_oddity, 
            total_steps=1500,
            save_every=1,
            show_progress=True
        )
        
        # Update the best-fit model if the current fit improves the objective (lower nLL)
        if objhist_i[-1] < objhist_end:
            objhist_end = objhist_i[-1]
            W_init, W_est, iters, objhist = W_init_i, W_est_i, iters_i, objhist_i
            W_INIT_KEY, OPT_KEY = W_INIT_KEY_i, OPT_KEY_i
            bestfit_seed = random_seeds[i]
            
    fig, ax = plt.subplots(1, 1)
    ax.plot(iters, objhist)
    fig.tight_layout(); plt.show()
    
    #%
    # -------------------------------------------------------
    # SECTION 5: Compute model predictions (66.7% correct )
    # -------------------------------------------------------
    # Generate a multidimensional grid based on the number of color dimensions
    grid = dcfg.grid
    # Compute the covariance matrices ('Sigmas') at each point in the grid using 
    # the model's compute_U function. 
    Sigmas_noise_grid = model.compute_Sigmas(model.compute_U(W_est, grid))
    
    # Initialize the Wishart model prediction using various parameters.
    target_pC = 0.667
    model_pred_Wishart = wishart_model_pred(model, opt_params, W_INIT_KEY,
                                            OPT_KEY, W_init,
                                            W_est, Sigmas_noise_grid,
                                            color_thres_data, 
                                            target_pC=target_pC,
                                            ngrid_bruteforce = 1000,
                                            bds_bruteforce = dcfg.bds_bruteforce
                                            )
    
    # batch compute 66.7% threshold contour based on estimated weight matrix
    model_pred_Wishart.convert_Sig_Threshold_oddity_batch(grid)
    
    # compute the covariance matrices at all the sampled reference/comparison stimuli
    Sigmas_noise_xref = model.compute_Sigmas(model.compute_U(W_est, xref_jnp))
    Sigmas_noise_x1 = model.compute_Sigmas(model.compute_U(W_est, x1_jnp))
    
    #compute mahalanobis distance
    model_pred_Wishart.compute_Mahalanobis_distance_batch(xref_jnp,
                                                          x1_jnp,
                                                          Sigmas_noise_xref,
                                                          Sigmas_noise_x1
                                                          )
    #%
    # -----------------------------------------
    # SECTION 6: Visualize model predictions
    # -----------------------------------------
    #specify figure name and path
    fig_name_part1 = f"Fitted_{session_file_name_part1}_decayRate{model.decay_rate}"+\
        f"_varScaler{model.variance_scale}_nBasisDeg{model.degree}{str_ext}{dcfg.adaptation_cond_str}" 
    pred2D_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
    pred2D_settings = replace(pred2D_settings, 
                              visualize_samples= False,
                              visualize_gt = False,
                              modelpred_alpha = 1,
                              modelpred_lw = 1.5,
                              ticks = np.linspace(-0.7, 0.7,5),
                              fig_name = fig_name_part1
                              )
        
    expt_trial = expt_data(xref_jnp, x1_jnp, y_jnp)
    wishart_pred_vis = WishartPredictionsVisualization(expt_trial,
                                                       model, 
                                                       model_pred_Wishart, 
                                                       color_thres_data,
                                                       settings = pltSettings_base,
                                                       save_fig = False
                                                       )
    
    #visualize samples and model-estimated cov matrices
    #customize cmap for the isoluminant plane
    fig1, ax1 = plt.subplots(1, 1, figsize = pred2D_settings.fig_size, dpi= pred2D_settings.dpi)
    wishart_pred_vis.plot_2D(grid, settings = pred2D_settings, ax = ax1)
    ax1.set_title(color_thres_data.plane_2D);
    # Save the figure as a PDF
    fig1.savefig(os.path.join(output_figDir_fits,f"{fig_name_part1}.pdf"),
                 format='pdf', bbox_inches='tight')
    plt.show()
    
    
    #% save data
    output_file = f'{fig_name_part1}.pkl'
    full_path = os.path.join(output_fileDir_fits, output_file)
    
    variable_names = ['subN', 'stim_dims', 'psyfield_dims','color_thres_data',
                      'nSessions', 'btst_seed', 'data_allSessions', 'nTrials_strat',
                      'xref_MOCS_list', 'x1_MOCS_list', 'y_MOCS_list', 'xref_MOCS', 
                      'x1_MOCS','y_MOCS', 'xref_unique_MOCS', 'nRefs_MOCS', 
                      'nTrials_MOCS','refStimulus_MOCS','compStimulus_MOCS', 
                      'responses_MOCS', 'nLevels_MOCS', 'aepsych_data',
                      'sobol_data', 'combined_data','nTrials_AEPsych', 
                      'nTrials_forFitting', 'random_seeds', 'bestfit_seed',
                      'objhist', 'grid','model_pred_Wishart', 'Sigmas_noise_xref', 
                      'Sigmas_noise_x1', 'target_pC', 'expt_trial', 'nTrials_strat',
                      'btst_indices']
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

