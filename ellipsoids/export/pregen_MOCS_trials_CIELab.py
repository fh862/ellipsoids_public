#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:12:55 2025

@author: fangfang
"""

import matplotlib.pyplot as plt
import dill as pickled
import pandas as pd
from copy import deepcopy
import sys
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from dataclasses import replace

#load functions from other modules
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.color_thres import color_thresholds
from analysis.data_validation import shuffle_trials_within_levels
from analysis.MOCS_thresholds import sim_MOCS_trials
from analysis.utils_load import select_file_and_get_path, extract_sub_number
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.visualize_MOCS import PlotCondSettings, MOCSConditionsVisualization

#define output directory for output files and figures
stim_dims =3
baseDir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
subfolder_name = 'Isoluminant plane' if stim_dims == 2 else '3D'

#-----------------------------------------------------
# SECTION 1: load files and define the ground truth
#-----------------------------------------------------
# Define the required parameters
if stim_dims == 2:
    flag_gt_CIE = False
    plane_2D = 'Isoluminant plane'  
    
    if flag_gt_CIE:
        # Create an instance of the class
        color_thres_data = color_thresholds(stim_dims, 
                                            baseDir, 
                                            plane_2D = plane_2D)
        SUBJ_gt_Wishart = 'CIE1994'
        color_thres_data.load_CIE_data(CIE_version = SUBJ_gt_Wishart)
        color_thres_data.load_model_fits(CIE_version = SUBJ_gt_Wishart)  
    else:
        SUBJ_gt_Wishart = 4
        # Create an instance of the class
        color_thres_data = color_thresholds(stim_dims, 
                                            baseDir, 
                                            plane_2D = plane_2D,
                                            manual_input= True)
        # Load Wishart model fits
        color_thres_data.load_model_fits()  
        color_thres_data.load_CIE_data(CIE_version = 'CIE1994')

    if plane_2D == 'Isoluminant plane':
        color_thres_data.load_transformation_matrix()

elif stim_dims == 3:
    # Prompt user to select a fitted model file (pickled .pkl format)
    # Example path:
    # '/Volumes/T9/.../ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits/'
    # Example filename:
    # 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.5_nBasisDeg5.pkl'

    input_fileDir_fits, file_name_fits = select_file_and_get_path()
    # Extract subject number (e.g., 'sub1' → 1)
    SUBJ_gt_Wishart = extract_sub_number(file_name_fits)
    
    # Load the model fit data
    full_path = os.path.join(input_fileDir_fits, file_name_fits)
    with open(full_path, 'rb') as f:
        vars_dict = pickled.load(f)
        
    model_pred_Wishart = deepcopy(vars_dict['model_pred_Wishart'])
    color_thres_data = vars_dict['color_thres_data']
    
# Define output directories using os.path.join for compatibility
output_fileDir = os.path.join(baseDir, 'ELPS_analysis', 'Simulation_DataFiles',
                              subfolder_name, 'MOCS', 'gt_CIE' if flag_gt_CIE else '')
output_figDir_sims = os.path.join(baseDir, 'ELPS_analysis', 'Simulation_FigFiles', 
                                  'Python_version', subfolder_name, 'MOCS', 
                                  'gt_CIE' if flag_gt_CIE else '')
# List of directories to ensure existence
directories = [output_figDir_sims, output_fileDir]
# Ensure directories exist (create if they don't)
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Retrieve specific data from Wishart_data
# GROUND TRUTH
try:
    model_pred_Wishart  = color_thres_data.get_data('model_pred_Wishart', 
                                                    dataset = 'Wishart_data'
                                                    )
    grid_stim = color_thres_data.get_data('grid_stim', dataset = 'Wishart_data')
except KeyError as e:
    print(f"Data not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

#%%
#-------------------------------------------------------------------------- 
# SECTION 2: Select ref locations and chromatic directions for MOCS trials
#--------------------------------------------------------------------------    
num_trials = 240          # total trials per reference condition
nLevels = 12              # number of stimulus levels along the chosen chromatic direction
trials_per_level = num_trials // nLevels  # repeats per level (e.g., 20)

nRefs = 25                # number of MOCS validation conditions (e.g., 25 in the eLife paper)
sobol_seed = 100          # seed for reproducible Sobol sampling


# ------------------------------------------------------------
# Sample reference locations + chromatic directions (Sobol)
#
# We sample:
#   - reference location(s) in model space (2D or 3D)
#   - direction parameters:
#       * 2D: one angle (deg) in [0, 360)
#       * 3D: two angles (deg): theta in [0, 360), phi in [0, 180]
#
# The helper `sample_sobol` is assumed to return an array of shape:
#   - 2D: (nRefs, 3)  -> [x_ref, y_ref, theta_deg]
#   - 3D: (nRefs, 5)  -> [x_ref, y_ref, z_ref, theta_deg, phi_deg]
# ------------------------------------------------------------
if stim_dims == 2:
    xref_and_angles = sim_MOCS_trials.sample_sobol(
        nRefs,
        lb=[-0.6, -0.6, 0],      # [x_ref, y_ref, theta_deg]
        ub=[ 0.6,  0.6, 360],
        force_center=True,
        seed=sobol_seed,
    )

    xref_unique = xref_and_angles[:, :-1]   # (nRefs, 2)
    theta_deg = xref_and_angles[:, -1]      # (nRefs,)

    # Convert angle to a unit direction vector in the plane:
    #   d = [cos(theta), sin(theta)]
    theta_rad = np.radians(theta_deg)
    chromatic_directions = np.column_stack([np.cos(theta_rad), np.sin(theta_rad)])  # (nRefs, 2)


elif stim_dims == 3:
    xref_and_angles = sim_MOCS_trials.sample_sobol(
        nRefs,
        lb=[-0.6, -0.6, -0.6, 0, 0],     # [x_ref, y_ref, z_ref, theta_deg, phi_deg]
        ub=[ 0.6,  0.6,  0.6, 360, 180],
        force_center=True,
        seed=sobol_seed,
    )

    xref_unique = xref_and_angles[:, :-2]   # (nRefs, 3)
    theta_deg = xref_and_angles[:, -2]      # azimuth in degrees, (nRefs,)
    phi_deg = xref_and_angles[:, -1]        # polar angle from +z in degrees, (nRefs,)

    # Convert spherical angles to a 3D unit vector.
    # Convention:
    #   theta: azimuth in x-y plane, 0..360
    #   phi:   polar angle from +z axis, 0..180
    #
    # Unit vector:
    #   x = sin(phi) * cos(theta)
    #   y = sin(phi) * sin(theta)
    #   z = cos(phi)
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)

    chromatic_directions = np.column_stack([
        np.sin(phi_rad) * np.cos(theta_rad),
        np.sin(phi_rad) * np.sin(theta_rad),
        np.cos(phi_rad),
    ])  # (nRefs, 3)

else:
    raise ValueError(f"stim_dims must be 2 or 3, got {stim_dims!r}")


# ------------------------------------------------------------
# (Optional) sanity checks
# ------------------------------------------------------------
# Ensure unit-length directions (within numerical tolerance)
dir_norm = np.linalg.norm(chromatic_directions, axis=1)
if not np.allclose(dir_norm, 1.0, atol=1e-7):
    raise RuntimeError("chromatic_directions are not unit vectors (unexpected).")

# xref_unique: (nRefs, stim_dims)
# chromatic_directions: (nRefs, stim_dims)

#%%
#--------------------------------------------------------------------------
# SECTION 3: Simulate Binary Responses
#--------------------------------------------------------------------------
# Define the target performance level (pC) to determine at which stimulus 
# intensity we achieve this performance.
t_pC = 0.95

# A scalar applied to determine the intensity of easier trials
scaler_s = 2.5 

# truncated levels
nLevels_wExtra = nLevels

# Number of points along each chromatic direction to discretize the stimulus space
nVec_len = 100  

# Generate an evenly spaced set of steps from 0 to the full unit vector length
unit_Vec = np.linspace(0, 1, nVec_len)

# Debug flag: Set to True if you want to visualize the selected MOCS trials
flag_debugplots = True

"""
Initialize data storage arrays
"""
vecLen_t_pC      = np.full((nRefs, 1), np.nan)  # Stores vector lengths at which performance reaches t_pC
vecLen_easy      = np.full((nRefs, 1), np.nan)  # Stores vector lengths for easier trials
sim_pX1          = np.full((nRefs, nLevels), np.nan)  # Stores simulated percent correct (pX1)
pX1_Wishart      = np.full((nRefs, nVec_len), np.nan)  # Stores Wishart-predicted pX1 for finer stimulus sampling
pX1_Wishart_stim = np.full((nRefs, nLevels), np.nan)  # Stores Wishart-predicted pX1 for MOCS stimulus levels
comp_unique_origin = np.full((nRefs, nLevels, stim_dims), np.nan)  # Stores unique comparison stimuli
comp_unique      = np.full((nRefs, nLevels, stim_dims), np.nan)  # Stores unique comparison stimuli
refStimulus      = np.full((nRefs, num_trials, stim_dims), np.nan)  # Stores reference stimuli per trial
compStimulus     = np.full(refStimulus.shape, np.nan)  # Stores comparison stimuli per trial
responses        = np.full((nRefs, num_trials), np.nan)  # Stores simulated binary responses

"""
Loop through each MOCS reference condition
"""
for idx_slc in range(nRefs):
    # Retrieve the chromatic direction unit vector for this reference condition
    cdir_idx = chromatic_directions[idx_slc]

    # Generate a finely sampled set of stimulus points along the chromatic direction
    finer_stim = sim_MOCS_trials.create_discrete_stim(cdir_idx, nVec_len,
                                                      startpoint=np.full((stim_dims,),0),
                                                      ndims = stim_dims
                                                      )

    # Compute the probability of choosing x1 as the odd stimulus for each stimulus pair    
    pX1_Wishart[idx_slc] = model_pred_Wishart._compute_pChoosingX1(
        np.full(finer_stim.shape, 0) + xref_unique[idx_slc],  # Reference stimuli
        finer_stim + xref_unique[idx_slc],  # Comparison stimuli
    )

    # Find the first index where the probability reaches the target performance t_pC
    idx_vecLen_t_pC = np.argmin(np.abs(pX1_Wishart[idx_slc] - t_pC))

    # Store the corresponding vector length
    vecLen_t_pC[idx_slc] = unit_Vec[idx_vecLen_t_pC]

    # Compute the vector length for easy trials by scaling the target length
    vecLen_easy[idx_slc] = vecLen_t_pC[idx_slc] * scaler_s

    """
    Define MOCS comparison stimulus levels
    """
    # Compute the actual chromatic direction vector at the performance threshold
    vecVal_vecLen_t_pC = cdir_idx * vecLen_t_pC[idx_slc]

    # Generate evenly spaced comparison stimuli along this vector
    comp_unique_temp = sim_MOCS_trials.create_discrete_stim(vecVal_vecLen_t_pC, nLevels_wExtra,
                                                            startpoint=np.full((stim_dims,),0),
                                                            ndims = stim_dims
                                                            )
    comp_unique_truncated = comp_unique_temp[-nLevels:]

    # Add the easier trial level to the comparison set
    easyTrial_stim = cdir_idx * vecLen_easy[idx_slc]    
    comp_unique_origin[idx_slc] = np.vstack((comp_unique_truncated[1:], easyTrial_stim))
    comp_unique[idx_slc] = comp_unique_origin[idx_slc] + xref_unique[idx_slc]
    
    # Check if values exceed bounds [-1, 1]
    if np.min(comp_unique) < -1 or np.max(comp_unique) > 1:
        raise ValueError("easyTrial_stim contains out-of-bounds values: "+\
                         f"min {np.min(comp_unique)}, max {np.max(comp_unique)}")
    
    # Compute the Wishart-predicted performance at selected MOCS stimulus levels
    pX1_Wishart_stim[idx_slc] = model_pred_Wishart._compute_pChoosingX1(
        np.full((nLevels, stim_dims), 0) + xref_unique[idx_slc],  # Reference stimuli
        comp_unique[idx_slc],  # Comparison stimuli
    )

    """
    Simulate binary responses for each MOCS level
    """    
    for n in range(nLevels):
        # Simulate binary responses based on the predicted probability of choosing x1
        sim_resp_n, pC_mean_n = sim_MOCS_trials.sim_binary_trials(
            pX1_Wishart_stim[idx_slc][n], trials_per_level
        )

        # Store the simulated responses
        idx_lb = n * trials_per_level  # Lower index for this level
        idx_ub = (n + 1) * trials_per_level  # Upper index for this level
        responses[idx_slc, idx_lb: idx_ub] = sim_resp_n

        # Store the mean simulated percent correct
        sim_pX1[idx_slc, n] = pC_mean_n

        # Repeat reference stimuli for the given number of trials
        refStimulus[idx_slc, idx_lb: idx_ub] = np.tile(
            xref_unique[idx_slc], (trials_per_level, 1)
        )

        # Compute and store the comparison stimuli, ensuring correct shifts
        compStimulus[idx_slc, idx_lb: idx_ub] = np.tile(
            comp_unique[idx_slc][n], (trials_per_level, 1)
        )

    """
    Debugging: Plot the Wishart-predicted pX1 and the simulated responses
    """    
    if flag_debugplots:
        plt.plot(unit_Vec, pX1_Wishart[idx_slc])  # Plot continuous probability function
        plt.scatter(
            np.linalg.norm(comp_unique_origin[idx_slc], axis=1),
            sim_pX1[idx_slc], color='r', marker='o'
        )  # Scatter plot of selected MOCS levels
        plt.xlim([0,0.6])
        plt.show()

#%%
#-------------------------------------------------------------------------- 
# SECTION 4: Visualize and save the file
#--------------------------------------------------------------------------
# Create a base plotting settings instance (shared across plots)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_sims, fontsize=10)
plotCond_Settings = replace(PlotCondSettings(), **pltSettings_base.__dict__)
MOCS_cond_vis = MOCSConditionsVisualization(settings = pltSettings_base, save_fig = False)
if stim_dims == 2:
    ttl = "Isoluminant plane"
    str_extension = f"sub{SUBJ_gt_Wishart}"
    fig_name = f'Sim{stim_dims}dTask_colorDiscrimination_{ttl}_MOCStrials_'+\
              f'{nRefs}refs_{nLevels}levels_{trials_per_level}trialsPerLevel_'+\
              f'{str_extension}_seed{sobol_seed}.pdf'
    plotCond_Settings = replace(plotCond_Settings, 
                                fig_size = (5,5), 
                                ticks = np.linspace(-0.8, 0.8, 5),
                                title = ttl, 
                                fig_name = fig_name)
    MOCS_cond_vis.plot_MOCS_conditions(stim_dims, xref_unique, 
                                       comp_unique, color_thres_data,
                                       settings = plotCond_Settings)
else:
    ttl = 'RGB cube'
    str_extension = f"sub{SUBJ_gt_Wishart}"
    fig_name = f'Sim{stim_dims}dTask_colorDiscrimination_{ttl}_MOCStrials_'+\
              f'{nRefs}refs_{nLevels}levels_{trials_per_level}trialsPerLevel_'+\
              f'{str_extension}_seed{sobol_seed}.pdf'
    plotCond_Settings = replace(plotCond_Settings, 
                                fig_size = (4.5, 5), 
                                ticks = np.linspace(-0.8, 0.8, 3),
                                title = ttl, 
                                fig_name = fig_name)   
    MOCS_cond_vis.plot_MOCS_conditions(stim_dims, xref_unique, 
                                       comp_unique, color_thres_data,
                                       settings = plotCond_Settings)
    
#%%
#-------------------------------------------------------------------------- 
# SECTION 5: Shuffle MOCS trials
#--------------------------------------------------------------------------
# Reshape the reference and comparison stimulus arrays into 2D arrays
# Each row corresponds to a trial, and the second dimension is preserved (-1 means inferred size)
MOCS_xref_unshuffled = np.reshape(refStimulus, (nRefs * nLevels * trials_per_level, -1))
MOCS_x1_unshuffled = np.reshape(compStimulus, (nRefs * nLevels * trials_per_level, -1))

# Create a shuffled array for levels within each reference condition
rng = np.random.default_rng(sobol_seed)  # Create a random number generator with a seed
shuffled_list = []
for n in range(nRefs):  
    #  Create a 2D array of levels (nLevels x trials_per_level)
    shuffled_array_n = np.tile(np.arange(nLevels).reshape(nLevels, 1), (1, trials_per_level))
    # Shuffle each column independently using rng.permutation()
    for col in range(shuffled_array_n.shape[1]):
        shuffled_array_n[:, col] = rng.permutation(shuffled_array_n[:, col])  # Shuffle while ensuring reproducibility
    # Flatten the shuffled array and store it in the list
    shuffled_list.append(shuffled_array_n.ravel())
# Concatenate all shuffled arrays from different references into a single 1D array
shuffled_array = np.concatenate(shuffled_list)

# Create a DataFrame containing trial information
pregenerated_trials = pd.DataFrame({
    'condition': np.repeat(np.arange(nRefs), nLevels * trials_per_level).tolist(),  # Reference condition index
    'level': np.tile(np.repeat(np.arange(nLevels), trials_per_level), nRefs).tolist(),  # Original level index
    'trial': list(range(trials_per_level)) * (nRefs * nLevels),  # Trial index within each level
    'shuffled_level': shuffled_array  # The shuffled levels for each trial
})

# Determine a subject-specific seed and shuffle the trials while preserving trial-level structure
MOCS_xref_shuffled, MOCS_x1_shuffled, MOCS_shuffled_idx = shuffle_trials_within_levels(
    pregenerated_trials,  # DataFrame containing trial information
    MOCS_xref_unshuffled,  # Unshuffled reference stimuli
    MOCS_x1_unshuffled,  # Unshuffled comparison stimuli
    seed=sobol_seed  # Random seed for reproducibility
)

if flag_debugplots:
    #visualize shuffled MOCS trials over time
    for t in range(trials_per_level):
        t_ub = nRefs*nLevels * (t+1)
        plt.scatter(MOCS_xref_shuffled[:t_ub, 0], MOCS_xref_shuffled[:t_ub,1], alpha = 0.03)
        plt.scatter(MOCS_x1_shuffled[:t_ub, 0], MOCS_x1_shuffled[:t_ub,1], alpha = 0.03)
        plt.xlim([-1,1]); plt.ylim([-1,1])
        plt.pause(1)
    plt.show()
    
    for idx_starting in range(trials_per_level):
        tt_lb = idx_starting*nRefs
        tt_ub = nRefs*(idx_starting+1)+1
        plt.scatter(MOCS_xref_shuffled[tt_lb:tt_ub, 0], MOCS_xref_shuffled[tt_lb:tt_ub,1], alpha = 0.5)
        plt.scatter(MOCS_x1_shuffled[tt_lb:tt_ub, 0], MOCS_x1_shuffled[tt_lb:tt_ub,1], alpha = 0.5)
        plt.plot([MOCS_xref_shuffled[tt_lb:tt_ub,0], MOCS_x1_shuffled[tt_lb:tt_ub, 0]],
                 [MOCS_xref_shuffled[tt_lb:tt_ub,1], MOCS_x1_shuffled[tt_lb:tt_ub, 1]],c = 'k')
        plt.xlim([-1,1]); plt.ylim([-1,1]); plt.gca().set_aspect('equal', adjustable='box')
        plt.show()   
        
#%% 
#--------------------------------------------------------------------------
# SECTION 6: Pre-generate Sobol Trials
#--------------------------------------------------------------------------
# In addition to MOCS trials, we might also slot in pre-generated Sobol trials 
# as a backup. These trials will only be used if MOCS trials reach their maximum 
# allowable shifts in presentation order. 
#
# Since we don’t know exactly how many Sobol trials will be needed, we simulate 
# more than necessary, but most will not be used in the actual experiment.

# Number of Sobol trials per session
nTrials_sobol_perSession = 1200  

# Lower and upper bounds for the 4D Sobol samples (representing different dimensions)
lb_sobol_trials = [-0.75, -0.75, -0.25, -0.25]  
ub_sobol_trials = [ 0.75,  0.75,  0.25,  0.25]  

# Scaling factors applied to the comparison stimulus to balance trial difficulty
sobol_scaler = [2/8, 3/8, 4/8]  

# Number of times to repeat the scaling factor set to match the number of trials
num_repeats = nTrials_sobol_perSession // len(sobol_scaler)

# Maximum number of experimental sessions (we generate more than needed)
nSessions = 15  

# Preallocate arrays to store generated Sobol reference (`xref`) and comparison (`x1`) stimuli
Sobol_xref = np.full((nSessions, nTrials_sobol_perSession, 2), np.nan)
Sobol_x1   = np.full(Sobol_xref.shape, np.nan)

for n in range(nSessions):
    # Define a session-specific seed for reproducibility
    seed_session_n = sobol_seed + n  
    
    # Generate Sobol-distributed samples in 4D space
    Sobol_samples = sim_MOCS_trials.sample_sobol(nTrials_sobol_perSession, 
                                                 lb=lb_sobol_trials,
                                                 ub=ub_sobol_trials,
                                                 force_center=False,
                                                 seed=seed_session_n)

    # Shuffle the scaling factors with a session-specific random seed for reproducibility
    np.random.seed(seed_session_n)  
    sobol_scaler_n = np.concatenate([np.random.permutation(sobol_scaler) for _ in range(num_repeats)])

    # Assign reference (`xref`) and comparison (`x1`) values
    Sobol_xref[n] = Sobol_samples[:, :2]  # First two dimensions for `xref`
    Sobol_x1[n] = Sobol_xref[n] + sobol_scaler_n[:, np.newaxis] * Sobol_samples[:, 2:]  # Apply scaling to last two dimensions
    
    if flag_debugplots:
        plt.scatter(Sobol_xref[n,:,0], Sobol_xref[n,:,1],marker='+')
        plt.scatter(Sobol_x1[n,:,0], Sobol_x1[n,:,1], marker = 'o', s = 1)
        for m in range(nTrials_sobol_perSession):
            plt.plot([Sobol_xref[n,m,0], Sobol_x1[n,m,0]], 
                     [Sobol_xref[n,m,1], Sobol_x1[n,m,1]])
        plt.show()

#-------------------------------------------------------------------------- 
# SECTION 7: Save the data
#--------------------------------------------------------------------------
if nLevels_wExtra != nLevels:
    str_ext = '_wConcentratedLevels'
else:
    str_ext = ''
output_file = f'Sim{stim_dims}dTask_colorDiscrimination_{ttl}_MOCStrials_'+\
                f'{nRefs}refs_{nLevels}levels_{trials_per_level}trialsPerLevel_'+\
                f'{str_extension}_{method_ref_generation}{str_ext}_seed{sobol_seed}.pkl'
full_path2 = os.path.join(output_fileDir, output_file)

variable_names = ['SUBJ_gt_Wishart','color_thres_data','model_pred_Wishart',
                  'model_pred_Wishart_indv_ell', 'grid_stim',
                  'num_trials', 'nLevels', 'trials_per_level',
                  'nRefs', 'large_grid_bound', 'small_grid_bound','num_pts_large', 
                  'num_pts_small','sobol_seed','chromatic_angles', 'chromatic_directions',
                  't_pC','xref_unique', 'scaler_s', 'nVec_len', 'unit_Vec',
                  'vecLen_t_pC', 'vecLen_easy', 'sim_pX1', 'pX1_Wishart', 
                  'pX1_Wishart_stim','comp_unique_origin',
                  'comp_unique', 'refStimulus', 'compStimulus', 'responses',
                  'MOCS_xref_unshuffled', 'MOCS_x1_unshuffled', 'pregenerated_trials',
                  'MOCS_xref_shuffled', 'MOCS_x1_shuffled', 'MOCS_shuffled_idx',
                  'nTrials_sobol_perSession', 'lb_sobol_trials','ub_sobol_trials',
                  'sobol_scaler', 'Sobol_xref', 'Sobol_x1']
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
with open(full_path2, 'wb') as f:
    pickled.dump(vars_dict, f)