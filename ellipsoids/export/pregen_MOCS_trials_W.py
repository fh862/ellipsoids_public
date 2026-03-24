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
from analysis.data_validation import shuffle_trials_within_levels
from analysis.MOCS_thresholds import sim_MOCS_trials
from analysis.utils_load import select_file_and_get_path, extract_sub_number
from analysis.ellipsoids_tools import angles_to_3Dchromatic_directions
from analysis.ellipses_tools import angles_to_2Dchromatic_directions
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.visualize_MOCS import PlotCondSettings, MOCSConditionsVisualization

#define output directory for output files and figures
stim_dims = 3
baseDir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
subfolder_name = 'Isoluminant plane' if stim_dims == 2 else '3D'

#-----------------------------------------------------
# SECTION 1: load files and define the ground truth
#-----------------------------------------------------
# Prompt user to select a fitted model file (pickled .pkl format)
# Example path:
#stim_dims == 2:
#'ELPS_analysis/Experiment_DataFiles/archived_pilot1/sub4'
#'Fitted_isothreshold_isoluminant_plane_360trialsPerRef_9refs_AEPsychSampling_bandwidth0.005_sub4.pkl'

#stim_dims == 3:
# 'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits/'
# 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.5_nBasisDeg5.pkl'

input_fileDir_fits, file_name_fits = select_file_and_get_path()
# Extract subject number (e.g., 'sub1' → 1)
SUBJ_gt_Wishart = extract_sub_number(file_name_fits)

# Load the model fit data
full_path_gt = os.path.join(input_fileDir_fits, file_name_fits)
with open(full_path_gt, 'rb') as f:
    vars_dict = pickled.load(f)
    
model_pred_Wishart = deepcopy(vars_dict['model_pred_Wishart'])
color_thres_data = vars_dict['color_thres_data']
    
# Define output directories using os.path.join for compatibility
output_fileDir = os.path.join(baseDir, 'ELPS_analysis', 'Simulation_DataFiles',
                              'MOCS', subfolder_name, 'gt_Wishart')
output_figDir = output_fileDir.replace('Simulation_DataFiles', 'Simulation_FigFiles')
# List of directories to ensure existence
directories = [output_figDir, output_fileDir]
# Ensure directories exist (create if they don't)
for directory in directories:
    os.makedirs(directory, exist_ok=True)

#%%
#-------------------------------------------------------------------------- 
# SECTION 2: Select ref locations and chromatic directions for MOCS trials
#--------------------------------------------------------------------------    
# total trials per reference condition (we probably just need 240 trials, but let's generate more in case)
num_trials = 300         
nLevels = 12              # number of stimulus levels along the chosen chromatic direction
trials_per_level = num_trials // nLevels  # repeats per level (e.g., 20)
nRefs = 20                # number of MOCS validation conditions (e.g., 25 in the eLife paper)
sobol_seed = 1000          # seed for reproducible Sobol sampling

# Define the target performance level (pC) to determine at which stimulus 
# intensity we achieve this performance.
t_pC = 0.975

# A scalar applied to determine the intensity of easier trials
scaler_s = 2

# Number of points along each chromatic direction to discretize the stimulus space
nVec_len = 100  

# Generate an evenly spaced set of steps from 0 to the full unit vector length
unit_Vec = np.linspace(0, 1, nVec_len)
    
# Some Sobol draws can place comparison stimuli outside gamut bounds after scaling.
# If that happens, increment the Sobol seed and resample until a valid set is found.
num_tries = 0
sobol_seed_success = sobol_seed
while True:
    try:
        #the current seed might lead to MOCS trials that are out of the gamut,
        #so we keep trying until we find a set that all trials are within the gamut,
        #if an attempt fails, we add 1 to the sobol_seed
        
        if stim_dims == 2:
            ref_sobol_lb = [-0.6, -0.6, 0]  # [x_ref, y_ref, theta_deg]
            ref_sobol_ub = [ 0.6,  0.6, 360]
            xref_and_angles = sim_MOCS_trials.sample_sobol(nRefs,
                                                           lb = ref_sobol_lb,
                                                           ub = ref_sobol_ub,
                                                           force_center = True,
                                                           seed = sobol_seed_success,
                                                           )
        
            xref_unique = xref_and_angles[:, :-1]   # (nRefs, 2)
            theta_deg = xref_and_angles[:, -1]      # (nRefs,)
        
            # Convert angle to a unit direction vector in the plane:
            #   d = [cos(theta), sin(theta)]
            chromatic_directions = angles_to_2Dchromatic_directions(theta_deg, normalize= True)
                
        elif stim_dims == 3:
            ref_sobol_lb = [-0.65, -0.65, -0.65, 0, 0]    # [x_ref, y_ref, z_ref, theta_deg, phi_deg]
            ref_sobol_ub = [ 0.65,  0.65,  0.65, 360, 180]
            xref_and_angles = sim_MOCS_trials.sample_sobol(nRefs,
                                                           lb = ref_sobol_lb,
                                                           ub = ref_sobol_ub,
                                                           force_center = True,
                                                           seed = sobol_seed_success,
                                                           )
        
            xref_unique = xref_and_angles[:, :-2]   # (nRefs, 3)
            theta_deg = xref_and_angles[:, -2]      # azimuth in degrees, (nRefs,)
            phi_deg = xref_and_angles[:, -1]        # polar angle from +z in degrees, (nRefs,)
        
            # Convert spherical angles to a 3D unit vector.
            chromatic_directions = angles_to_3Dchromatic_directions(theta_deg,
                                                                    phi_deg,
                                                                    normalize= True
                                                                    )
        
        else:
            raise ValueError(f"stim_dims must be 2 or 3, got {stim_dims!r}")        
                
        # Initialize data storage arrays
        vecLen_t_pC        = np.full((nRefs, 1), np.nan)  # Stores vector lengths at which performance reaches t_pC
        vecLen_easy        = np.full((nRefs, 1), np.nan)  # Stores vector lengths for easier trials
        sim_pX1            = np.full((nRefs, nLevels), np.nan)  # Stores simulated percent correct (pX1)
        pX1_Wishart        = np.full((nRefs, nVec_len), np.nan)  # Stores Wishart-predicted pX1 for finer stimulus sampling
        pX1_Wishart_stim   = np.full((nRefs, nLevels), np.nan)  # Stores Wishart-predicted pX1 for MOCS stimulus levels
        comp_unique_origin = np.full((nRefs, nLevels, stim_dims), np.nan)  # Stores unique comparison stimuli
        comp_unique        = np.full((nRefs, nLevels, stim_dims), np.nan)  # Stores unique comparison stimuli
        refStimulus        = np.full((nRefs, num_trials, stim_dims), np.nan)  # Stores reference stimuli per trial
        compStimulus       = np.full(refStimulus.shape, np.nan)  # Stores comparison stimuli per trial
        responses          = np.full((nRefs, num_trials), np.nan)  # Stores simulated binary responses
        
        #----------------------------------------------------
        # Loop through each MOCS reference condition
        #----------------------------------------------------
        for i in range(nRefs):
            # Retrieve the chromatic direction unit vector for this reference condition
            cdir_idx = chromatic_directions[i]
        
            # Generate a finely sampled set of stimulus points along the chromatic direction
            finer_stim = sim_MOCS_trials.create_discrete_stim(cdir_idx, nVec_len,
                                                              startpoint=np.full((stim_dims,),0),
                                                              ndims = stim_dims
                                                              )
        
            # Compute the probability of choosing x1 as the odd stimulus for each stimulus pair    
            pX1_Wishart[i] = model_pred_Wishart._compute_pChoosingX1(
                np.full(finer_stim.shape, 0) + xref_unique[i],  # Reference stimuli
                finer_stim + xref_unique[i],  # Comparison stimuli
            )
        
            # Find the first index where the probability reaches the target performance t_pC
            idx_vecLen_t_pC = np.argmin(np.abs(pX1_Wishart[i] - t_pC))
        
            # Store the corresponding vector length
            vecLen_t_pC[i] = unit_Vec[idx_vecLen_t_pC]
        
            # Compute the vector length for easy trials by scaling the target length
            vecLen_easy[i] = vecLen_t_pC[i] * scaler_s
        
            # Compute the actual chromatic direction vector at the performance threshold
            vecVal_vecLen_t_pC = cdir_idx * vecLen_t_pC[i]
        
            # Generate evenly spaced comparison stimuli along this vector
            comp_unique_temp = sim_MOCS_trials.create_discrete_stim(vecVal_vecLen_t_pC, nLevels,
                                                                    startpoint=np.full((stim_dims,),0),
                                                                    ndims = stim_dims
                                                                    )
            comp_unique_truncated = comp_unique_temp[-nLevels:]
        
            # Add the easier trial level to the comparison set
            easyTrial_stim = cdir_idx * vecLen_easy[i]    
            comp_unique_origin[i] = np.vstack((comp_unique_truncated[1:], easyTrial_stim))
            comp_unique[i] = comp_unique_origin[i] + xref_unique[i]
            
            # Check if values exceed bounds [-1, 1]
            if np.min(comp_unique) < -1 or np.max(comp_unique) > 1:
                raise ValueError("easyTrial_stim contains out-of-bounds values: "+\
                                 f"min {np.min(comp_unique)}, max {np.max(comp_unique)}")
            
            # Compute the Wishart-predicted performance at selected MOCS stimulus levels
            pX1_Wishart_stim[i] = model_pred_Wishart._compute_pChoosingX1(
                np.full((nLevels, stim_dims), 0) + xref_unique[i],  # Reference stimuli
                comp_unique[i],  # Comparison stimuli
            )
            
            for j in range(nLevels):
                # Simulate binary responses based on the predicted probability of choosing x1
                sim_resp_n, pC_mean_n = sim_MOCS_trials.sim_binary_trials(
                    pX1_Wishart_stim[i][j], trials_per_level
                )

                # Store the simulated responses
                idx_lb = j * trials_per_level  # Lower index for this level
                idx_ub = (j + 1) * trials_per_level  # Upper index for this level
                responses[i, idx_lb: idx_ub] = sim_resp_n

                # Repeat reference stimuli for the given number of trials
                refStimulus[i, idx_lb: idx_ub] = np.tile(
                    xref_unique[i], (trials_per_level, 1)
                )

                # Compute and store the comparison stimuli, ensuring correct shifts
                compStimulus[i, idx_lb: idx_ub] = np.tile(
                    comp_unique[i][j], (trials_per_level, 1)
                )
        # If we got here, all conditions passed the gamut checks
        break
    except Exception as e:
        # Resample with a new seed if any condition fails gamut checks or other constraints
        num_tries += 1
        sobol_seed_success += 1
        print(f'Attempt {num_tries} failed: {e}')
    
#--------------------------------------------------------------------------
# Debugging: Plot the Wishart-predicted pX1 and the simulated responses
#--------------------------------------------------------------------------
# Debug flag: Set to True if you want to visualize the selected MOCS trials
flag_debugplots = True
if flag_debugplots:
    for i in range(nRefs):
        plt.plot(unit_Vec, pX1_Wishart[i])  # Plot continuous probability function
        plt.scatter(vecLen_easy[i], 1, edgecolor='r', marker='o', facecolor = 'w', lw = 2)  
        plt.scatter(np.linspace(0, vecLen_t_pC[i], nLevels)[1:], 
                    pX1_Wishart_stim[i][:-1], color = 'r', marker = 'o')
        plt.xlim([0,0.6])
        plt.show()

#%%
#-------------------------------------------------------------------------- 
# SECTION 4: Visualize and save the file
#--------------------------------------------------------------------------
# Create a base plotting settings instance (shared across plots)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize=10)
plotCond_Settings = replace(PlotCondSettings(), **pltSettings_base.__dict__)
MOCS_cond_vis = MOCSConditionsVisualization(settings = pltSettings_base, save_fig = True)
if stim_dims == 2:
    ttl = color_thres_data.plane_2D
    fig_name = f'Sim{stim_dims}dTask_colorDiscrimination_{ttl}_'+\
               f'MOCStrials_{nRefs}refs_{nLevels}levels_{trials_per_level}trialsPerLevel_'+\
               f'gtSub{SUBJ_gt_Wishart}_seed{sobol_seed}.pdf'
    plotCond_Settings = replace(plotCond_Settings, 
                                fig_size = (5,5), 
                                ticks = np.linspace(-0.7, 0.7, 5),
                                title = color_thres_data.plane_2D, 
                                fig_name = fig_name
                                )
    MOCS_cond_vis.plot_MOCS_conditions(stim_dims, xref_unique, 
                                       comp_unique, color_thres_data,
                                       settings = plotCond_Settings
                                       )
else:
    ttl = 'RGB cube'
    fig_name = f'Sim{stim_dims}dTask_colorDiscrimination_{ttl}_MOCStrials_'+\
              f'{nRefs}refs_{nLevels}levels_{trials_per_level}trialsPerLevel_'+\
              f'gtSub{SUBJ_gt_Wishart}_seed{sobol_seed}.pdf'
    plotCond_Settings = replace(plotCond_Settings, 
                                fig_size = (5.5, 6), 
                                ticks = np.linspace(-0.7, 0.7, 5),
                                ref_ms = 50,
                                fig_name = fig_name
                                )   
    MOCS_cond_vis.plot_MOCS_conditions(stim_dims, xref_unique, 
                                       comp_unique, color_thres_data,
                                       settings = plotCond_Settings
                                       )
    
#%%
#-------------------------------------------------------------------------- 
# SECTION 5: Shuffle MOCS trials
#--------------------------------------------------------------------------
# Reshape the reference and comparison stimulus arrays into 2D arrays
# Each row corresponds to a trial, and the second dimension is preserved (-1 means inferred size)
shape_flat = (nRefs * num_trials, stim_dims)
MOCS_xref_unshuffled = np.reshape(refStimulus, shape_flat)
MOCS_x1_unshuffled = np.reshape(compStimulus, shape_flat)

# Create a shuffled array for levels within each reference condition
rng = np.random.default_rng(sobol_seed_success)  # Create a random number generator with a seed
shuffled_list = []
for n in range(nRefs):  
    #  Create a 2D array of levels (nLevels x trials_per_level)
    shuffled_array_n = np.tile(np.arange(nLevels)[:,None], (1, trials_per_level))
    
    # Shuffle each column independently using rng.permutation()
    for col in range(shuffled_array_n.shape[1]):
        shuffled_array_n[:, col] = rng.permutation(shuffled_array_n[:, col])  # Shuffle while ensuring reproducibility
    
    # Flatten the shuffled array and store it in the list
    shuffled_list.append(shuffled_array_n.ravel())

# Concatenate all shuffled arrays from different references into a single 1D array
shuffled_array = np.concatenate(shuffled_list)

# Create a DataFrame containing trial information
pregenerated_trials = pd.DataFrame({
    'condition': np.repeat(np.arange(nRefs), num_trials).tolist(),                      # Reference condition index
    'level': np.tile(np.repeat(np.arange(nLevels), trials_per_level), nRefs).tolist(),  # Original level index
    'trial': list(range(trials_per_level)) * (nRefs * nLevels),                         # Trial index within each level
    'shuffled_level': shuffled_array                                                    # The shuffled levels for each trial
})

# Determine a subject-specific seed and shuffle the trials while preserving trial-level structure
MOCS_xref_shuffled, MOCS_x1_shuffled, MOCS_shuffled_idx = shuffle_trials_within_levels(
    pregenerated_trials,         # DataFrame containing trial information
    MOCS_xref_unshuffled,        # Unshuffled reference stimuli
    MOCS_x1_unshuffled,          # Unshuffled comparison stimuli
    seed=sobol_seed_success      # Random seed for reproducibility
)

if flag_debugplots:
    if stim_dims == 2:
        slc_idx = [0,1]
    else:
        #for 3D, we can only look at 2 coordinates
        slc_idx = [0,2] #[0, 1], [0, 2], [1,2]
    #visualize shuffled MOCS trials over time
    for t in range(trials_per_level):
        t_ub = nRefs*nLevels * (t+1)
        xref = MOCS_xref_shuffled[:t_ub, slc_idx].T
        x1 = MOCS_x1_shuffled[:t_ub, slc_idx].T
        plt.scatter(*xref, alpha = 0.03)
        plt.scatter(*x1, alpha = 0.03)
        plt.xlim([-1,1]); plt.ylim([-1,1])
        plt.pause(1)
    plt.show()
    
    for idx_starting in range(trials_per_level):
        tt_lb = idx_starting*nRefs
        tt_ub = nRefs*(idx_starting+1)+1
        xref = MOCS_xref_shuffled[tt_lb:tt_ub, slc_idx].T
        x1 = MOCS_x1_shuffled[tt_lb:tt_ub, slc_idx].T
        plt.scatter(*xref, alpha = 0.5)
        plt.scatter(*x1, alpha = 0.5)
        plt.plot([xref[0], x1[0]], [xref[1], x1[1]], c = 'k')
        plt.xlim([-1,1]); plt.ylim([-1,1]); plt.gca().set_aspect('equal', adjustable='box')
        plt.show()   
        
#%%
#-------------------------------------------------------------------------- 
# SECTION 7: Save the data
#--------------------------------------------------------------------------
output_file = fig_name[:-4] + '.pkl'
full_path_output = os.path.join(output_fileDir, output_file)

variable_names = ['stim_dims','SUBJ_gt_Wishart','full_path_gt','color_thres_data',
                  'model_pred_Wishart', 'num_trials', 'nLevels', 'trials_per_level',
                  'nRefs', 'sobol_seed','t_pC', 'scaler_s', 'nVec_len', 'unit_Vec',
                  'num_tries', 'sobol_seed_success', 'ref_sobol_lb', 'ref_sobol_ub',
                  'xref_unique', 'theta_deg','phi_deg', 'chromatic_directions',
                  'vecLen_t_pC', 'vecLen_easy', 'sim_pX1', 'pX1_Wishart', 
                  'pX1_Wishart_stim','comp_unique_origin', 'comp_unique',
                  'refStimulus', 'compStimulus', 'responses', 'MOCS_xref_unshuffled',
                  'MOCS_x1_unshuffled', 'pregenerated_trials', 'MOCS_xref_shuffled',
                  'MOCS_x1_shuffled', 'MOCS_shuffled_idx']
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
with open(full_path_output, 'wb') as f:
    pickled.dump(vars_dict, f)