#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:29:31 2025

@author: fangfang

This script compares the slope and correlation coefficient values 
between simulated and empirical data for a color discrimination task.

For each subject, thresholds predicted by the Wishart model were compared 
against thresholds from Weibull psychometric fits. Correlation coefficients 
and slopes were computed from 120 bootstrapped MOCS datasets.

For context, simulated AEPsych and MOCS trials were also analyzed using CIE1994
ground truth, following the same pipeline. The slopes and correlation coefficients 
for human subjects are qualitatively similar to those from the simulation.

"""

import dill as pickled
import numpy as np
import re
import matplotlib.pyplot as plt
import os
from analysis.utils_load import get_path
from analysis.utils_load import select_file_and_get_path

#specify the file name
base_dir = get_path("dropbox_root_mac_elps")

#%%
#---------------------------------------------------------------------------
# SECTION 1: load simulated data
# --------------------------------------------------------------------------
# Navigate to the directory: 'ELPS_analysis/Simulation_DataFiles/MOCS/gt_CIE/'
#'Fitted_weibull_psychometric_func_Isoluminant plane_240totalTrials_25refs_MOCS_subCIE1994_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl'
input_fileDir_fits, file_name = select_file_and_get_path()

# Construct the full path to the selected file
full_path = os.path.join(input_fileDir_fits, file_name)

# Load the necessary variables from the file
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)

# extract slope and correlation coefficient
key_sub = ['slope_modelPred_org', 
           'corr_coef_modelPred_org', 
           'slope_btst_CI',
           'corr_coef_btst_CI']

slope_sim, corr_coef_sim, slope_CI_sim, corr_coef_CI_sim = (
    vars_dict['slope_corr_analysis_matched_btst'][k] for k in key_sub
)

#%% 
#---------------------------------------------------------------------------
# SECTION 2: load actual subject data
# --------------------------------------------------------------------------
#'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits'
#'Fitted_weibull_psychometric_func_Isoluminant plane_6000totalTrials_25refs_MOCS_sub1.pkl'
input_fileDir_fits_others, file_name_others = select_file_and_get_path()

# Define subject information
subject_list = [1,2,4,6,7,8,10,11]         # Only used if loading other subjects
subject_init = ['CH','ME', 'SG','DK','BH','FM','HG', 'FW']
nDatasets = len(subject_list)

# Initialize
slope_data        = np.full((nDatasets,), np.nan)
corr_coef_data    = np.full((nDatasets,), np.nan)
slope_CI_data     = np.full((nDatasets,2), np.nan)
corr_coef_CI_data = np.full((nDatasets,2), np.nan)

#loop through each dataset
for r in range(nDatasets):
    subject_id = subject_list[r]
    # Replace subject number
    match = re.search(r'sub\d+', file_name_others)
    if not match:
        raise ValueError('No "subXXX" pattern found in the file name!')
    old_sub = match.group()
    input_fileDir_fits_others_r = input_fileDir_fits_others.replace(old_sub, f'sub{subject_id}')
    file_name_r = file_name_others.replace(old_sub, f'sub{subject_id}')
    
    full_path_btst_r = f"{input_fileDir_fits_others_r}/{file_name_r}"
    
    # Load the variables from the current dataset
    with open(full_path_btst_r, 'rb') as f:
        vars_dict_sub_r = pickled.load(f)

    slope_data[r], corr_coef_data[r], slope_CI_data[r], corr_coef_CI_data[r] = (
        vars_dict_sub_r['slope_corr_analysis_matched_btst'][k] for k in key_sub
    )
        
#%% Plotting
# Combine simulation and empirical data for plotting
slope_all = np.append(slope_sim, slope_data)
corr_coef_all = np.append(corr_coef_sim, corr_coef_data)

slope_CI_all = np.vstack((slope_CI_sim, slope_CI_data))
corr_coef_CI_all = np.vstack((corr_coef_CI_sim, corr_coef_CI_data))

# Compute error bars (upper and lower deviations)
slope_CI_err = np.vstack((slope_all - slope_CI_all[:,0],
                              slope_CI_all[:,1] - slope_all))
corr_coef_CI_err = np.clip(np.vstack((corr_coef_all - corr_coef_CI_all[:,0], 
                              corr_coef_CI_all[:,1] - corr_coef_all)), 0, np.inf)

# Bar positions
x = np.concatenate(([0], np.arange(1.5, nDatasets+1.5, 1)))

# Plot
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 16
fig, axs = plt.subplots(2, 1, figsize=(6, 4), dpi=1024, sharex=True) #(5, 6) (7, 5)
# Slope plot
# Make a custom color array (each row is an RGB color)
colors = np.vstack((
    np.array([[0.75, 0.75, 0.75]]),                  # First bar: gray
    np.tile(np.array([[0.95, 0.95, 0.95]]), (nDatasets, 1)) # Other bars: white
))
axs[0].plot([0, nDatasets + 0.5], [1,1], lw= 2, ls= '--', c='k')
axs[0].bar(x, slope_all, yerr=slope_CI_err, capsize=10, 
           color= colors, edgecolor = 'grey', lw = 2)
axs[0].set_ylabel('Slope')
axs[0].set_ylim([0, 1.3]); axs[0].set_yticks(np.linspace(0, 1, 3)); 

# Correlation Coefficient plot
axs[1].bar(x, corr_coef_all, yerr=corr_coef_CI_err, capsize=10, 
           color= colors, edgecolor = 'grey', lw = 2)
axs[1].set_ylabel('Correlation Coefficient')
axs[1].set_ylim([0, 1]); axs[1].set_yticks(np.linspace(0, 1, 3))

# X-axis
axs[1].set_xticks(x)
axs[1].set_xticklabels(['Simulated\nsubject'] + [f'\n{s}' for s in subject_init])
plt.setp(axs[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()
# Save the figure as a PDF
output_figDir = os.path.join(base_dir, 'Experiment_FigFiles', 'pilot2', 'groupData', 'fits')
os.makedirs(output_figDir, exist_ok=True)
fig.savefig(os.path.join(output_figDir, 'Slope_corrcoef_groupData.pdf'),
             format='pdf', bbox_inches='tight')
