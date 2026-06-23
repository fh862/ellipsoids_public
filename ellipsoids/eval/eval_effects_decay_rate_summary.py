#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 12:45:02 2026

@author: fangfang

Goal
----
This script:
1) Loads the linear-regression results comparing WPPM-predicted thresholds 
   against validation thresholds, for all subjects and a sweep of decay-rate 
   hyperparameters.
2) Plots the regression slope and correlation coefficient as a function of 
   the decay rate, to visualize how this hyperparameter affects validation 
   performance.

In short, this is a diagnostic script for evaluating how the decay-rate 
hyperparameter influences the quality and stability of WPPM predictions.

"""

import dill as pickled
import re
import numpy as np
import matplotlib.pyplot as plt
import os
from analysis.utils_load import get_path
import matplotlib as mpl

#%% 
# -------------------------------------------------------------------------
# Set up paths so we can import from the parent directory if needed
# -------------------------------------------------------------------------

# Base directory where all result files are stored
base_dir = get_path("dropbox_root_mac")

# -------------------------------------------------------------------------
# SECTION 1: Load regression results across subjects and decay rates
# -------------------------------------------------------------------------

def find_file_path(file_name, base_dir):
    """Recursively search for a file in base_dir and return its full path."""
    for root, dirs, files in os.walk(base_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    raise FileNotFoundError(f"Data files directory not found for file {file_name}.")

# Decay-rate values to sweep over
decayRates = np.linspace(0.1, 1.0, 10)
num_dr = len(decayRates)

# Subject IDs included in the analysis
subject_list = [1, 2, 4, 6, 7, 8, 10, 11]
num_sub = len(subject_list)

# Preallocate arrays for regression metrics
# Rows: decay rates, Columns: subjects
slope = np.full((num_dr, num_sub), np.nan)
corr_coef = np.full(slope.shape, np.nan)

# Template filename (subject ID and decay rate will be replaced)
file_name_template = (
    "Fitted_weibull_psychometric_func_Isoluminant plane"
    "_6000totalTrials_25refs_MOCS_sub1_decayRate0.1_varScaler0.0003_nBasisDeg5.pkl"
)

# Loop over decay rates and subjects to load results
for i, dr in enumerate(decayRates):
    dr_str = f"{dr:.1f}"  # Format to match filename (e.g., 0.1, 0.2, ..., 1.0)
    
    for j, sub in enumerate(subject_list):
        # Replace subject number in filename
        file_name = re.sub(r"_sub\d+", f"_sub{sub}", file_name_template)
        
        # Replace decay rate in filename
        file_name = re.sub(r"decayRate[0-9.]+", f"decayRate{dr_str}", file_name)
        
        print(file_name)
        
        # Find the full path to the file
        file_dir_s = find_file_path(file_name, base_dir)

        # Load saved regression results
        with open(file_dir_s, 'rb') as f:
            vars_dict = pickled.load(f)
        
        # Extract slope and correlation coefficient
        slope[i, j] = vars_dict['slope_modelPred_org'].item()
        corr_coef[i, j] = vars_dict['corr_coef_modelPred_org']

# Compute across-subject mean and standard deviation
slope_avgSub = np.mean(slope, axis=1)
slope_stdSub = np.std(slope, axis=1)

corr_coef_avgSub = np.mean(corr_coef, axis=1)
corr_coef_stdSub = np.std(corr_coef, axis=1)

#%%
# -------------------------------------------------------------------------
# SECTION 2: Plot results as a function of decay rate
# -------------------------------------------------------------------------
mpl.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(2, 1, sharex=True)

# Plot individual-subject data as grey scatter points
for i, dr in enumerate(decayRates):
    label = f"individual data (N = {num_sub})" if i == 0 else None
    
    ax[0].scatter([dr] * num_sub, slope[i],
                  color='grey', edgecolor='white', alpha=0.5, s=50, label=label)
    
    ax[1].scatter([dr] * num_sub, corr_coef[i],
                  color='grey', edgecolor='white', alpha=0.5, s=50, label=label)

# Plot mean ± SD for slope
ax[0].plot(decayRates, slope_avgSub, color='k', marker='.', ms=5)
ax[0].errorbar(decayRates, slope_avgSub, yerr=slope_stdSub,
               fmt='none', color='k', capsize=3)

# Plot mean ± SD for correlation coefficient
ax[1].plot(decayRates, corr_coef_avgSub, color='k', marker='.', ms=5, label='mean')
ax[1].errorbar(decayRates, corr_coef_avgSub, yerr=corr_coef_stdSub,
               fmt='none', color='k', capsize=3, label=r'$\pm$1 SD')

# Axis labels
ax[0].set_ylabel("Slope")
ax[1].set_ylabel("Correlation coefficient")
ax[1].set_xlabel(r"Hyperparameter $\epsilon$")

# Add light grid to both panels
for a in ax:
    a.grid(True, alpha=0.3)

# Hide x-axis ticks on the top panel
ax[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Legend only on the bottom panel
ax[1].legend(fontsize=9, ncol=3, frameon=True, fancybox=True,
             framealpha=1.0, edgecolor='0.7', loc='best')

output_figdir = os.path.join(base_dir, 'ELPS_analysis', 'Experiment_FigFiles', 
                             'pilot2', 'GroupData', 'effects_decayRates')
os.makedirs(output_figdir, exist_ok = True)
fig.savefig(os.path.join(output_figdir, 'slope_corrcoef_w_varyingDecayRates.pdf'),
    bbox_inches='tight'
)

