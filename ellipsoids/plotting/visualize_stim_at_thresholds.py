#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 22:45:31 2025

@author: fangfang

Goal of this script: 
    Reproduce and visualize the stimuli at discrimination thresholds for all MOCS conditions.

1. Load model fits to extract three types of stimuli:
(a) Thresholds predicted by the Weibull psychometric function fit to MOCS trials
(b) Thresholds predicted by the Wishart model fit to AEPsych trials
(c) Catch trials (i.e., easy comparisons) for each MOCS condition

2. Convert stimuli from Wishart space to RGB
The threshold stimuli are defined in Wishart (2D chromatic) space. We convert them 
to RGB using the transformation matrix and preview the colors using imshow() in Python.

3. Export the RGB values for each reference and comparison stimulus to a .pkl file.
This file can be loaded by sender.py to run a short Unity experiment using only these stimulus pairs.
⚠️ Don’t forget to enable the screenshot-saving flag in Unity!

4. Rename and organize saved screenshots
Unity will save .png images using RGB values in the filenames (e.g., Ref_R..._Target_R...).
We copy these into the designated folder:
/ELPS_analysis/Experiment_FigFiles/Pilot2/sub#/expt_stimuli/blobby
To make the filenames more readable, we match the RGB values from the filenames to 
the original stimulus conditions and rename them accordingly (e.g., MOCS_thres_1.png, 
                                                              Wishart_thres_1.png, etc.).

"""

import jax
jax.config.update("jax_enable_x64", True)
import dill as pickled
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

import os
from analysis.utils_load import select_file_and_get_path
from plotting.sim_CIELab_plotting import CIELabVisualization
from plotting.wishart_plotting import PlotSettingsBase
from plotting.sim_CIELab_plotting import PlotStimAtThresSettings
from analysis.color_thres import color_thresholds

#specify the file name
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'

#%%
# Load the fitted Weibull psychometric functions for a selected subject and condition
# File format: 'Fitted_weibull_psychometric_func_Isoluminant plane_6000totalTrials_25refs_MOCS_sub#.pkl'
input_fileDir_fits_MOCS, file_name_MOCS = select_file_and_get_path()
full_path_MOCS = os.path.join(input_fileDir_fits_MOCS, file_name_MOCS)

# Create an output directory for saving stimulus visualization figures
output_figDir = input_fileDir_fits_MOCS.replace('DataFiles', 'FigFiles').replace('fits', 'expt_stimuli/flat')
os.makedirs(output_figDir, exist_ok=True)

# Load data from the pickled file
with open(full_path_MOCS, 'rb') as f:
    MOCS = pickled.load(f)

# Extract thresholded stimulus data and reference locations
x1_at_thres_MOCS = MOCS['stim_at_targetPC_MOCS']  # Stimuli at estimated thresholds from MOCS
xref_MOCS = MOCS['xref_unique']                   # Reference stimuli locations
nRefs = MOCS['nRefs']                             # Number of unique reference locations

# Find the "easy trials" (catch trials with highest stimulus magnitude for each ref)
x1_catch_MOCS = np.full((nRefs, 2), np.nan)
for i in range(nRefs):
    vec_i = MOCS['fit_PMF_MOCS'][i].unique_stim
    norm_i = np.linalg.norm(vec_i, axis=-1)
    max_idx = np.argmax(norm_i)
    x1_catch_MOCS[i] = xref_MOCS[i] + vec_i[max_idx]

# Extract Wishart model-predicted thresholds at the same reference points
x1_at_thres_Wishart = MOCS['stim_at_targetPC_Wishart']

#%% OPTIONAL: Add suprathreshold stimuli
# This section is optional and is used to include a few hand-picked suprathreshold stimuli
# These are chosen manually for illustrative purposes (e.g., for making figures)

flag_add_stim_supra = False
if flag_add_stim_supra:
    # The Unity stimulus generation code only supports two types of stimuli per trial: xref and x1.
    # However, a suprathreshold experiment typically requires three distinct stimuli.
    # To work around this, we replicate the same stimulus across all three locations.
    # This allows us to later crop and reassemble screenshots into custom combinations for display.

    x1_supra = np.array([[0.3, 0.3],
                         [0, 0], 
                         [0.6, 0.6], 
                         [0.6, 0]])
    xref_supra = x1_supra

    # Concatenate all reference and comparison stimuli for plotting:
    # includes suprathreshold examples, MOCS threshold trials, MOCS catch trials, and Wishart model threshold trials
    MOCS_xref_cat = np.vstack((xref_supra, xref_MOCS, xref_MOCS, xref_MOCS))
    MOCS_x1_cat   = np.vstack((x1_supra,  x1_at_thres_MOCS, x1_catch_MOCS, x1_at_thres_Wishart))
else:
    # If not including suprathreshold stimuli, only plot MOCS and Wishart threshold-related stimuli
    MOCS_xref_cat = np.vstack((xref_MOCS, xref_MOCS, xref_MOCS))
    MOCS_x1_cat   = np.vstack((x1_at_thres_MOCS, x1_catch_MOCS, x1_at_thres_Wishart))

#%% Store in a dictionary for potential saving or export
MOCS_trials_W = {
    'MOCS_xref_shuffled': MOCS_xref_cat,
    'MOCS_x1_shuffled': MOCS_x1_cat
}

# -----------------------------------------------------------------------------
# SECTION 2: Visualize the stimuli at threshold
# -----------------------------------------------------------------------------
# Create color threshold transformation instance for the Isoluminant plane
color_thres_data = color_thresholds(2, base_dir, plane_2D='Isoluminant plane')
color_thres_data.load_transformation_matrix()

# Convert 2D chromatic coordinates (plus homogeneous 1) into RGB space
rgb_x1_at_thres_MOCS = color_thres_data.W2D_to_rgb(MOCS_x1_cat)
rgb_xref_MOCS = color_thres_data.W2D_to_rgb(MOCS_xref_cat)

MOCS_trials_RGB = {
    'MOCS_xref_shuffled': rgb_xref_MOCS.T,
    'MOCS_x1_shuffled': rgb_x1_at_thres_MOCS.T
}

# Set up plotting settings
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize=8)
pltStimSettings = replace(PlotStimAtThresSettings(), **pltSettings_base.__dict__)

# Construct figure names
fig_name_part1 = ['MOCS_thres'] * nRefs + ['MOCS_catch'] * nRefs + ['Wishart_thres'] * nRefs
fig_name_part2 = list(range(1, nRefs + 1)) * 3
fig_name_supra = [f'supra_thres_{i}' for i in range(xref_supra.shape[0])] if flag_add_stim_supra else []
fig_name = fig_name_supra + [f'{s1}_{s2}' for s1, s2 in zip(fig_name_part1, fig_name_part2)]

# Plot each reference + target stimulus pair as a small figure
for i in range(12):#MOCS_xref_cat.shape[0]
    xref_for_show_i = rgb_xref_MOCS[:, i]
    x1_for_show_i = rgb_x1_at_thres_MOCS[:, i]
    
    # Update figure name for current plot
    pltStimSettings = replace(pltStimSettings, fig_name=fig_name[i])
    
    # Visualize reference and comparison stimuli
    CIELabVisualization.visualize_stimuli_at_thres(
        np.vstack((xref_for_show_i, x1_for_show_i)).T,
        settings=pltStimSettings,
        save_fig=False
    )
    
    plt.show()
    
#%% 
# -----------------------------------------------------------------------------
# SECTION 3: Export out MOCS_trials_output
# -----------------------------------------------------------------------------
match = re.search(r'_sub\d+', file_name_MOCS)
output_file = f'Stim_at_thres_for_image_generation{match[0]}.pkl'
full_path = os.path.join(input_fileDir_fits_MOCS, output_file)

variable_names = ['MOCS_trials_W', 'MOCS_trials_RGB']
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

#%% 
# -----------------------------------------------------------------------------
# SECTION 4: Update figure names
# -----------------------------------------------------------------------------
# flag for updating the file names
flag_change_blobby_filename = False
output_figDir_blobby = output_figDir.replace('flat','blobby')
# Helper function to safely extract and convert a float from a potentially malformed string
def extract_float(s):
    match = re.search(r'\d+\.\d+', s)  # match first valid float pattern
    if match:
        return float(match.group())
    else:
        raise ValueError(f"Invalid float format: {s}")

if flag_change_blobby_filename and os.path.exists(output_figDir_blobby):        
    for fname in os.listdir(output_figDir_blobby):
        if fname.endswith('.png') and fname.startswith('Ref'):
            # Match RGB values for Ref and Target in the filename
            match = re.search(
                r'Ref_R([0-9.]+)_G([0-9.]+)_B([0-9.]+)_Target_R([0-9.]+)_G([0-9.]+)_B([0-9.]+)',
                fname
            )
            if match:
                # Safely extract all 6 floats
                blobby_ref_rgb = [extract_float(match.group(i)) for i in range(1, 4)]
                blobby_comp_rgb = [extract_float(match.group(i)) for i in range(4, 7)]
            
            #find the index that matches
            rgb_diff_ref = np.sum(np.abs(np.array(blobby_ref_rgb)[np.newaxis] - MOCS_trials_RGB['MOCS_xref_shuffled']), axis = 1)
            rgb_diff_comp = np.sum(np.abs(np.array(blobby_comp_rgb)[np.newaxis] - MOCS_trials_RGB['MOCS_x1_shuffled']), axis = 1)
            idx_match = np.argmin(rgb_diff_ref + rgb_diff_comp)
            file_rename = fig_name[idx_match] + '.png'
        
            #replace the original file name with file_rename
            src_path = os.path.join(output_figDir_blobby, fname)
            dst_path = os.path.join(output_figDir_blobby, file_rename)

            # Rename file
            os.rename(src_path, dst_path)

#indicate the odd stimulus location in the file name
if flag_change_blobby_filename and os.path.exists(output_figDir_blobby):        
    # Look for the Unity CSV trial data file
    csv_file = next(
        (os.path.join(output_figDir_blobby, f)
         for f in os.listdir(output_figDir_blobby)
         if f.startswith('Unity_trial_data') and f.endswith('.csv')),
        None
    )

    if csv_file:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()  # clean up column names
        oddLoc = df['OddLocation']

        # Map numeric odd location to label
        loc_map = {1: 'top', 2: 'left', 3: 'right'}

        # Loop through each reference and rename the file
        for n in range(nRefs * 3):
            src = os.path.join(output_figDir_blobby, fig_name[n] + '.png')
            odd_str = loc_map.get(oddLoc[n], 'unknown')
            file_rename = f"{fig_name[n]}_odd_{odd_str}.png"
            dst = os.path.join(output_figDir_blobby, file_rename)
            os.rename(src, dst)



