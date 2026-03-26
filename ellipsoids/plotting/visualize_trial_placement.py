#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:47:02 2025

@author: fangfang

# The purpose of this script is solely to illustrate trial placement.
# While other scripts (e.g., sim_4d6d_color_discrimination.py, fit_4d_human.py)
# also visualize trial placement, they include additional components such as model fitting.
# In contrast, this script is focused exclusively on visualizing trial placement, 
# with no other functionality.

"""

import matplotlib.pyplot as plt
import dill as pickled
from dataclasses import replace
import re
import numpy as np
np.random.seed(None)

import os
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization, Plot2DSamplingSettings
from plotting.wishart_plotting import PlotSettingsBase 
from analysis.utils_load import select_file_and_get_path, extract_sub_number

#%%
#---------------------------------------------------------------------------
# SECTION 1: load the model fits to the empirical data
# --------------------------------------------------------------------------
COLOR_DIMENSION = 4
#the loaded variable names will be different depending on whether we are loading AEPsych trials or MOCS trials
flag_load_MOCS = True

"""
We have 4 options:
    1. Experimental adaptive trials (Adaptively sampled AEPsych trials)
    2. Simulated adaptive trials (Adaptively sampled AEPsych trials based on a known ground truth)
    3. Experimental MOCS trials
    4. Simulated MOCS trials (based on a known ground truth)
    
Case 1: ELPS_analysis/Experiment_DataFiles/sub#/fits
    'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub#_decayRate0.5_nBasisDeg5.pkl'

Case 2: META_analysis/ModelFitting_DataFiles/4dTask/CIE/sub#/decayRate0.5
    'Fitted_byWishart_Isoluminant plane_4DExpt_300_300_300_5100_AEPsychSampling_EAVC_decayRate0.5_nBasisDeg5_sub#.pkl'
    
Case 3: META_analysis/ModelFitting_DataFiles/4dTask/CIE/sub#/decayRate0.5
    'Fitted_weibull_psychometric_func_Isoluminant plane_6000totalTrials_25refs_MOCS_sub#.pkl'

Case 4: ELPS_analysis/Simulation_DataFiles/Isoluminant plane/MOCS/gt_CIE
    'Sim2dTask_colorDiscrimination_Isoluminant plane_MOCStrials_25refs_12levels_20trialsPerLevel_subCIE1994_Sobol_seed2000.pkl'
"""
input_fileDir_fits, file_name = select_file_and_get_path()

# Construct the full path to the selected file
full_path = os.path.join(input_fileDir_fits, file_name)

#specify figure name and path
# First replace 'ModelFitting' with 'Simulation', then 'DataFiles' with 'FigFiles'
output_figDir_fits = re.sub(r'ModelFitting', 'Simulation', input_fileDir_fits)
output_figDir_fits = re.sub(r'DataFiles', 'FigFiles', output_figDir_fits)

# Create directories if they don't exist
os.makedirs(output_figDir_fits, exist_ok=True)

#%% 
# ---------------------------------------------------------------------------
# SECTION 2: Load the necessary variables from the file
# --------------------------------------------------------------------------
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)

# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict['color_thres_data']
# - Experimental trial data
if flag_load_MOCS:
    try: #simulated data
        #subject number
        subN = re.search(r'(?<=sub)([^_]+)', file_name)[0]
        nUnique_cond = vars_dict['nRefs']*vars_dict['nLevels']
        xref = vars_dict['MOCS_xref_shuffled'][:nUnique_cond]
        x1 = vars_dict['MOCS_x1_shuffled'][:nUnique_cond]
    except: #experimental data
        subN = extract_sub_number(file_name)
        #take the first 12 unique levels
        xref_temp = [arr[:vars_dict['nLevels']] for arr in vars_dict['refStimulus']]  # Take first 12 rows of each (240, 2) array
        xref = np.concatenate(xref_temp, axis=0)  # Concatenate into a single (300, 2) array
        x1_temp = [arr[:vars_dict['nLevels']] for arr in vars_dict['compStimulus']]
        x1 = np.concatenate(x1_temp, axis=0)
else:
    subN = extract_sub_number(file_name)
    try: #experimental data
        expt_trial = vars_dict['expt_trial']
        NTRIALS_STRAT = [300,300,300,5100]#vars_dict['NTRIALS_STRAT']
    except: #simulated data
        expt_trial = vars_dict['AEPsych_trial_given_Wishart_gt']
        strat_dict = vars_dict['strat_dict']
        NTRIALS_STRAT = list(vars_dict['strat_dict'].values())
    nTrials_AEPsych = sum(NTRIALS_STRAT)
    xref = expt_trial.xref_all
    x1 = expt_trial.x1_all
    nTrials_total = xref.shape[0]

#%%
# -----------------------------------------------------------------------
# SECTION 3: Visualize trial placement (separately for Sobol and EAVC)
# -----------------------------------------------------------------------
# Create settings instance with custom fig_dir
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 8)
pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
sampling_vis = SamplingRefCompPairVisualization(2,
                                                color_thres_data,
                                                settings = pltSettings_tp,
                                                save_fig = False)

if flag_load_MOCS:
    marker_alpha = [0.6]
    slc_datapoints_to_show_lb = [0]
    slc_datapoints_to_show_ub = [xref.shape[0]]
    str_ext = '_MOCS'
else:
    # This array defines the opacity of markers in the plots, decreasing with more trials.
    marker_alpha = [0.5, 0.3, 0.5, 0.2]
    # Define specific slices of data points to be visualized, ranging from very few to many.
    slc_datapoints_to_show_lb = [0,                       
                                 sum(NTRIALS_STRAT[:-1]),   
                                 nTrials_AEPsych,             
                                 0]
    slc_datapoints_to_show_ub = [sum(NTRIALS_STRAT[:-1]),         
                                 nTrials_AEPsych,     
                                 nTrials_total, 
                                 nTrials_total]
    str_ext = ''

# Loop over the selected data points to generate and visualize each corresponding figure.
for i, (lb_i, ub_i) in enumerate(zip(slc_datapoints_to_show_lb, slc_datapoints_to_show_ub)):
    # Construct a filename for each figure based on the plane and number of experiments.
    str_idx = f'{ub_i:05}total' if lb_i == 0 else f'{ub_i:05}total_from{lb_i:05}'
    fig_name = f"TrialPlacement{str_ext}_isothreshold_Isoluminant plane_{COLOR_DIMENSION}DExpt_{str_idx}_sub{subN}"
    pltSettings_tp = replace(pltSettings_tp,
                             ref_markeralpha = 0.7,#marker_alpha[i],
                             comp_markeralpha = marker_alpha[i],
                             linealpha = marker_alpha[i], 
                             ticks = np.linspace(-0.8, 0.8, 5),
                             bounds = 0.75 * np.array([-1,1]),
                             fig_name = fig_name)

    fig, ax = plt.subplots(1, 1, figsize = (3,3.5), dpi= pltSettings_tp.dpi)
    # Visualize the trials up to the nth data point with specified marker transparency.
    sampling_vis.plot_sampling(xref[lb_i:ub_i],  # Reference points up to the nth data point
                               x1[lb_i:ub_i],    # Comparison points up to the nth data point
                               settings = pltSettings_tp,
                               ax = ax)              # Filename under which the figure will be saved
    ax.set_title('Isoluminant plane')
    # Save the figure as a PDF
    #fig.savefig(os.path.join(output_figDir_fits, fig_name +'.pdf'), bbox_inches='tight')    
    plt.show()