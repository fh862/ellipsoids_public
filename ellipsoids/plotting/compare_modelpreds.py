#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:24:00 2025

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import dill as pickled
import numpy as np
import re
import matplotlib.pyplot as plt
from dataclasses import replace
import os
from analysis.utils_load import get_path
from analysis.utils_load import select_file_and_get_path
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.wishart_predictions_plotting import Plot2DPredSettings
#specify the file name
base_dir = get_path("dropbox_root_mac_elps")

#%%
#---------------------------------------------------------------------------
# SECTION 1: load the model fits to the empirical data
# --------------------------------------------------------------------------
# Select the file containing the model fits
# Navigate to the directory: ELPS_analysis/Experiment_DataFiles/sub#
input_fileDir_fits_set1, file_name_set1 = select_file_and_get_path()

# Construct the full path to the selected file
full_path_set1 = os.path.join(input_fileDir_fits_set1, file_name_set1)

# Load the necessary variables from the file
with open(full_path_set1, 'rb') as f:
    vars_dict_set1 = pickled.load(f)
    
# Retrieve all the keys from load_varnames and create variables with '_set1' suffix
subN = vars_dict_set1['subN']
expt_trial = vars_dict_set1['expt_trial']
model_pred_Wishart_set1 = vars_dict_set1['model_pred_Wishart']
color_thres_data = vars_dict_set1['color_thres_data']
grid = vars_dict_set1['grid']

#set2
load_varnames_set2 = ['model_pred_Wishart']

input_fileDir_fits_set2, file_name_set2 = select_file_and_get_path()

# Construct the full path to the selected file
full_path_set2 = os.path.join(input_fileDir_fits_set2, file_name_set2)
#extract session information
match = re.search(r'sub\d+_(.*?)\.pkl', file_name_set2)
sesInfo = match.group(1)

# Load the necessary variables from the file
with open(full_path_set2, 'rb') as f:
    vars_dict_set2 = pickled.load(f)
model_pred_Wishart_set2 = vars_dict_set2['model_pred_Wishart']
    
#%%
output_figDir_fits = os.path.join(base_dir, 'Experiment_FigFiles','pilot2',f'sub{subN}', 'add_MOCS_data')#'by_sessions')
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 8)
fig_name_part1 = 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_'+\
    f'all_vs_partial_AEPsychData_{sesInfo}' #decay0.4vs0.8
# -----------------------------------------
# SECTION 6: Visualize model predictions
# -----------------------------------------
#specify figure name and path    
pred2D_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
wishart_pred_vis_set1 = WishartPredictionsVisualization(expt_trial,
                                                   model_pred_Wishart_set1.model, 
                                                   model_pred_Wishart_set1, 
                                                   color_thres_data,
                                                   settings = pred2D_settings)

wishart_pred_vis_set2 = WishartPredictionsVisualization(expt_trial,
                                                   model_pred_Wishart_set2.model, 
                                                   model_pred_Wishart_set2, 
                                                   color_thres_data,
                                                   settings = pred2D_settings)
pred2D_settings1 = replace(pred2D_settings, 
                           visualize_samples= False,
                           visualize_gt = False,
                           visualize_model_estimatedCov = False,
                           modelpred_alpha =1,
                           modelpred_lw = 0.7,
                           modelpred_lc = 'k',
                           modelpred_ls = '--',
                           modelpred_label = 'Model predictions (w/o MOCS trials)', #(all AEPsych data)
                           ticks = np.linspace(-0.6, 0.6,5),
                           flag_rescale_axes_label = False)

pred2D_settings2 = replace(pred2D_settings, 
                           visualize_samples= False,
                           visualize_gt = False,
                           visualize_model_estimatedCov = False,
                           modelpred_alpha = 0.75,
                           modelpred_lw = 1.5,
                           modelpred_ls = '-',
                           modelpred_label = 'Model predictions (w MOCS trials)', #partial AEPsych data
                           ticks = np.linspace(-0.6, 0.6,5),
                           flag_rescale_axes_label = False)

#visualize samples and model-estimated cov matrices
#customize cmap for the isoluminant plane
fig1, ax1 = plt.subplots(1, 1, figsize = pred2D_settings1.fig_size, dpi= pred2D_settings1.dpi)
wishart_pred_vis_set1.plot_2D(grid, ax = ax1, settings = pred2D_settings1)
wishart_pred_vis_set2.plot_2D(grid, ax = ax1, settings = pred2D_settings2)
ax1.set_title('Isoluminant plane');
# Save the figure as a PDF
fig1.savefig(os.path.join(output_figDir_fits,f"{fig_name_part1}.pdf"),
             format='pdf', bbox_inches='tight')
plt.show()
