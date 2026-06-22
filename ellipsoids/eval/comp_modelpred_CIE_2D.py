#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:34:27 2025

@author: fangfang

This script compares discrimination thresholds predicted by the Wishart
Process Psychophysical Model (WPPM) with thresholds derived from CIE
color-difference metrics (e.g., ΔE76, ΔE94, ΔE2000).

Workflow
--------
1. Load the WPPM fit to empirical data (ellipses in model W space).
2. Load bootstrap fits and compute confidence envelopes of the predicted
   threshold contours across bootstrap datasets.
3. Load WPPM fits to simulated CIE threshold predictions.
4. Visualize the comparison between WPPM predictions and CIE predictions
   in the model space.
5. Convert predictions from model W space to CIELab space and visualize
   the comparison in perceptual coordinates.

Key outputs
-----------
- Model predictions and CIE predictions plotted in model space.
- Corresponding comparison plots in CIELab space.

Notes
-----
- The script assumes that the stimulus grid used in the WPPM fit matches
  the grid used in the CIE simulations.
- Bootstrap datasets are ranked using a similarity metric (summed NBS)
  and the top 95% are retained to construct CIs.

Figures are saved to:
    ELPS_analysis/Experiment_FigFiles/.../comp_CIE

"""

import jax
jax.config.update("jax_enable_x64", True)
import dill as pickled
import re
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
from copy import deepcopy
from tqdm import trange
import os
from analysis.utils_load import select_file_and_get_path
from analysis.conf_interval import find_inner_outer_contours_for_gridRefs, \
    find_btst_dataset_within_CI
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization,\
    Plot2DPredSettings, add_CI_ellipses
from plotting.wishart_plotting import PlotSettingsBase 

#specify the file name
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'

#%%
# -------------------------------------------------------------------------
# SECTION 1: Load the model fit to the empirical dataset
# -------------------------------------------------------------------------
# Example:
# ELPS_analysis/Experiment_DataFiles/sub#/...
# Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.4_nBasisDeg5.pkl
input_fileDir_fits, file_name = select_file_and_get_path()
full_path = os.path.join(input_fileDir_fits, file_name)

# Load variables from the selected fit file.
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)

# Extract key variables needed for downstream analysis.
subN = vars_dict['subN']
# expt_trial = vars_dict_set1['expt_trial']
mp = deepcopy(vars_dict['model_pred_Wishart'])
color_thres_data = vars_dict['color_thres_data']
grid = vars_dict['grid']

# -------------------------------------------------------------------------
# SECTION 2: Load bootstrap fits and keep those within the CI 
# -------------------------------------------------------------------------
# Example:
# ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits/AEPsych_btst/decayRate0.4/...
# Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.4_...
#     _btst_AEPsych[0].pkl
input_fileDir_fits_btst, file_name_btst = select_file_and_get_path()
input_path_btst = os.path.join(input_fileDir_fits_btst, file_name_btst)

# number of bootstrapped datasets
nBtst = 120
# initialize normalized Bures Similarity
NBS_sum = []
# fit ellipses on a grid of references
fitEll_params = []

for n in trange(nBtst):
    # Replace the bootstrap index in the template filename.
    input_path_btst_n = input_path_btst.replace('AEPsych[0]', f'AEPsych[{n}]')

    # Load the nth bootstrap fit.
    with open(input_path_btst_n, 'rb') as f:
        vars_dict_btst_n = pickled.load(f)

    # Store fitted ellipses for this bootstrap dataset.
    fitEll_params.append(vars_dict_btst_n['model_pred_Wishart'].params_ell)

    # Collapse NBS across the full grid to obtain one similarity score
    # per bootstrap dataset.
    NBS_sum.append(np.sum(vars_dict_btst_n['NBS_fine_grid']))

# Rank bootstrap datasets by similarity to the original fit and keep
# the top 95% (default behavior of find_btst_dataset_within_CI).
fitEll_params_keep_list, *_ = find_btst_dataset_within_CI(NBS_sum, fitEll_params)

fitEll_params_keep = np.moveaxis(np.array(fitEll_params_keep_list), 0, -2)

# Compute confidence interval contours at each grid point
#   - fitEll_min / fitEll_max represent the inner/outer envelopes across datasets
ell_min, ell_max = find_inner_outer_contours_for_gridRefs(fitEll_params_keep)

#%%
# -------------------------------------------------------------------------
# SECTION 3: Load WPPM fit to CIE-based threshold predictions
# -------------------------------------------------------------------------
# Example location:
#   ELPS_analysis/Simulation_DataFiles/
# Example filename:
#   'Isothreshold_ellipses_isoluminant_CIE1994.pkl'
input_fileDir_fits_CIE, file_name_CIE = select_file_and_get_path()

# Extract the CIE metric version from the filename (e.g., CIE1976, CIE1994, CIE2000)
CIE_version = re.search(r'(CIE\d+)', file_name_CIE)[0]

# Construct the full path to the selected file
full_path_CIE = os.path.join(input_fileDir_fits_CIE, file_name_CIE)

# Load simulation results from the pickle file
with open(full_path_CIE, 'rb') as f:
    vars_dict_set2 = pickled.load(f)

# Retrieve predicted ellipses in W space (scaled) from the simulation results
# The key depends on the grid resolution used in the Wishart model fits
CIE_pred_W_unit = vars_dict_set2[f'results_grid{grid.shape[0]}']['fitEllipse_scaled'][0]

# Retrieve the stimulus grid used in the CIE simulations
CIE_grid = vars_dict_set2[f'stim_grid{grid.shape[0]}']['grid_ref']

# Retrieve the object used to convert RGB values to CIELab space
sim_thres_CIELab = vars_dict_set2['sim_thres_CIELab_grid7']

# Verify that the stimulus grids match between the Wishart model fit
# and the CIE simulation results.
if not np.all(np.abs(np.sort(np.unique(grid)) - CIE_grid) < 1e-5):
    raise ValueError("The stimulus grid does not match between the two files.")
    
#%%
# -------------------------------------------------------------------------
# SECTION 4: Visualize model predictions in model space
# -------------------------------------------------------------------------
output_figDir = os.path.join(os.path.dirname(input_fileDir_fits.replace('Experiment_DataFiles',
                                             'Experiment_FigFiles')), 'comp_CIE')
os.makedirs(output_figDir, exist_ok=True)
fig_name_part1 = f'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_vs_{CIE_version}'
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 8) 

#specify figure name and path    
# Initialize 2D prediction settings by copying from base and overriding method-specific parameters
pred2D_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
pred2D_settings = replace(pred2D_settings, 
                          visualize_samples= False,
                          visualize_gt = False,
                          visualize_model_estimatedCov = False,
                          modelpred_alpha = 0.9,
                          modelpred_lw = 1,
                          modelpred_ls = '-',
                          title = f'Comparison with {CIE_version}',
                          modelpred_label = 'Model predictions given the original dataset',
                          ticks = np.linspace(-0.7, 0.7,5),
                          flag_rescale_axes_label = False)

wishart_pred_vis_set1 = WishartPredictionsVisualization(None,
                                                        mp.model, 
                                                        mp, 
                                                        color_thres_data,
                                                        settings = pltSettings_base)

#visualize samples and model-estimated cov matrices
#customize cmap for the isoluminant plane
fig, ax = plt.subplots(1, 1, figsize = pred2D_settings.fig_size, dpi= pred2D_settings.dpi)
#model fits to the original dataset
wishart_pred_vis_set1.plot_2D(grid, ax = ax, settings = pred2D_settings)
lbls1 = f'{CIE_version} predictions'
lbls2 = f'The 95% confidence interval of model\npredictions given {nBtst} bootstrap datasets'

for idx in np.ndindex(grid.shape[:-1]):
    lbl1 = lbls1 if idx == (0, 0) else None
    lbl2 = lbls2 if idx == (0, 0) else None
    ax.plot(*CIE_pred_W_unit[*idx], color = 'k', alpha = 0.7,lw= 1, ls = '-', label = lbl1)
    
    #model fits to the bootstrapped datasets
    add_CI_ellipses(ell_min[*idx], ell_max[*idx], ax, 
                    cm =color_thres_data.W2D_to_rgb(grid[*idx]),
                    alpha = 0.5, label = lbl2)

# Manually add the legend
ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.45), fontsize = pred2D_settings.fontsize-1)
# Save the figure as a PDF
fig.savefig(os.path.join(output_figDir,f"{fig_name_part1}.pdf"), bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------------
# SECTION 5: Visualize model predictions in CIELab space
# -------------------------------------------------------------------------
# Convert points from W space to CIELab space while preserving the original grid structure
def W_to_Lab(W):
    # Move the coordinate axis (size=2) to the last dimension so that
    # each row represents one W coordinate: (..., 2)
    W = np.moveaxis(W, -2, -1)

    # Flatten the grid so the conversion can be applied to all points at once
    W_flat = W.reshape(-1, 2)

    # Convert W → RGB using the experiment's display transformation
    rgb_flat = color_thres_data.W2D_to_rgb(W_flat)

    # Convert RGB → CIELab
    lab_flat, *_ = sim_thres_CIELab.convert_rgb_lab(rgb_flat)

    # Restore the original grid shape and move the Lab channel axis back
    return np.moveaxis(lab_flat.reshape(W.shape[:-1] + (3,)), -2, -1)


# mp.fitEll_unscaled has shape (7, 7, 2, 200):
#   grid_x × grid_y × (W coordinates) × ellipse points
#
# After conversion:
# fitEll_lab has shape (7, 7, 3, 200):
#   grid_x × grid_y × (Lab channels) × ellipse points
fitEll_lab = W_to_Lab(mp.fitEll_unscaled)

#CIE predictions
CIE_pred_lab = W_to_Lab(CIE_pred_W_unit)

#grid points to LAB space
grid_lab = W_to_Lab(grid[..., None])

#reshape
ell_min_lab = W_to_Lab(ell_min)
ell_max_lab = W_to_Lab(ell_max)
    
fig2, ax2 = plt.subplots(1, 1, figsize = pred2D_settings.fig_size, dpi= pred2D_settings.dpi)
for idx in np.ndindex(grid.shape[:-1]):
    lbl0 = 'Model predictions given the original dataset'  if idx == (0, 0) else None
    lbl1 = lbls1 if idx == (0, 0) else None
    lbl2 = lbls2 if idx == (0, 0) else None
    
    #WPPM predictions
    ax2.plot(*fitEll_lab[*idx, 1:], color = color_thres_data.W2D_to_rgb(grid[*idx]),
             alpha = 0.7,lw= 1, label = lbl0)

    #CIE predictions
    ax2.plot(*CIE_pred_lab[*idx,1:], ls= '-', color = 'k', alpha = 0.7, lw = 1, label = lbl1)
    
    # WPPM prediction CI
    add_CI_ellipses(ell_min_lab[*idx,1:], ell_max_lab[*idx,1:], ax2, 
                    cm =color_thres_data.W2D_to_rgb(grid[*idx]),
                    alpha = 0.5, label = lbl2)
    
ax2.set_xlabel(r'CIE $a^*$')
ax2.set_ylabel(r'CIE $b^*$')
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, linestyle=':', linewidth=0.7, color='lightgrey')
ax2.grid(alpha=0.2,linestyle='-')
ax2.set_xlim([-80,65])
ax2.set_ylim([-50,70])
plt.legend(loc='lower center', bbox_to_anchor= (0.5, -0.57), fontsize = 7.5)
fig2.tight_layout()
fig2.savefig(os.path.join(output_figDir,f"{fig_name_part1}_Lab_space.pdf"),
             format='pdf', bbox_inches='tight')
plt.show()
