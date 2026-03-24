#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:55:32 2025

@author: fangfang

To quantify how well the WPPM estimated thresholds agree with validation threshold,
we performed a linear regression. The results further support strong agreement 
between the two sets of estimates (mean correlation coefficient = 0.82, range = 0.73–0.97). 
For 7 out of 8 participants, the regression slope was not significantly different 
from 1 (mean slope = 1.04, range = 0.97 - 1.13), indicating no systematic bias 
in the WPPM fits with respect to scale. 

The goal of this script is to assess whether there were more subtle sources of bias 
not captured by the regression slope. To this aim, we conducted an additional analysis
of the residuals—the discrepancies between the WPPM estimates and validation thresholds. 

While we found no evidence that residuals depended on the orientation or shape of the 
WPPM-predicted ellipses, we did observe a small but statistically significant trend: 
    the model slightly overestimated thresholds when validation thresholds were low and 
    underestimated them when validation thresholds were high. Nonetheless, the magnitude 
    of this bias in residuals was small relative to the overall range of thresholds.

"""

import dill as pickled
import re
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import linregress
import statsmodels.api as sm
import sys
import os
script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from analysis.utils_load import select_file_and_get_path
from analysis.ellipses_tools import symmetric_angle_difference

#%% 
#---------------------------------------------------------------------------
# SECTION 1: load data
# --------------------------------------------------------------------------
flag_data = False #whether we want to load data or simulated subject
if flag_data:
    # Step 1: Define mode
    subject_list = [1,2,4,6,7,8,10,11]      #subject numbers
else:
    subject_list = [1]
nDatasets = len(subject_list)
nRefs = 25  # Number of reference stimuli per subject

# Step 1: Select the bootstrapped data file (choose one as a reference)
# 'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits'
# 'Fitted_weibull_psychometric_func_Isoluminant plane_6000totalTrials_25refs_MOCS_sub#.pkl'

# OR
#'ELPS_analysis/Simulation_DataFiles/MOCS/gt_CIE'
#'Fitted_weibull_psychometric_func_Isoluminant plane_240totalTrials_25refs_MOCS_subCIE1994_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl'
input_fileDir_fits_others, file_name_others = select_file_and_get_path()

# Initialize storage arrays for extracted values
params_ell = np.full((nDatasets, nRefs, 5), np.nan)          # Ellipse params: [x, y, major, minor, angle]
diff_rotAngle = np.full((nDatasets, nRefs), np.nan)          # Rotation angle difference (deg)
thres_est_Wishart = np.full((nDatasets, nRefs), np.nan)      # Thresholds from WPPM
thres_est_MOCS = np.full((nDatasets, nRefs), np.nan)         # Thresholds from MOCS validation
diff_thres_est = np.full((nDatasets, nRefs), np.nan)         # Residuals: WPPM - MOCS
ratio_major_over_minor = np.full((nDatasets, nRefs), np.nan) # Aspect ratio (major/minor)
xref_unique = np.full((nDatasets, nRefs, 2), np.nan)         # Reference chromaticity

#% Step 2: Loop through each bootstrap dataset and load data
for r in range(nDatasets):
    subject_id = subject_list[r]
    # Replace subject number
    if flag_data:
        match = re.search(r'sub\d+', file_name_others)
        if not match:
            raise ValueError('No "subXXX" pattern found in the file name!')
        old_sub = match.group()
        input_fileDir_fits_r = input_fileDir_fits_others.replace(old_sub, f'sub{subject_id}')
        file_name_r = file_name_others.replace(old_sub, f'sub{subject_id}')
    else:
        input_fileDir_fits_r = input_fileDir_fits_others
        file_name_r = file_name_others

    # Generate the file name for the current bootstrap dataset
    full_path_others_r = f"{input_fileDir_fits_r}/{file_name_r}"
    
    # Load the variables 
    with open(full_path_others_r, 'rb') as f:
        MOCS = pickled.load(f)
        
    # Compute chromatic directions (vectors from reference to comparison stimuli)
    xref_unique[r] = MOCS['xref_unique']
    vec_chromaDir_MOCS = MOCS['stim_at_targetPC_Wishart'] - xref_unique[r]
    rotAngle_chromaDir_MOCS = np.rad2deg(np.arctan2(vec_chromaDir_MOCS[:, 1],
                                                    vec_chromaDir_MOCS[:, 0]))   # deg

    # Load Wishart model predictions at each validation location
    model_pred_Wishart_MOCS = MOCS['model_pred_Wishart_MOCS']
    for j in range(nRefs):
        # Extract ellipse parameters and compute aspect ratio
        params_ell[r,j] = model_pred_Wishart_MOCS.params_ell[0][j]
        _, _, major_axis, minor_axis, _ = params_ell[r,j]
        ratio_major_over_minor[r,j] = major_axis / minor_axis
        
        # Compute threshold estimates and residuals (difference between predicted and validation)
        thres_est_Wishart[r,j] = MOCS['vecLen_at_targetPC_Wishart'][j]
        thres_est_MOCS[r,j] = MOCS['fit_PMF_MOCS'][j].stim_at_targetPC
        diff_thres_est[r,j] = thres_est_Wishart[r,j] - thres_est_MOCS[r,j]                             
        
    # Compute absolute angular difference between predicted ellipse orientation and 
    # chromatic direction
    rotAngle_Wishart_MOCS = params_ell[r,:,-1]
    diff_rotAngle[r] = symmetric_angle_difference(rotAngle_Wishart_MOCS, 
                                                  rotAngle_chromaDir_MOCS)

# Retrieve the 2D Wishart → RGB transformation matrix (for use in color mapping, etc.)
color_thres_data = model_pred_Wishart_MOCS.color_thres_data
M_2DWToRGB =color_thres_data.M_2DWToRGB

#%% debug: check the correspondence between thres_est and rotAngle
# slc_idx = 0
# x_r = np.cos(np.deg2rad(rotAngle_chromaDir_MOCS))*thres_est_MOCS[slc_idx]
# y_r = np.sin(np.deg2rad(rotAngle_chromaDir_MOCS))*thres_est_MOCS[slc_idx]

# x_w = np.cos(np.deg2rad(rotAngle_Wishart_MOCS))*thres_est_Wishart[slc_idx]
# y_w = np.sin(np.deg2rad(rotAngle_Wishart_MOCS))*thres_est_Wishart[slc_idx]
# fig, ax = plt.subplots(1, 1, figsize=(2,2), dpi=1024)
# for i in range(nRefs):
#     ax.scatter(*xref_unique[slc_idx,i], marker = '+',
#                c = color_thres_data.W2D_to_rgb(xref_unique[0,i]))
#     ax.plot([xref_unique[slc_idx,i,0], x_r[i] + xref_unique[slc_idx,i,0]], 
#             [xref_unique[slc_idx,i,1], y_r[i] + xref_unique[slc_idx,i,1]],
#             c = color_thres_data.W2D_to_rgb(xref_unique[0,i])) 
#     ax.plot([xref_unique[slc_idx,i,0], x_w[i] + xref_unique[slc_idx,i,0]], 
#             [xref_unique[slc_idx,i,1], y_w[i] + xref_unique[slc_idx,i,1]], c = 'k')

#%% Step 3: Fit linear regressions and generate predictions for three explanatory variables
# Number of components (explanatory variables to analyze)
n_comparisons = 3
n_points_pred = 1000  # Number of points for predicted regression lines

# Initialize arrays to store regression results and predictions
stats_model = []
slopes = np.full(n_comparisons, np.nan)
intercepts = np.full(n_comparisons, np.nan)
R2_values = np.full(n_comparisons, np.nan)
p_values = np.full((n_comparisons, 2), np.nan)
std_errs = np.full((n_comparisons, 2), np.nan)
t_values = np.full((n_comparisons, 2), np.nan)
conf_int = np.full((n_comparisons, 2, 2), np.nan)

x_preds = np.full((n_comparisons, n_points_pred), np.nan)
y_preds = np.full((n_comparisons, n_points_pred), np.nan)
x_plot_ranges = np.full((n_comparisons, 2), np.nan)

# Explanatory variables: (1) angular difference, (2) aspect ratio, (3) validation threshold
x_vars = [diff_rotAngle, ratio_major_over_minor, thres_est_MOCS]

# Loop over each variable to fit linear regression and compute predictions
for s in range(n_comparisons):
    x_raw = x_vars[s].ravel()
    y_raw = diff_thres_est.ravel()

    # Define plotting range with ±5% buffer beyond data range
    x_min, x_max = np.min(x_raw), np.max(x_raw)
    x_plot_ranges[s] = np.array([x_min, x_max]) + 0.05 * (x_max - x_min) * np.array([-1, 1])
    
    #alternative package
    x_with_const = sm.add_constant(x_raw)
    model_s = sm.OLS(y_raw, x_with_const).fit()
    stats_model.append(model_s)
    print(model_s.summary())
    
    intercepts[s], slopes[s] = model_s.params
    R2_values[s] = model_s.rsquared
    p_values[s] = model_s.pvalues
    t_values[s] = model_s.tvalues
    conf_int[s] = model_s.conf_int()
    
    # Compute predicted y-values for a smooth regression line
    x_preds[s] = np.linspace(x_min, x_max, n_points_pred)
    y_preds[s] = slopes[s] * x_preds[s] + intercepts[s]
    
    print("... ...")

#degree of freedom
dof = int(model_s.nobs - 2)

#%% step 4: visualization
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
fig, ax = plt.subplots(3, 1, figsize=(12,10), dpi=1024)
for r in range(nDatasets):
    for j in range(nRefs):
        for s in range(n_comparisons):
            ax[s].plot([x_vars[s][r,j], x_vars[s][r,j]], 
                    [0, diff_thres_est[r,j]], 
                    c = color_thres_data.W2D_to_rgb(xref_unique[r,j]),
                    alpha = 0.2, lw = 2 if flag_data else 6)
            ax[s].scatter(x_vars[s][r,j], diff_thres_est[r,j], marker = 'o',
                          s =18 if flag_data else 38,
                          c = color_thres_data.W2D_to_rgb(xref_unique[r,j]),
                          alpha = 0.75)
            ax[s].plot(x_preds[s], y_preds[s], c = 'k', ls = '-', lw = 1)
for s in range(n_comparisons):
    y_ub = np.max(np.abs(diff_thres_est))*1.2
    #stats results
    ax[s].text(
        0.05, 0.05,
        f"Slope = {slopes[s]:.3f}, $p$ = {'< 0.001' if p_values[s,1] < 0.001 else f'{p_values[s,1]:.3f}'}, " + \
        f"$t({dof:d}) = {t_values[s,1]:.3f}$, $R^2$ = {R2_values[s]:.3f}",
        transform=ax[s].transAxes,
        verticalalignment='bottom'
    )
    ax[s].grid(True, alpha = 0.2)
    ax[s].set_ylim([-y_ub, y_ub])
    ax[s].set_xlim(x_plot_ranges[s])
    ax[s].set_ylabel('Threshold difference\n(WPPM − validation)')
ax[0].set_xlabel('Absolute difference of angles (major axis of WPPM-predicted ellipse - validation, deg)')
ax[0].set_xticks(np.linspace(0,90,7))
ax[1].set_xlabel('Ratio of major to minor axis lengths of WPPM-predicted ellipses')
ax[2].set_xlabel('Validation thresholds')

#save the figure
output_figDir_fits = input_fileDir_fits_r.replace('DataFiles', 'FigFiles')
output_figDir_fits = re.sub(r'sub\d+', 'groupData', output_figDir_fits)
str_cond = re.search(r"(decayRate.*?)(?=\.pkl$)", file_name_others).group(1)
fig_name = f'Residuals_btw_Validation_WPPMestimates_groupData_{str_cond}.pdf'
# Make sure the destination folder exists
os.makedirs(output_figDir_fits, exist_ok=True)
# Full file path
output_path = os.path.join(output_figDir_fits, fig_name)
fig.savefig(output_path, bbox_inches='tight')

#%% linear mixed effects
# import pandas as pd
# import statsmodels.formula.api as smf
# # Create a subject-ID vector, e.g. [0,0,…,0, 1,1,…,1, …, 7,7…7]
# subjects = np.repeat(np.arange(nDatasets), MOCS['nRefs'])

# # Assemble the DataFrame
# df = pd.DataFrame({
#     'participant': subjects,
#     'thres_diff': diff_thres_est.ravel(),
#     'angle_diff' : diff_rotAngle.ravel(),
#     'val_thres' : thres_est_MOCS.ravel(),
#     'aspect_ratio' : ratio_major_over_minor.ravel()
# })

# print(df.head())         # preview: first 5 rows
# print(df.shape)          # (200, 5)

# df['val_thres_c']   = df['val_thres'] - df['val_thres'].mean()
# df['angle_diff_c']  = df['angle_diff'] - df['angle_diff'].mean()
# df['aspect_ratio_c']= df['aspect_ratio'] - df['aspect_ratio'].mean()

# model = smf.mixedlm("thres_diff ~ angle_diff_c * val_thres_c * aspect_ratio_c",
#                     data=df,
#                     groups=df["participant"])
# result = model.fit()
# print(result.summary())