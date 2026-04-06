#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:45:48 2024

@author: fangfang

Context
-------
We aim to fully characterize human color discrimination thresholds using
adaptively sampled trials and post-hoc fitting with the Wishart Process
Psychophysical Model (WPPM). To evaluate the accuracy of the WPPM predictions,
we interleaved a set of validation trials throughout the experiment.

Specifically, the validation dataset consists of 6,000 trials organized into
25 reference conditions. For each condition, 12 comparison levels were tested,
with each level repeated 20 times.

Purpose of this script
----------------------
The goal of this script is to fit a psychometric function to each validation
condition, bootstrap the validation data, refit the psychometric function for
each bootstrap replicate, and compute a 95% confidence interval for the
behaviorally estimated threshold. Using these estimates, we assess how
empirically derived thresholds align with WPPM-predicted thresholds at the same
validation conditions.

Scope and limitations
---------------------
This script provides an initial evaluation of the agreement between validation-
based threshold estimates and WPPM predictions. It does not compute confidence
intervals for the WPPM-predicted thresholds themselves. For analyses involving
95% confidence intervals on WPPM predictions, see
`Visualize_btstCI_Wishart_atMOCS.py`.

For this reason, all the stats (slope and corr coef) computed in this script
are not saved in the output .pkl file since they only provide an initial 
assessment of the agreement. Those stats will be computed again in 
`Visualize_btstCI_Wishart_atMOCS.py` once we have WPPM-predicted thresholds.

"""

import jax
jax.config.update("jax_enable_x64", True)
import os
import dill as pickled
import numpy as np
import re
from tqdm import tqdm
from dataclasses import replace
from tqdm import trange
from copy import deepcopy
from core.model_predictions import rerun_model_pred_wExisting_model
from dconfig.config_4Ddata import DatasetConfig_4D_MOCS
from dconfig.config_6Ddata import DatasetConfig_6D
from analysis.color_thres import color_thresholds
from analysis.MOCS_thresholds import fit_PMF_MOCS_trials, sim_MOCS_trials
from plotting.visualize_MOCS import MOCSTrialsVisualization, PlotPMFSettings,\
    PlotThresCompSettings, PlotThresCompSettings_bds, PlotThresComp3DSettings_bds
from plotting.wishart_plotting import PlotSettingsBase 

# %%
# -----------------------------------------------------------
# SECTION 1: Load data
# -----------------------------------------------------------
# this is the dir where we store our .db files
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

#change the knob here
subN = 1

# choose one dataset
dcfg = DatasetConfig_6D.human_fullcube(base_dir, subN)
# dcfg = DatasetConfig_4D_MOCS.human_isoluminant(base_dir, subN)
# dcfg = DatasetConfig_4D_MOCS.simulated_isoluminant(base_dir)

#print out summary
dcfg.print_summary()

#color thres object
color_thres_data = color_thresholds(dcfg.stim_dims,
                                    base_dir, 
                                    plane_2D = dcfg.plane_2D,
                                    )
if dcfg.stim_dims == 2: 
    color_thres_data.load_transformation_matrix(file_date = dcfg.file_date)

# Load the Wishart fit and, when needed, the separate simulated MOCS dataset.
#
# 6D human data stores the MOCS summaries inside the fitted-model pickle, so we
# only need the Wishart fit path. In contrast, simulated 4D MOCS analyses use
# one pickle for the simulated MOCS trials and another for the fitted Wishart
# model.
fits_path = dcfg.wishart_dir
file_name = dcfg.wishart_file_name
MOCS_full_path = dcfg.mocs_full_path
Wishart_full_path = dcfg.wishart_full_path
        
# Output directories for figures and processed data
decayRate = float(re.search(r"decayRate([0-9.]+)", file_name).group(1))
output_figDir_temp = os.path.join(fits_path.replace('DataFiles', 'FigFiles'))
output_figDir = os.path.join(os.path.dirname(output_figDir_temp),
                             f'decayRate{decayRate}') 
os.makedirs(output_figDir, exist_ok= True)   

# %%
# -----------------------------------------------------------
# SECTION 2: Load and organize the pilot data 
# -----------------------------------------------------------
# Load the necessary variables from the file
mocs_data = dcfg.load_mocs_data()
xref_unique = mocs_data['xref_unique']
refStimulus = mocs_data['refStimulus']
compStimulus = mocs_data['compStimulus']
responses = mocs_data['responses']
nRefs = mocs_data['nRefs']
nLevels = mocs_data['nLevels']
nTrials = mocs_data['nTrials']
        
# %%
# ==-----------------------------------------------------------
# SECTION 3: Analyze MOCS trials (fit PMF and bootstrap)
# -------------------------------------------------------------
# number of bootstrap
numBtst = 120
#initialize this list
fit_PMF_MOCS = []
#loop through each reference stimulus
for n in tqdm(range(nRefs), desc="Bootstrapping Progress"):
    #subtract the ref stimulus to recenter the comparison stimulus to the origin
    xy_coords_n = compStimulus[n] - refStimulus[n]
    #append origin 
    #we need to add a filler trial so that the PMF can be fixed at 1/3 (chance 
    #performance) when comp = ref
    xy_coords_n_wOrigin = np.vstack((xy_coords_n, np.array([0]*dcfg.stim_dims)))
    responses_n_wOrigin = np.append(responses[n], np.array([1/3]))
    # call the class
    fit_PMF_MOCS_exptN = fit_PMF_MOCS_trials(dcfg.stim_dims, 
                                             xy_coords_n_wOrigin, 
                                             responses_n_wOrigin,
                                             nLevels + 1, #+1 because we stick 0 in there
                                             dist_metric= 'Euclidean',
                                             bounds = [(1e-4, 0.5), (1e-2, 8)]
                                             ) 
    #fit a psychometric function
    fit_PMF_MOCS_exptN.fit_PsychometricFunc_toData()
    #given the best-fit parameters, reconstruct the PMF with finer grid points
    fit_PMF_MOCS_exptN.reconstruct_PsychometricFunc_givenData()
    #find the stimulus that corresponds to the threshold (defined as 66.7%)
    fit_PMF_MOCS_exptN.find_stim_at_targetPC_givenData()
    #bootstrap the data and refit
    fit_PMF_MOCS_exptN.bootstrap_and_refit(nBtst = numBtst,
                                           seed = subN * 100 + n if isinstance(subN, int) else 1)
    #find the 95% confidence interval on the reconstructed PMF as well as the 
    #stimulus that corresponds to the threshold
    fit_PMF_MOCS_exptN.compute_95btstCI()
    #append the fits
    fit_PMF_MOCS.append(fit_PMF_MOCS_exptN)
        
#%%
# -------------------------------------------------------------
# SECTION 4: Derive thresholds predicted by Wishart / Weibull
# -------------------------------------------------------------
# fitting file    
with open(Wishart_full_path, 'rb') as f: 
    data_load = pickled.load(f)

#gt_Wishart = data_load['gt_Wishart']
try:
    model_pred_Wishart = deepcopy(data_load['model_pred_Wishart'])
    model = deepcopy(model_pred_Wishart.model)
except:
    model_pred_Wishart = deepcopy(data_load['model_pred_Wishart_grid_isoluminant'])
    model = deepcopy(model_pred_Wishart.model)
W_est = model_pred_Wishart.W_est

#initialize
pChoosingX1_Wishart          = np.full((nRefs, fit_PMF_MOCS_exptN.nGridPts), np.nan)
vecLen_at_targetPC_Wishart   = np.full((nRefs,), np.nan)
vecLen_at_targetPC_MOCS      = np.full((nRefs,), np.nan)
vecLen_at_targetPC_MOCS_btst = np.full((nRefs, numBtst), np.nan)
stim_at_targetPC_MOCS        = np.full((nRefs, dcfg.stim_dims), np.nan)
stim_at_targetPC_Wishart     = np.full((nRefs, dcfg.stim_dims), np.nan)

for n in trange(nRefs):
    # Compute the Euclidean distance of each point from the origin
    # Get indices that sort the distances in descending order
    sorted_indices = np.argsort(-np.linalg.norm(fit_PMF_MOCS[n].unique_stim, axis=1))
    
    # Use the indices to sort the array
    sorted_array = fit_PMF_MOCS[n].unique_stim[sorted_indices]
    
    # Generate a finely sampled set of stimulus points along the chromatic direction
    finer_stim = sim_MOCS_trials.create_discrete_stim(sorted_array[0], 
                                                      fit_PMF_MOCS[n].nGridPts,
                                                      ndims= dcfg.stim_dims
                                                      )

    # Compute the probability of choosing x1 as the odd stimulus for each stimulus pair
    pChoosingX1_Wishart[n] = model_pred_Wishart._compute_pChoosingX1(
        np.full(finer_stim.shape, 0) + xref_unique[n], 
        finer_stim + xref_unique[n], 
        )
    
    #find the vector length that corresponds to 66.7% correct response
    #either based on the Wishart model or Weibull psychometric functions
    #NOTE: this is not the same as stimulus; we basically subtract the ref location from all the comp locations for this calculation
    vecLen_at_targetPC_Wishart[n] = fit_PMF_MOCS[n]._find_stim_at_targetPC(pChoosingX1_Wishart[n]) 
    vecLen_at_targetPC_MOCS_btst[n] = fit_PMF_MOCS[n].stim_at_targetPC_btst
    
    #find the exact stimulus location (x,y coordinates)
    vecLen_at_targetPC_MOCS[n] = fit_PMF_MOCS[n]._find_stim_at_targetPC(fit_PMF_MOCS[n].fine_pC)
    
    #the line below looks messy, but the idea is simple: retrieve the vector,
    #apply the vector length that corresponds to the threshold
    vec_midlevel = fit_PMF_MOCS[n].unique_stim[nLevels//2]
    unit_vec = vec_midlevel / np.linalg.norm(vec_midlevel)
    stim_at_targetPC_MOCS[n] = vecLen_at_targetPC_MOCS[n] * unit_vec + xref_unique[n]
    stim_at_targetPC_Wishart[n] = vecLen_at_targetPC_Wishart[n] * unit_vec + xref_unique[n]
    
#%%
# ---------------------------------------------------------------------------
# SECTION 5: Threshold contours for all MOCS conditions predicted by Wishart
# ---------------------------------------------------------------------------
# Compute the covariance matrices ('Sigmas') at each point in the grid using 
# the model's compute_U function. 
xref_unique_ext = xref_unique[(None,) * (dcfg.stim_dims - 1)]
Sigmas_est_xref_unique = model.compute_Sigmas(model.compute_U(W_est, xref_unique_ext))

# Initialize the Wishart model prediction using various parameters.
model_pred_Wishart_MOCS, _ = rerun_model_pred_wExisting_model(xref_unique_ext,
                                                              model_pred_Wishart,
                                                              color_thres_data
                                                              )

#%%
# ---------------------------------------------------------------------------
# SECTION 6: Compute some summary stats
# ---------------------------------------------------------------------------
# fit a linear regression to the original dataset
slope_modelPred_org, *_ = np.linalg.lstsq(vecLen_at_targetPC_MOCS[:,None],
                                          vecLen_at_targetPC_Wishart, 
                                          rcond = None
                                          )
corr_coef_modelPred_org = np.corrcoef(vecLen_at_targetPC_MOCS,  
                                      vecLen_at_targetPC_Wishart)[0,1]

# Initialize arrays to store bootstrap results for slope and correlation coefficient
slope_modelPred_btst = np.full((numBtst,), np.nan)
corr_coef_modelPred_btst = np.full((numBtst,), np.nan)

# Bootstrap loop
for n in range(numBtst):
    # Extract the n-th bootstrap sample from the empirical data (MOCS trials)
    data_emp_n = vecLen_at_targetPC_MOCS_btst[:,n]
    # Reshape the empirical data to a column vector for regression
    data_emp_n_reshape = data_emp_n.reshape(-1, 1)
    
    # Perform linear regression (no intercept) to compute the slope
    slope_modelPred_btst_n, *_ = np.linalg.lstsq(data_emp_n_reshape,
                                                 vecLen_at_targetPC_Wishart, 
                                                 rcond=None
                                                 )
    slope_modelPred_btst[n] = slope_modelPred_btst_n[0]
    
    # Compute the Pearson correlation coefficient between empirical and predicted data
    corr_coef_modelPred_btst[n] = np.corrcoef(data_emp_n,
                                              vecLen_at_targetPC_Wishart)[0,1]

# Compute 95% confidence intervals for slope and correlation coefficient
idx_95CI = np.array([int(numBtst*0.025), int(numBtst*0.975) - 1]) # Upper bound index (convert to 0-based indexing)

# Slope analysis
slope_modelPred_btst_sorted = np.sort(slope_modelPred_btst)
slope_modelPred_btst_CI = slope_modelPred_btst_sorted[idx_95CI]

# Correlation coefficient analysis
corr_coef_modelPred_btst_sorted = np.sort(corr_coef_modelPred_btst)
corr_coef_modelPred_btst_CI = corr_coef_modelPred_btst_sorted[idx_95CI]

#%%
# ------------------------------------------------------------
# SECTION 7: Visualize PMF and thresholds
# ------------------------------------------------------------
# Create a base plotting settings instance (shared across plots)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize=9.5)
# Initialize 2D prediction settings based on the base, with method-specific overrides
predPMF_settings = replace(PlotPMFSettings(), **pltSettings_base.__dict__)

#visualization object
vis_MOCS = MOCSTrialsVisualization(fit_PMF_MOCS, 
                                   settings = pltSettings_base, 
                                   save_fig= False
                                   )
#initialize color map
cmap_allref = []
for n in range(nRefs):    
    fig_name_part1 = file_name[:-4]
    fig_name_n = f"{fig_name_part1}Ref{n}_Wdim1_{np.round(xref_unique[n][0],2)}_"+\
        f"Wdim2_{np.round(xref_unique[n][1],2)}.pdf"
        
    #define color map for each reference
    if dcfg.stim_dims == 2:
        cmap_n = color_thres_data.W2D_to_rgb(xref_unique[n])
    else:
        cmap_n = color_thres_data.W_unit_to_N_unit(xref_unique[n])
        
    predPMF_settings = replace(predPMF_settings,
                               filler_pts = [0,1/3],
                               cmap_PMF = 'k',
                               cmap_dots = 'k',
                               CI_area_alpha = 0.2,
                               CI_thres_errorbar_lw= 5,
                               Wishart_pred_lw = 1,
                               Wishart_pred_lc = cmap_n,
                               fig_name = fig_name_n
                               )
    #append colormap so we can reuse it for the next plot
    cmap_allref.append(cmap_n)
    vis_MOCS.plot_PMF(slc_idx=n, 
                      pX1_Wishart_slc=pChoosingX1_Wishart[n], 
                      xref = xref_unique[n],
                      settings = predPMF_settings
                      )
    
#%% 
# -----------------------------------------------------------------------------
# SECTION 8: Visualize the comparison of thresholds between WPPM predictions 
# and MOCS predictions
# -----------------------------------------------------------------------------
# NOTE that this is just a quick comparison. So far we have not computed the 
# error bar for the Wishart threshold predictions. 
# we will replot this in visualize_btstCI_Wishart_atMOCS.py
settings_lookup = PlotThresComp3DSettings_bds if dcfg.stim_dims == 3 else PlotThresCompSettings_bds
settings_bds = settings_lookup[f"sub{subN if dcfg.flag_load_datafile else None}"]
plt_bds = settings_bds["bds"]
corr_txt_loc = settings_bds["corr_text_loc"]
slope_txt_loc = settings_bds["slope_text_loc"]

predComp_settings = replace(PlotThresCompSettings(), **pltSettings_base.__dict__)
predComp_settings = replace(predComp_settings,
                            fontsize = 9.5,
                            ms = 6,
                            fig_size = (4.8, 5), 
                            alpha = 0.8,
                            lw = 1.5,
                            bds = plt_bds, 
                            corr_text_loc = corr_txt_loc,
                            slope_text_loc = slope_txt_loc,
                            cmap = cmap_allref,
                            fig_name = f"{fig_name_part1}_comparison_btw_MOCS"+\
                                        "_WishartPredictions.pdf"
                            )

vis_MOCS.plot_comparison_thres(thres_Wishart = vecLen_at_targetPC_Wishart,
                               slope_org = slope_modelPred_org.item(),
                               slope_CI= slope_modelPred_btst_CI,
                               xref_unique = xref_unique,
                               corr_coef_org = corr_coef_modelPred_org,
                               corr_coef_CI = corr_coef_modelPred_btst_CI,
                               settings = predComp_settings
                               )

#%% save data
try:
    str_ext = file_name.split('sub{subN}')[1].split('.pkl')[0]
except:
    str_ext = ''
str_end = re.search(r'(decayRate.*)\.pkl', file_name).group(1)
str_stim = 'RGB cube' if dcfg.plane_2D is None else dcfg.plane_2D
output_file = f'Fitted_weibull_psychometric_func_{str_stim}_{nTrials}totalTrials'+\
    f'_{nRefs}refs_MOCS_sub{subN}{str_ext}_{str_end}.pkl'
full_path = os.path.join(fits_path, output_file)

variable_names = ['subN','dcfg', 'color_thres_data', 'xref_unique','refStimulus',
                  'compStimulus','responses', 'nRefs', 'nLevels', 'nTrials',
                  'numBtst', 'fit_PMF_MOCS', 'model_pred_Wishart', 
                  'pChoosingX1_Wishart', 'vecLen_at_targetPC_Wishart', 
                  'vecLen_at_targetPC_MOCS', 'vecLen_at_targetPC_MOCS_btst',
                  'stim_at_targetPC_MOCS', 'stim_at_targetPC_Wishart',
                  'Sigmas_est_xref_unique', 'model_pred_Wishart_MOCS']
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
