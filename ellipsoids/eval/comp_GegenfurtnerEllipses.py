#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:18:02 2024

@author: fangfang

This script compares the threshold ellipses obtained from our color discrimination 
experiment with those reported in Krauskopf & Gegenfurtner (1992). The workflow 
is organized into the following steps:

1. Load transformation matrices and experimental data
    Load the matrices that convert between DKL space and the Wishart (W) space. 
    Also, load the experimental data containing fitted parameters from our color 
    discrimination experiment.

2. Compute the covariance matrix for the achromatic reference
    Using the fitted Wishart model, derive the covariance matrix corresponding 
    to the elliptical threshold at the achromatic reference location. Then, 
    transform this matrix from W space to DKL space.

3. Standardize the DKL space via axis stretching
    Construct a diagonal stretching matrix to normalize the ellipse in DKL space 
    such that both the x- and y-axis (not major or minor) are scaled to unit length. 
    This defines the “stretched DKL” space.

4. Sample stimulus locations on an enlarged circle
    Select a set of points uniformly distributed along a circle in the stretched 
    DKL space. The number of points and the radius of the circle are chosen to 
    match the configuration shown in the Krauskopf & Gegenfurtner (1992) study. 
    These points are then transformed from stretched DKL → DKL → W, where model 
    predictions can be evaluated.

5. Transform predicted ellipses into stretched DKL space
    Convert the model-predicted ellipses (computed in W space) into covariance 
    matrices in the stretched DKL space for comparison with published data.

6. Visualize and compare
    Generate visualizations to compare the threshold ellipses predicted by the 
    Wishart model with those reported in the Krauskopf & Gegenfurtner (1992) study.

"""
# Toggle between HPC batch mode and local/interactive mode.
flag_running_on_hpc = False
import jax
jax.config.update("jax_enable_x64", True)
import dill as pickled
import numpy as np
import re
from dataclasses import replace
import os
import matplotlib.pyplot as plt
from analysis.ellipses_tools import UnitCircleGenerate,convert_2Dcov_to_points_on_ellipse,\
    covMat_to_ellParamsQ, GegenfurtnerEll
from analysis.utils_load import select_file_and_get_path, extract_sub_number
from analysis.conf_interval import find_inner_outer_contours_for_gridRefs
from core.model_predictions import rerun_model_pred_wExisting_model
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.color_spaces_plotting import DKLRelatedSpacesVisualization, PlotGegenfurtner
from plotting.wishart_predictions_plotting import add_CI_ellipses

# Base directory where data lives. On HPC, prefer paths relative to the script.
base_dir = os.path.dirname(__file__) if flag_running_on_hpc else \
    '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
            
#%% step 1
# load transformation matrices and data
ndims = 2

# Replace everything after 'sub1' with 'Gegenfurtner_ellipses'
if flag_running_on_hpc:
    subN = 2
    input_fileDir_fits = os.path.join(base_dir, 'ELPS_analysis', 'Experiment_DataFiles',
                                      'pilot2', f'sub{subN}', 'fits')
    file_name_fits = f'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub{subN}'+\
        '_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl'
else:
    # Prompt user to select a fitted model file (pickled .pkl format)
    # Example path:
    # '/Volumes/T9/.../ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits/'
    # Example filename:
    # 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.5_nBasisDeg5.pkl'

    input_fileDir_fits, file_name_fits = select_file_and_get_path()
    # Extract subject number (e.g., 'sub1' → 1)
    subN = extract_sub_number(file_name_fits)
    
    #output figure dir
    output_figDir_fits = re.sub(r'(sub\d+)(/.*)?$', r'\1/Gegenfurtner_ellipses',
                                input_fileDir_fits).replace('DataFiles', 'FigFiles')
    os.makedirs(output_figDir_fits, exist_ok = True)

# Load the model fit data
full_path = os.path.join(input_fileDir_fits, file_name_fits)
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)
    
# Unpack data: model predictions, model structure, estimated weights, and raw stimuli
model_pred_Wishart = vars_dict['model_pred_Wishart']
color_thres_data = vars_dict['color_thres_data']
color_thres_data.base_path = base_dir #change directory 
expt_trial = vars_dict['aepsych_data']
grid_Wishart = vars_dict['grid']
grid_pts = grid_Wishart.shape[0]

# Extract relevant portions of the transformation matrices
# Note: the last row/column are placeholders for 3D coordinates; we truncate for 2D
M_2DWToDKL = color_thres_data.M_2DWToDKL[:2, :2]  # Wishart → DKL (2D only)
M_DKLTo2DW = color_thres_data.M_DKLTo2DW          # Full inverse transform
# Number of fine samples for threshold contours
nDir = 200  
    
#%%
#check if this is already computed before
key_pred = 'model_pred_Wishart_Gegenfurtner'
flag_append_data = True
if key_pred not in vars_dict:    
    #step 2
    # Select the center stimulus (corresponding to the gray stimulus)
    idx_row = model_pred_Wishart.num_grid_pts1 // 2  # Row index for the center
    idx_col = model_pred_Wishart.num_grid_pts2 // 2  # Column index for the center
    
    # Retrieve the ellipse parameters for the gray stimulus at the center of the grid
    # Parameters: x center, y center, major axis, minor axis, rotation angle
    ell_params_grey_W = model_pred_Wishart.params_ell[idx_row][idx_col]
    
    # Convert the ellipse parameters from W space to a covariance matrix in DKL space
    covMat_ell_grey_DKL = GegenfurtnerEll.convert_ellParamsW_to_covMatDKL(*ell_params_grey_W[2:],
                                                                          M_2DWToDKL)
    
    #step 3
    # Compute the stretching matrices for transforming between DKL space and unit space
    stretchingMat_DKL_to_sDKL, stretchingMat_sDKL_to_DKL = \
        GegenfurtnerEll.stretchingMat_from_covMatDKL(covMat_ell_grey_DKL)
    
    #step 4
    # Generate points on a scaled unit circle (to serve as reference)
    # Note that nPts_sDKL_circle and scaler are selected to best match the stimulus
    # condition in Krauskopf & Gegenfurtner (1992)
    nPts_sDKL_circle = 16
    scaler = 6.5
    sDKL_circle_pts = (np.eye(2) * scaler) @ UnitCircleGenerate(nPts_sDKL_circle+1)[:,:-1]
    
    # Transform sDKL circle points from sDKL space to DKL space
    ref_pts_DKL = stretchingMat_sDKL_to_DKL @ sDKL_circle_pts
    
    # Transform DKL points to W space using the inverse of the transformation matrix
    ref_pts_W = M_DKLTo2DW @ np.vstack((ref_pts_DKL, np.full((1, nPts_sDKL_circle), 1)))
    
    # Extract original W space coordinates for the reference points
    ref_pts_W_trunc_trans = ref_pts_W[:-1].T  # Shape: (nPts_sDKL_circle, 2)
    ref_pts_W_org = ref_pts_W_trunc_trans[np.newaxis]  # Add an extra dimension for batch processing
    
    model_pred_Wishart_Gegenfurtner, _= rerun_model_pred_wExisting_model(ref_pts_W_org, 
                                                                         model_pred_Wishart, 
                                                                         color_thres_data
                                                                         )
    #%step 5
    #-----------------------------------------------------------
    #Process the gray stimulus at the center
    # Convert the ellipse parameters to covariance matrices in DKL and stretched DKL spaces
    covMat_grey_DKL = GegenfurtnerEll.convert_ellParamsW_to_covMatDKL(*ell_params_grey_W[2:],
                                                                      M_2DWToDKL)
    covMat_grey_sDKL = GegenfurtnerEll.normalize_ellipse_axes(stretchingMat_DKL_to_sDKL,
                                                              covMat_ell_grey_DKL)
    
    # Reconstruct the threshold contours for the gray stimulus
    fine_ell_grey_DKL = convert_2Dcov_to_points_on_ellipse(covMat_grey_DKL)
    fine_ell_grey_sDKL = convert_2Dcov_to_points_on_ellipse(covMat_grey_sDKL)
    
    #------------------------------------------------------------
    #Repeat for all the stimuli around the grey stimulus
    
    # Initialize arrays for covariance matrices and threshold contours in DKL and sDKL space
    covMat_around_ref_DKL = np.full((nPts_sDKL_circle, ndims, ndims), np.nan)
    covMat_around_ref_sDKL = np.full(covMat_around_ref_DKL.shape, np.nan)
    fine_ell_W = np.full((nPts_sDKL_circle, ndims, nDir), np.nan)
    fine_ell_DKL = np.full(fine_ell_W.shape, np.nan)
    fine_ell_sDKL = np.full(fine_ell_W.shape, np.nan)
    
    # Iterate over each point on the sDKL circle
    for n in range(nPts_sDKL_circle):
        # Retrieve the threshold contour in W space for the sampled location
        fine_ell_W[n] = model_pred_Wishart_Gegenfurtner.fitEll_unscaled[0, n]
        
        # Retrieve the ellipse parameters for the sampled location in W space
        params_around_grey_n = model_pred_Wishart_Gegenfurtner.params_ell[0][n]
        
        # Convert the ellipse parameters to a covariance matrix in DKL space
        covMat_around_ref_DKL[n] = GegenfurtnerEll.convert_ellParamsW_to_covMatDKL(
            *params_around_grey_n[2:], M_2DWToDKL
        )
        
        # Convert the covariance matrix to stretched space
        covMat_around_ref_sDKL[n] = GegenfurtnerEll.normalize_ellipse_axes(
            stretchingMat_DKL_to_sDKL, covMat_around_ref_DKL[n]
        )
        
        # Reconstruct the threshold contours in DKL and sDKL spaces
        fine_ell_DKL[n] = convert_2Dcov_to_points_on_ellipse(covMat_around_ref_DKL[n])
        fine_ell_sDKL[n] = convert_2Dcov_to_points_on_ellipse(covMat_around_ref_sDKL[n])
        
    # put all the data to a dictionary
    comp_Gegenfurtner = {name: eval(name) for name in [
            'ell_params_grey_W', 'covMat_ell_grey_DKL','stretchingMat_DKL_to_sDKL',
            'stretchingMat_sDKL_to_DKL','nPts_sDKL_circle', 'scaler', 'sDKL_circle_pts',
            'ref_pts_DKL', 'ref_pts_W', 'covMat_grey_DKL','covMat_grey_sDKL', 
            'fine_ell_grey_DKL', 'fine_ell_grey_sDKL', 'covMat_around_ref_DKL', 
            'covMat_around_ref_sDKL', 'fine_ell_W', 'fine_ell_DKL', 'fine_ell_sDKL']}

    if flag_append_data:
        vars_dict[key_pred] = model_pred_Wishart_Gegenfurtner
        vars_dict['grid_Gegenfurtner'] = ref_pts_W_org
        vars_dict['comp_Gegenfurtner'] = comp_Gegenfurtner

        # Save the updated pickle file
        with open(full_path, 'wb') as f:
            pickled.dump(vars_dict, f)
else:
    # load previous calculations
    ref_pts_W_org = vars_dict['grid_Gegenfurtner']
    comp_Gegenfurtner = vars_dict['comp_Gegenfurtner']   
    for name, value in comp_Gegenfurtner.items():
        globals()[name] = value
    idx_row = model_pred_Wishart.num_grid_pts1 // 2  # Row index for the center
    idx_col = model_pred_Wishart.num_grid_pts2 // 2  # Column index for the center
    
#%% 
# ------------------------------------------------------------
# Locate one bootstrap-fit file; we will swap the AEPsych[n]
# index below to iterate through all bootstrap fits.
# ------------------------------------------------------------
if flag_running_on_hpc:
    input_fileDir_btst = os.path.join(
        base_dir, 'ELPS_analysis', 'Experiment_DataFiles',
        'pilot2', f'sub{subN}', 'fits', 'AEPsych_btst', 'decayRate0.4'
    )

    # Template filename: we will replace the integer inside AEPsych[...]
    file_name_btst = (
        f'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub{subN}_'
        'decayRate0.4_varScaler0.0003_nBasisDeg5_btst_AEPsych[0].pkl'
    )
else:
    # Local: user selects any one bootstrap-fit .pkl as the template;
    # we will then load AEPsych[0..nBtst-1] by string substitution.
    input_fileDir_btst, file_name_btst = select_file_and_get_path()
    
# ------------------------------------------------------------
# Load all bootstrap fits and compute a scalar "fit quality"
# score (sum of NBS similarities on the fine grid). We then keep
# the top 95% of bootstraps as a confidence-interval set.
# ------------------------------------------------------------
nBtst = 120

# Scalar score per bootstrap: sum of normalized Bures–Wasserstein similarities
# across the fine grid (higher = more similar / better fit in this convention).
NBS_sum = []

# Cached model predictions at the original model grid refs (one per bootstrap)
model_pred_Wishart_btst = []

# Cached model predictions at the Gegenfurtner ref set (one per bootstrap)
# We either load from the pickle (if already stored) or rerun and optionally
# append back into the pickle to avoid recomputation next time.
model_pred_Wishart_Gegenfurtner_btst = []

for n in range(nBtst):
    # Swap in the bootstrap index n: AEPsych[0] -> AEPsych[n]
    file_name_btst_n = re.sub(r'AEPsych\[\d+\]', f'AEPsych[{n}]', file_name_btst)

    # Load the bootstrap fit
    full_path_btst = os.path.join(input_fileDir_btst, file_name_btst_n)
    with open(full_path_btst, 'rb') as f:
        vars_dict_btst = pickled.load(f)

    # Compute and store the fit-quality scalar for this bootstrap
    # (sum of normalized Bures–Wasserstein similarities on the fine grid)
    NBS_sum.append(np.sum(vars_dict_btst['NBS_fine_grid']))

    # Store the fitted Wishart model prediction object for this bootstrap
    model_pred_Wishart_btst_n = vars_dict_btst['model_pred_Wishart']
    model_pred_Wishart_btst.append(model_pred_Wishart_btst_n)

    print(f'Bootstrap {n}: NBS = {NBS_sum[-1]:.2f}')

    # Ensure predictions at the Gegenfurtner reference set exist:
    # - If missing, rerun prediction using the already-fitted model
    # - Optionally write results back into the pickle for caching
    if key_pred not in vars_dict_btst:
        model_pred_Wishart_Gegenfurtner_btst_n, _ = rerun_model_pred_wExisting_model(
            ref_pts_W_org,
            model_pred_Wishart_btst_n,
            color_thres_data
        )
        model_pred_Wishart_Gegenfurtner_btst.append(model_pred_Wishart_Gegenfurtner_btst_n)

        if flag_append_data:
            vars_dict_btst[key_pred] = model_pred_Wishart_Gegenfurtner_btst_n
            vars_dict_btst['grid_Gegenfurtner'] = ref_pts_W_org

            # Overwrite the pickle with the cached predictions included
            with open(full_path_btst, 'wb') as f:
                pickled.dump(vars_dict_btst, f)
    else:
        # Fast path: use cached Gegenfurtner predictions from the pickle
        model_pred_Wishart_Gegenfurtner_btst.append(vars_dict_btst[key_pred])

# ------------------------------------------------------------
# Select the top 95% of bootstraps by NBS_sum (descending).
# idx_CI holds the bootstrap indices kept.
# ------------------------------------------------------------
idx_descending = np.argsort(np.asarray(NBS_sum))[::-1]
idx_cutoff = int(nBtst * 0.95)
idx_CI = idx_descending[:idx_cutoff]

# Subset both prediction lists to the selected bootstrap indices
model_pred_Wishart_Gegenfurtner_btst_CI = [
    model_pred_Wishart_Gegenfurtner_btst[i] for i in idx_CI
]
model_pred_Wishart_btst_CI = [model_pred_Wishart_btst[i] for i in idx_CI
]

#%%
fine_ell_sDKL_grid = np.full(grid_Wishart.shape + (nDir,), np.nan)
grid_Wishart_reshape = np.reshape(grid_Wishart, (grid_pts**ndims,2))
grid_sDKL_temp = stretchingMat_DKL_to_sDKL @ M_2DWToDKL @ grid_Wishart_reshape.T
grid_sDKL = np.transpose(np.reshape(grid_sDKL_temp, (ndims, grid_pts, grid_pts)), (1,2,0))
# Iterate over each point on the sDKL circle
for n in range(grid_pts):
    for m in range(grid_pts):        
        # Retrieve the ellipse parameters for the sampled location in W space
        params_around_grey_nm = model_pred_Wishart.params_ell[n][m]
        
        # Convert the ellipse parameters to a covariance matrix in DKL space
        covMat_around_ref_DKL_nm = GegenfurtnerEll.convert_ellParamsW_to_covMatDKL(
            *params_around_grey_nm[2:], M_2DWToDKL
        )
        
        # Convert the covariance matrix to stretched space
        covMat_around_ref_sDKL_nm = GegenfurtnerEll.normalize_ellipse_axes(
            stretchingMat_DKL_to_sDKL, covMat_around_ref_DKL_nm
        )
        
        # Reconstruct the threshold contours in DKL and sDKL spaces
        fine_ell_sDKL_grid[n,m] = convert_2Dcov_to_points_on_ellipse(covMat_around_ref_sDKL_nm)
        
#%%
# ---------------------------------------------------------------------
# Extract model-predicted threshold ellipse parameters at Gegenfurtner's
# reference locations, in both:
#   (1) W-space (model space), and
#   (2) stretched DKL space (sDKL) after transforming the ellipse covariance.
#
# We store ellipse parameters in the 5-parameter form:
#   [x0, y0, a, b, theta]
# for each reference, for each kept bootstrap model.
#
# We include an extra "reference" at the end: the central gray location,
# so total refs = nPts_sDKL_circle + 1.
# ---------------------------------------------------------------------

# Preallocate ellipse params:
#   (nRefs_total, nBoot_kept, 5)
# where nRefs_total = nPts_sDKL_circle + 1 (circle refs + center gray)
ellParams_W_btst    = np.full((nPts_sDKL_circle + 1, idx_cutoff, 5), np.nan)
ellParams_sDKL_btst = np.full_like(ellParams_W_btst, np.nan)

for n in range(idx_cutoff):
    # Gegenfurtner reference set (all refs on the sDKL circle), stored in model predictions
    # NOTE: params_ell[0] is expected to contain one ellipse per reference.
    ellParams_W_btst[:-1, n, :] = np.asarray(
        model_pred_Wishart_Gegenfurtner_btst_CI[n].params_ell[0]
    )

    # Add the central gray reference ellipse (from the original model-grid predictions)
    ellParams_W_btst[-1, n, :] = np.asarray(
        model_pred_Wishart_btst_CI[n].params_ell[idx_row][idx_col]
    )

    # Convert each ellipse from W-space -> DKL covariance -> sDKL
    #     and then back into ellipse parameters in sDKL coordinates
    for m in range(nPts_sDKL_circle + 1):
        # Convert W-ellipse params (a, b, theta) into a covariance matrix in DKL space
        covMat_DKL_nm = GegenfurtnerEll.convert_ellParamsW_to_covMatDKL(
            *ellParams_W_btst[m, n, 2:],  # (a, b, theta) in W space
            M_2DWToDKL
        )

        # Apply stretching/normalization to express the ellipse in stretched DKL (sDKL)
        covMat_sDKL_nm = GegenfurtnerEll.normalize_ellipse_axes(
            stretchingMat_DKL_to_sDKL,
            covMat_DKL_nm
        )

        # Convert covariance matrix back to ellipse parameters (a, b, theta) in sDKL
        _, _, ab, theta = covMat_to_ellParamsQ(covMat_sDKL_nm)

        # Store final ellipse params in sDKL coordinates:
        if m == nPts_sDKL_circle:
            ellParams_sDKL_btst[m, n] = [0, 0, *ab, theta]
        else:
            ellParams_sDKL_btst[m, n] = [*sDKL_circle_pts[:, m], *ab, theta]

# Compute inner/outer envelopes across bootstraps (confidence interval)
ell_min_W,    ell_max_W    = find_inner_outer_contours_for_gridRefs(ellParams_W_btst)
ell_min_sDKL, ell_max_sDKL = find_inner_outer_contours_for_gridRefs(ellParams_sDKL_btst)
    
#%% retrieve the model predictions at a grid of reference in the model space
# and then transform it to stretched DKL space
if not flag_running_on_hpc:            
    # Create a base plotting settings instance with output directory and font size
    pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize=13)
    
    # Initialize the visualization object for plotting ellipses in different color spaces
    # This object holds model predictions, comparison data, and color transformation utilities
    Gegenfurtner_vis = DKLRelatedSpacesVisualization(model_pred_Wishart, 
                                                     comp_Gegenfurtner,
                                                     color_thres_data, 
                                                     pltSettings_base, 
                                                     save_fig=True
                                                     )
    
    # Generate the full 2x3 comparison figure with five subplots:
    # (1) local Wishart space ellipses,
    # (2) zoomed-out Wishart space with full grid,
    # (3) standard DKL space,
    # (4) stretched DKL (sDKL) with normalized central ellipse,
    # (5) zoomed-out sDKL with full grid.
    plt_st = replace(PlotGegenfurtner(), **pltSettings_base.__dict__)
    plt_st = replace(plt_st, fig_name = f"Comp_GegenfurtnerEllipses_sub{subN}.pdf")
    Gegenfurtner_vis.plot_Gegenfurtner_comparison(grid_sDKL,
                                                  grid_Wishart, 
                                                  fine_ell_sDKL_grid, 
                                                  settings=plt_st,
                                                  )
    
    #-------------------------------------------------------------------------
    # individual plots
    #-------------------------------------------------------------------------
    #add confidence interval
    fig1, ax1 = plt.subplots(1, 1, figsize= (3.75, 3.75), dpi=plt_st.dpi)
    #plot the confidence interval
    for n in range(nPts_sDKL_circle+1):
        if n == 0: cm_n='gray'
        else: cm_n = color_thres_data.W2D_to_rgb(ref_pts_W_org[0,n-1])
        add_CI_ellipses(ell_min_W[n], ell_max_W[n], cm = cm_n, alpha = 0.4, ax = ax1)

    _,_, box_ub = Gegenfurtner_vis.plot_Gegenfurtner_Wishart_space(plt_st, ax = ax1)
    fig1.savefig(os.path.join(output_figDir_fits, f"Comp_GegenfurtnerEllipses_sub{subN}_W.pdf"), 
                format='pdf', bbox_inches='tight')
    
    fig2, ax2 = plt.subplots(1, 1, figsize= (3.75, 3.75), dpi=plt_st.dpi)
    #plot the confidence interval
    for n in range(nPts_sDKL_circle+1):
        if n == 0: cm_n='gray'
        else: cm_n = color_thres_data.W2D_to_rgb(ref_pts_W_org[0,n-1])
        add_CI_ellipses(ell_min_sDKL[n], ell_max_sDKL[n], cm = cm_n, alpha = 0.4, ax = ax2)
        
    Gegenfurtner_vis.plot_sDKL_space(plt_st, ax = ax2)
    fig2.savefig(os.path.join(output_figDir_fits, f"Comp_GegenfurtnerEllipses_sub{subN}_sDKL.pdf"), 
                format='pdf', bbox_inches='tight')
    
    Gegenfurtner_vis.plot_DKL_space(plt_st)
    Gegenfurtner_vis.plot_Gegenfurtner_Wishart_space_zoomed_out(box_ub, plt_st)
    Gegenfurtner_vis.plot_sDKL_zoomed_out(grid_sDKL, fine_ell_sDKL_grid, plt_st)
    

