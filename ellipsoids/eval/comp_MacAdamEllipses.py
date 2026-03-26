#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:52:56 2025

@author: fangfang

This script compares the threshold ellipses derived from our color 
discrimination experiments with classic MacAdam ellipses. The workflow is 
divided into the following main steps:

1. Load Color Matching Functions and Monitor Primaries
    Load the 1931 color matching functions (T_xyz1931) and the monitor's spectral 
    primaries (P_device) to enable color space conversions.

2. Load and Generate MacAdam Ellipses
    Import ellipse parameters from MacAdam’s original dataset and generate their 
    2D contours in the CIE 1931 xy space.

3. Load Wishart Model Predictions
    Load the fitted Wishart model from our experimental data and compute model
    -predicted threshold ellipses on an N×N reference grid in Wishart space.

4. Convert Stimuli from Wishart Space to CIE xyY
    Convert both the reference stimuli and their threshold ellipses from Wishart 
    space to linear RGB, then to XYZ using the primaries, and finally to xyY 
    chromaticity space.

5. Visualize Ellipses in Chromaticity Space
    Overlay the model-predicted ellipses and MacAdam ellipses on the CIE 1931 
    chromaticity diagram to qualitatively assess their similarity in shape, size,
    and orientation.

"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dill as pickled
import numpy as np
from tqdm import trange
import re
import colour
from colour.plotting import plot_chromaticity_diagram_CIE1931
from colour.models import XYZ_to_xyY
import os
from analysis.ellipses_tools import rotAngle_to_eigenvectors
from core.model_predictions import rerun_model_pred_wExisting_model
from analysis.utils_load import load_util_files
from analysis.conf_interval import find_inner_outer_contours_nonellipse, \
    find_btst_dataset_within_CI

# Toggle between HPC batch mode and local/interactive mode.
flag_running_on_hpc = False

# Base directory where data lives. On HPC, prefer paths relative to the script.
base_dir = os.path.dirname(__file__) if flag_running_on_hpc else \
    '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
    
#%% 
#----------------------------------------------------------------------
# Section 1: Load MacAdam ellipses
#----------------------------------------------------------------------
# Directory containing utility files (e.g., MacAdam ellipse parameters)
ellipses_dir = os.path.join(base_dir, 'ELPS_materials', 'util_files')

# Name of the Excel file containing MacAdam ellipse data
file_macadam = 'MacAdam Ellipse Data.xlsx'

# Full path to the MacAdam ellipse data file
macadam_path = os.path.join(ellipses_dir, file_macadam)

# Scaling factor converting MacAdam axis units to chromaticity units
macadam_scaler = 1 / 100

# Number of points used to sample each ellipse contour
n_ellipse_pts = 200

# Load ellipse parameters and generate finely sampled ellipse contours
ellipses_fine, macadam_df, xc_array, yc_array, semi_major, semi_minor, angles_deg, \
    n_ellipses = load_util_files.load_MacAdam_ellipses(macadam_path)
    
x_gamut, y_gamut, T_xyz, P_device, M_RGBToXYZ, rgb_white, XYZ_white, xyY_white = \
    load_util_files.load_matchingFunc_monitorSPD(ellipses_dir, '02242025')
    
#%%
#----------------------------------------------------------------------
# Section 2: Load data and compute
#----------------------------------------------------------------------
subN = 11
nBtst = 120

# Construct path to the directory containing model fits
input_fileDir_fits = os.path.join(
    base_dir, 'ELPS_analysis', 'Experiment_DataFiles',
    'pilot2', f'sub{subN}', 'fits'
)

# Name of the pickle file containing the model fit to the original dataset
file_name = (
    f'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub{subN}_'
    'decayRate0.4_varScaler0.0003_nBasisDeg5.pkl'
)

# Extract decayRate from the filename (used to locate bootstrap fits)
match = re.search(r'decayRate([0-9.]+)', file_name)
decay_rate = float(match.group(1)) if match else None

# Visual scaling factor applied to predicted ellipses
# (used only for plotting so their size matches MacAdam ellipses)
W_scaler = 2

if flag_running_on_hpc:   
    # ------------------------------------------------------------
    # Loop through all bootstrap fits and the original dataset
    # ------------------------------------------------------------
    for i in range(nBtst+1):
        if i < nBtst:
            # Path to bootstrap fit
            file_name_i = f'{file_name[:-4]}_btst_AEPsych[{i}].pkl'
            full_path = os.path.join(
                input_fileDir_fits,
                'AEPsych_btst',
                f'decayRate{decay_rate}',
                file_name_i
            )
        else:
            # Original dataset fit
            full_path = os.path.join(input_fileDir_fits, file_name)
            
        # Load the pickled data file
        with open(full_path, 'rb') as f: 
            vars_dict = pickled.load(f)    
        
        # Extract model prediction, weight matrix, model instance, and color threshold data
        model_pred_Wishart = vars_dict['model_pred_Wishart']
        color_thres_data = vars_dict['color_thres_data']
        
        # Generate a custom grid to recompute model predictions
        # Although the file already contains predictions on a 5x5 grid,
        # we regenerate predictions here to maintain flexibility in choosing grid resolution
        grid_MacAdam = jnp.stack(jnp.meshgrid(*[jnp.linspace(-0.7, 0.7, 5) for _ in range(2)]), axis=-1)
        
        model_pred_Wishart_MacAdam, _ = \
            rerun_model_pred_wExisting_model(grid_MacAdam, model_pred_Wishart, color_thres_data)
        
        # Store the total number of reference locations used in the grid
        nRefs = np.prod(grid_MacAdam.shape[:2])
                
        # Add a third dimension (value=1) to 2D grid locations for affine transformation (W → RGB)
        # Resulting shape: (nRefs, 3)
        grid_W = np.reshape(grid_MacAdam, (nRefs, 2))
        
        # Apply scaling to the predicted ellipses to enlarge them visually.
        # Subtract the center, scale the deviation, and add back the center (broadcasted).
        # Shape: (num_grid_pts, num_grid_pts, 2, n_ellipse_pts)
        ellipses_thres_W_scaled = (model_pred_Wishart_MacAdam.fitEll_unscaled - \
                                   grid_MacAdam[..., None]) * W_scaler + \
                                    grid_MacAdam[..., None]
        

        # --------------------------------------------------------
        # Convert reference points from W space → xyY
        # --------------------------------------------------------
        
        # Step 1: W → RGB
        edge_RGB = color_thres_data.W2D_to_rgb([[-1,-1],  #4 vertices of the Wishart square
                                                [-1, 1],
                                                [ 1, 1],
                                                [ 1,-1]]).T
        stim_grid_RGB = color_thres_data.W2D_to_rgb(grid_W).T
        
        # Step 2: RGB → XYZ using monitor primaries
        XYZ_edge = M_RGBToXYZ @ edge_RGB
        XYZ_ref = M_RGBToXYZ @ stim_grid_RGB  # Shape: (3, nRefs)
        
        # Step 3: XYZ → xyY (transpose to shape: nRefs x 3)
        xyY_edge = XYZ_to_xyY(XYZ_edge.T)
        xyY_ref = XYZ_to_xyY(XYZ_ref.T)
        
        # --------------------------------------------------------
        # Convert predicted threshold ellipses
        # W space → RGB → XYZ → xyY
        # --------------------------------------------------------
        
        # Number of points per ellipse (typically 200)
        n_ellipse_pts_W = ellipses_thres_W_scaled.shape[-1]
        
        # Add third row of ones to each 2D ellipse point for affine RGB conversion
        # Shape after stacking: (nRefs, 3, n_ellipse_pts_W)
        ell_thres_W = np.concatenate(
            (np.reshape(ellipses_thres_W_scaled, (nRefs, 2, n_ellipse_pts_W)),
             np.ones((nRefs, 1, n_ellipse_pts_W))), axis=1)
        
        # Initialize array to hold threshold ellipses in xyY space
        ell_thres_xyY = np.full(ell_thres_W.shape, np.nan)
        
        # Convert each ellipse: W → RGB → XYZ → xyY
        for i in range(nRefs):
            ell_thres_RGB_i = color_thres_data.M_2DWToRGB @ ell_thres_W[i]   
            ell_thres_XYZ_i = M_RGBToXYZ @ ell_thres_RGB_i                
            ell_thres_xyY[i] = XYZ_to_xyY(ell_thres_XYZ_i.T).T        
            
        # --------------------------------------------------------
        # Store all relevant variables for later plotting
        # --------------------------------------------------------
        comp_MacAdam = {name: eval(name) for name in [
                'x_gamut', 'y_gamut', 'T_xyz', 'P_device', 'M_RGBToXYZ', 'rgb_white',
                'XYZ_white', 'xyY_white', 'macadam_df', 'macadam_scaler', 'xc_array',
                'yc_array', 'semi_major', 'semi_minor', 'angles_deg', 'n_ellipses',
                'n_ellipse_pts', 'nRefs', 'grid_W', 'W_scaler', 'ellipses_thres_W_scaled', 
                'edge_RGB', 'stim_grid_RGB', 'XYZ_edge', 'XYZ_ref', 'xyY_edge', 'xyY_ref',
                'n_ellipse_pts_W','ell_thres_W', 'ell_thres_xyY']}
    
        vars_dict['model_pred_MacAdam'] = model_pred_Wishart_MacAdam
        vars_dict['grid_MacAdam'] = grid_MacAdam
        vars_dict['comp_MacAdam'] = comp_MacAdam
    
        # Save the updated pickle file
        with open(full_path, 'wb') as f:
            pickled.dump(vars_dict, f)
else:
    # ------------------------------------------------------------
    # Load precomputed results (no recomputation needed)
    # ------------------------------------------------------------
    full_path = os.path.join(input_fileDir_fits, file_name)

    # Load the pickled data file
    with open(full_path, 'rb') as f: 
        vars_dict_load = pickled.load(f)   
    globals().update(vars_dict_load)
    globals().update(comp_MacAdam)
    ref_rgb = color_thres_data.W2D_to_rgb(grid_MacAdam.reshape(nRefs,-1))
        
    # we need to store NBS summed across a grid of references in order to determine
    # which bootstrapped dataset should be left out in the 95% confidence intervals
    NBS_sum = []
    ell_thres_xyY_btst_list = []
    for i in trange(nBtst):
        file_name_i =  f'{file_name[:-4]}_btst_AEPsych[{i}].pkl'
        full_path_i = os.path.join(input_fileDir_fits, 'AEPsych_btst', 
                                f'decayRate{decay_rate}', file_name_i)
        
        # Load the pickled data file
        with open(full_path_i, 'rb') as f: 
            vars_dict_load_i = pickled.load(f)    
       
        NBS_sum_i = np.sum(vars_dict_load_i['NBS_fine_grid'])
        NBS_sum.append(NBS_sum_i)
        
        ell_thres_xyY_btst_list.append(vars_dict_load_i['comp_MacAdam']['ell_thres_xyY'])
         
    # Keep the top 95% of bootstraps by NBS score
    ell_thres_xyY_btst_keep, *_ = find_btst_dataset_within_CI(NBS_sum, 
                                                              ell_thres_xyY_btst_list, 
                                                              CI_percent=0.95
                                                              )
    
    # ------------------------------------------------------------
    # Compute outer (union) and inner (intersection) contours
    # across bootstrap ellipses
    # ------------------------------------------------------------
    ell_min_xy = np.full((nRefs, 1000,2), np.nan)
    ell_max_xy = np.full(ell_min_xy.shape, np.nan)
    for j in range(nRefs):
        ell_thres_xyY_1ref_j = [E[j,:2].T for E in ell_thres_xyY_btst_keep]
        #note that central_fraction is set to 1 because the passed in array is
        #already 95% kept bootstrapped datasets
        xu_j, yu_j, xi_j, yi_j = find_inner_outer_contours_nonellipse(ell_thres_xyY_1ref_j, 
                                                                      central_fraction = 1) 
        ell_min_xy[j, :len(xi_j)] = np.vstack((xi_j, yi_j)).T
        ell_max_xy[j, :len(xu_j)] = np.vstack((xu_j, yu_j)).T
         
#%% load macAdams
if not flag_running_on_hpc:
            
    # Visualization of MacAdam ellipses or pilot data in CIE 1931 chromaticity diagram
    plt.rcParams.update({
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5,
        'axes.facecolor': 'white',  # Background inside plot
        'figure.facecolor': 'white',  # Background outside plot
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'font.family': 'Arial',
        'font.size': 15,
    })
    
    flag_save_fig = True     # If True, save the figure as a PDF

    # Note: To avoid clutter, we either plot pilot data or MacAdam ellipses.
    # Use the flags below to control what gets displayed/saved.
    flag_diagram_color = False
    flag_plot_MacAdam = True
    flag_plot_data = True  # If True, show pilot threshold ellipses
    flag_plot_major = True  # If True, show the major axis
    flag_show_CI = True
    
    # Plot the CIE 1931 chromaticity diagram (without rendering immediately)
    cie_fig, cie_ax = plot_chromaticity_diagram_CIE1931(standalone=False, 
                                                        show_diagram_colours= flag_diagram_color
                                                        )
    # Set figure size in inches (width, height)
    cie_fig.set_size_inches(5.6, 6)
    # Slightly grey background
    bg_color = 'white'#'#f5f5f5'  # or '#eeeeee' for slightly darker
    
    cie_fig.patch.set_facecolor(bg_color)  # Figure background
    cie_ax.set_facecolor(bg_color)         # Axes (plot area) background
    # Ensure the canvas is initialized
    cie_fig.canvas.draw()
    
    # Plot MacAdam ellipses if flag is off
    if flag_plot_MacAdam:
        for i in range(nRefs):
            # Mark center of each MacAdam ellipse
            cie_ax.scatter(xc_array[i], yc_array[i], marker='+', lw=1, c='k', s=20)
            # Plot the ellipse contour
            cie_ax.plot(ellipses_fine[i, 0], ellipses_fine[i, 1], c='k', lw=0.5)
    
    # Plot pilot threshold data if flag is on
    if flag_plot_data:
        for n in range(nRefs):
            # Mark the reference location
            if flag_diagram_color:
                cie_ax.scatter(*xyY_ref[n,:2], marker='x', c='k', s=10, lw=1)
            # Plot the ellipse contour at threshold
            cie_ax.plot(*ell_thres_xyY[n], color= ref_rgb[n] if not flag_diagram_color else 'k', lw=1)
    #plot the major axis
    if flag_plot_major:
        for n in range(nRefs):
            eigvec_n = rotAngle_to_eigenvectors(angles_deg[n]) @ np.array([[1,-1],[0, 0]])
            major_coord_n = eigvec_n * semi_major[n] #+ np.array([xc_array[n], yc_array[n]])
            cie_ax.plot(major_coord_n[0] +xc_array[n],
                        major_coord_n[1] +yc_array[n], color='k', lw=1)
            
    if flag_show_CI:
        for n in range(nRefs):
            cie_ax.fill(*ell_max_xy[n].T, color = ref_rgb[n], linewidth = 0, alpha = 0.7)
            cie_ax.fill(*ell_min_xy[n].T, color = 'white', linewidth = 0)
    
    # Draw monitor gamut triangle by looping through the RGB primaries
    x_loop = np.append(x_gamut, x_gamut[0])
    y_loop = np.append(y_gamut, y_gamut[0])
    cie_ax.plot(x_loop, y_loop, color='k', lw=0.5)
    
    # Draw the quadralateral gamut constrained by the isoluminant plane 
    x_loop_edge = np.append(xyY_edge[:,0], xyY_edge[0,0])
    y_loop_edge = np.append(xyY_edge[:,1], xyY_edge[0,1])
    cie_ax.plot(x_loop_edge, y_loop_edge, color = 'k', ls = '--', lw = 0.5)
    
    # Set axis limits and title
    cie_ax.set_xlim([0, 0.8])
    cie_ax.set_ylim([0, 0.9])
    cie_ax.set_title('')
    cie_ax.minorticks_on()  # Turn on minor ticks
    # Major grid lines
    cie_ax.grid(which='major', color='lightgrey', linestyle=':', linewidth=0.75)
    # Final rendering (uses internal colour-science render settings)
    colour.plotting.render(show=True)
    
    # Undo tight layout that is automatically applied by render()
    cie_fig.set_tight_layout(False)
    
    # Save the figure if enabled
    if flag_save_fig:
        # Replace 'DataFiles' with 'FigFiles' and everything after 'sub#' with 'MacAdam_ellipses'
        fileDir_prefix = input_fileDir_fits.split(f'sub{subN}')[0]
        fig_output_dir = os.path.join(fileDir_prefix.replace('DataFiles', 'FigFiles'), 
                                      f'sub{subN}', 'MacAdam_ellipses')
        os.makedirs(fig_output_dir, exist_ok=True)
        fig_full_path = os.path.join(fig_output_dir, f'MacAdam_Ellipses_CIE1931_sub{subN}'+\
                                     f'_{int(flag_diagram_color)}_{int(flag_plot_MacAdam)}'+\
                                     f'_{int(flag_plot_data)}_{int(flag_plot_major)}'+\
                                     f'_{int(flag_show_CI)}.pdf')
    
        # Save without tight bounding box
        cie_fig.savefig(fig_full_path, dpi=1024, bbox_inches=None)
    
    # Show the plot
    plt.show()

