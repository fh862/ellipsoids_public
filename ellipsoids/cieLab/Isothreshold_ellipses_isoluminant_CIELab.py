#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 22:29:40 2025

@author: fangfang

The goal of this script is to compute threshold contours on the isoluminant plane
based on either CIE1976 (CIELAB), CIE1994 or CIE2000

The logic:
1. **Two spaces** are involved: RGB space (3D) and W-space (2D square).
2. **Threshold points** are computed in RGB space (where CIELab operates).
3. **Transformations** are applied:
   - RGB → W using a homography matrix.
4. **Fit an ellipse** in the Wishart space.

"""

import os
from analysis.utils_load import get_path
from scipy.io import loadmat
import numpy as np
import colour
import matplotlib.pyplot as plt
from dataclasses import replace
import dill as pickled
from analysis.simulations_CIELab import SimThresCIELab
from analysis.color_thres import color_thresholds
from analysis.ellipses_tools import fit_2d_isothreshold_contour, UnitCircleGenerate
from plotting.wishart_plotting import PlotSettingsBase, PlottingTools
from plotting.sim_CIELab_plotting import CIELabVisualization, Plot2DSinglePlaneSettings
base_dir = get_path("dropbox_root_mac")

#%%
# -----------------------------------------------------------
# define parameters
# -----------------------------------------------------------
# Initialize color-threshold handler for the isoluminant plane
stim_dim = 2
color_thres_data = color_thresholds(stim_dim, base_dir, plane_2D = 'Isoluminant plane')

# Load monitor-specific calibration and transformation matrices
cal_date = "02012026"
color_thres_data.load_transformation_matrix(file_date = cal_date)
M_RGBTo2DW = color_thres_data.M_RGBTo2DW  # RGB → 2D Wishart space
M_2DWToRGB = color_thres_data.M_2DWToRGB  # 2D Wishart space → RGB

# Bounds for the coarse reference grid used for threshold simulations / ellipse fitting
lb_W_grid = -0.7 
ub_W_grid = 0.7

# Number of reference points per axis in the 2D grid
nGridPts_ref = 7  # Can be adjusted for finer/coarser sampling

# Reference grid spanning the normalized Wishart space
grid_ref = np.linspace(lb_W_grid, ub_W_grid, nGridPts_ref)

# Create a 2D mesh of reference locations in W space
# Shape: (nGridPts_ref, nGridPts_ref, 2)
ref_points_W = np.stack(np.meshgrid(grid_ref, grid_ref), axis=-1)

# Flatten grid for batch processing
# Shape: (nGridPts_ref^2, 2)
ref_points_W_flat = ref_points_W.reshape(-1, stim_dim)

# Define chromatic directions for isothreshold probing
numDirPts = 16
grid_theta_xy = UnitCircleGenerate(numDirPts)

# Specify color-difference metric and threshold criterion
color_diff_algorithm = 'CIE2000'  # Options: 'CIE2000', 'CIE1994', 'CIE1976'

# Define 1-JND criterion in ΔE units
# (larger for CIE1976 to roughly match perceptual scale)
deltaE_1JND = 1.75 #2.5

# Angular resolution used to render fitted isothreshold ellipses
nTheta = 200 

# Optional global scaling applied to fitted ellipses
scaler = 1

#%% 
# -----------------------------------------------------------
#  derive CIELab predicted thresholds
# -----------------------------------------------------------
# Background RGB value used for Lab conversion (neutral gray)
#background = np.array([0.2357, 0.2969, 0.4673]) #RGB
background = np.array([0.2649, 0.2811, 85.6234] ) #xyY

# Initialize simulator for CIELab-based threshold computations restricted to the isoluminant plane
sim_thres_CIELab = SimThresCIELab(background, 
                                  plane_2D_list=['Isoluminant plane'],
                                  file_date= cal_date,
                                  flag_force_Lstar50=True,
                                  background_space = "xyY"
                                  )
if sim_thres_CIELab.flag_force_Lstar50:
    if sim_thres_CIELab.background_space.lower() == 'xyy':
        background_RGB = sim_thres_CIELab.M_XYZToRGB @ colour.xyY_to_XYZ(background)
    else:
        background_RGB = background
    bg_lab, _ = sim_thres_CIELab.convert_rgb_lab(background_RGB)
    print(f'rescaled lab: {np.round(bg_lab, 2)}')

# Preallocate arrays for threshold computation and ellipse fitting
base_size  = (nGridPts_ref, nGridPts_ref)
opt_vecLen = np.full(base_size + (numDirPts,), np.nan)     # Vector lengths for ΔE=1
W_comp_contour_scaled     = np.full(base_size + (stim_dim, numDirPts,), np.nan)
W_comp_contour_unscaled   = np.full(base_size + (stim_dim, numDirPts,), np.nan)
fitEllipse_scaled         = np.full(base_size + (stim_dim, nTheta,), np.nan)
fitEllipse_unscaled       = np.full(base_size + (stim_dim, nTheta,), np.nan)
rgb_comp_contour_unscaled = np.full(base_size + (3, numDirPts,), np.nan)  # Threshold points in RGB
rgb_comp_contour_scaled   = np.full(base_size + (3, numDirPts,), np.nan)
ellParams                 = np.full(base_size + (5,), np.nan)  # 5 free parameters for ellipse fitting

# Loop over reference locations and chromatic directions
for idx in np.ndindex(*base_size):
    # Reference stimulus location in Wishart (W) space
    W_ref_idx = ref_points_W[idx] #(2, )
    
    # For each chromatic direction, find the threshold point
    for k in range(numDirPts):
        # Search along the specified direction until the target ΔE is reached
        rgb_ref_idx, _, opt_vecLen[*idx, k], rgb_comp_contour_unscaled[*idx,:,k], \
            W_comp_contour_unscaled[*idx,:,k] = \
            sim_thres_CIELab.find_threshold_point_on_isoluminant_plane(\
                            W_ref_idx, 
                            grid_theta_xy[:, k], 
                            M_RGBTo2DW,
                            M_2DWToRGB, 
                            deltaE = deltaE_1JND,
                            coloralg = color_diff_algorithm
                            )
    
    rgb_comp_contour_scaled[idx] = rgb_comp_contour_unscaled[idx] * opt_vecLen[idx][None] + rgb_ref_idx[:,None]
        
    # Fit an isothreshold ellipse in the model space
    fitEllipse_scaled[idx], fitEllipse_unscaled[idx], ellParams[idx], \
        W_comp_contour_scaled[idx] = fit_2d_isothreshold_contour(W_ref_idx, 
                                    W_comp_contour_unscaled[idx],
                                    nTheta = nTheta, 
                                    ellipse_scaler = scaler,
                                    flag_force_centered_ref=True,
                                    )

#%% 
# output directories
output_figDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_FigFiles','2D',f'{color_diff_algorithm}')
output_fileDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_DataFiles', '2D',f'{color_diff_algorithm}')
os.makedirs(output_figDir, exist_ok=True)
os.makedirs(output_fileDir, exist_ok=True)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 15)

# -----------------------------------------------------------
# plot on a 2D plane
# -----------------------------------------------------------
# color map
rgb_ref_points = color_thres_data.W2D_to_rgb(ref_points_W_flat)
cmap = np.reshape(rgb_ref_points, base_size + (3,))

# Initialize visualization object
sim_CIE_vis = CIELabVisualization(sim_thres_CIELab, 
                                  settings = pltSettings_base,
                                  save_fig=False)

pred2D_settings = replace(Plot2DSinglePlaneSettings(), **pltSettings_base.__dict__)
pred2D_settings = replace(pred2D_settings, 
                          rgb_background=None, 
                          lim = [-1,1],
                          ticks = np.linspace(-0.7,0.7,5),
                          ref_mc=cmap,
                          ell_lc=cmap)
# Plot 2D plane with computed ellipses and thresholds
fig1, ax1 = sim_CIE_vis.plot_2D_single_plane(ref_points_W, 
                                             fitEllipse_scaled, 
                                             rawData = W_comp_contour_scaled,
                                             settings = pred2D_settings
                                             )
for idx in np.ndindex(*base_size):
    ax1.scatter(*W_comp_contour_unscaled[idx],
                marker = 'o', s = 25,
                color=cmap[idx], 
                edgecolor= 'none'
                )
ax1.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()
# Save the figure as a PDF
space_tag = "xyY" if sim_thres_CIELab.background_space.lower() == "xyy" else "RGB"
str_cr = f"_cr_{space_tag}_{background[0]:.4f}_{background[1]:.4f}_{background[2]:.4f}"
if sim_thres_CIELab.flag_force_Lstar50:
    str_cr += '_bgLstar_fixed50'
else:
    str_cr += ''
fig1.savefig(os.path.join(output_figDir, f"{color_diff_algorithm}_derived_threshold"+\
                          f"_contours_isoluminant_plane_Wspace_grid{nGridPts_ref}{str_cr}.pdf"))

#%%
# -----------------------------------------------------------
# plot in 3D
# -----------------------------------------------------------
#convert 2DW to RGB
fE_u_1 = np.moveaxis(fitEllipse_unscaled, 2, -1)
fE_u_2 = np.reshape(fE_u_1, (-1, 2))
fE_u_3 = color_thres_data.W2D_to_rgb(fE_u_2)
fitEllipse_unscaled_RGB = np.moveaxis(np.reshape(fE_u_3, fE_u_1.shape[:-1] + (3,)), -1, 2)

fE_s_1 = np.moveaxis(fitEllipse_scaled, 2, -1)
fE_s_2 = np.reshape(fE_s_1, (-1, 2))
fE_s_3 = color_thres_data.W2D_to_rgb(fE_s_2)
fitEllipse_scaled_RGB = np.moveaxis(np.reshape(fE_s_3, fE_s_1.shape[:-1] + (3,)), -1, 2)

# Load precomputed color-space transformation data from MATLAB
mat_file = loadmat('Transformation_btw_color_spaces.mat')
iso_mat = mat_file['DELL_02242025_texture_right'][0]

# Extract the monitor gamut RGB values (background + primaries)
gamut_rgb = iso_mat['gamut_bg_primary'][0]

# visualize the isoluminant slice in the RGB cube
fig2 = plt.figure(figsize=(9, 7),dpi = 1024)
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot(*gamut_rgb,color ='k')

# Scatter plot
for idx in np.ndindex(*base_size):
    ax2.scatter(*cmap[idx], c=cmap[idx], marker='+')
    ax2.scatter(*fitEllipse_unscaled_RGB[idx], c=cmap[idx], 
               marker='.', s=2, alpha=0.7, edgecolors=cmap[idx])
    ax2.scatter(*rgb_comp_contour_unscaled[idx], marker = 'o', s = 15,
               color=cmap[idx], edgecolor= cmap[idx],alpha =0.5)
ax2.view_init(elev=30, azim=-120)
pltTools = PlottingTools(pltSettings_base)
pltTools._update_axes_labels(ax2, np.linspace(0,1,5), np.linspace(0,1,5), ndims = 3)
pltTools._update_axes_limits(ax2, lim = [0, 1], ndims = 3)
pltTools._configure_labels_and_title(ax2, title = 'RGB cube', ndims = 3)
ax2.set_aspect('equal')
plt.show()
# Save the figure as a PDF
#fig2.savefig(os.path.join(output_figDir, f"{color_diff_algorithm}_derived_threshold"+\
#                          f"_contours_isoluminant_plane_RGBspace_grid{nGridPts_ref}{str_cr}.pdf"))
        
#%%
# -----------------------------------------------------------
# save the data
# -----------------------------------------------------------
file_name = f'Isothreshold_ellipses_isoluminant_{color_diff_algorithm}{str_cr}.pkl'
full_path = os.path.join(output_fileDir, file_name)

# NOTE ON SHAPES / COMPATIBILITY
# In `isothreshold_ellipses_3slices_CIELab.py`, we generate three 2D slices of the RGB cube
# (fixed R, fixed G, fixed B). For that script, `ref_points` is stored as:
#   ref_points.shape == (3, nGridPts_ref, nGridPts_ref, 3)
#
# Here we only simulate ONE plane (isoluminant plane in W-space), but we still export the
# results in the same "3-slice" container format to keep downstream analysis/plotting code
# consistent and to simplify indexing in `color_thresholds` and related utilities.

# We use 3 repeats as a placeholder “slice axis” for compatibility with the 3-slice pipeline.
num_repeats = 3

# Convert to the “3-slice” format:
#   ref_points: (num_repeats, nGridPts_ref, nGridPts_ref, 3)
# - repeat along the slice axis to make the shape compatible with the 3-slice pipeline
fixed_RGBvec = 1 #filler value for w
W_ref = np.concatenate([ref_points_W, np.full((nGridPts_ref, nGridPts_ref, 1),fixed_RGBvec)], axis=-1)
ref_points = np.repeat(W_ref[None], num_repeats, axis = 0)
stim_keys = ['fixed_RGBvec', 'grid_ref', 'nGridPts_ref', 'ref_points', 
             'background_RGB','numDirPts', 'grid_theta_xy', 'deltaE_1JND']

stim = {}
for i in stim_keys: stim[i] = eval(i)

#save the results
results_keys = ['opt_vecLen', 'fitEllipse_scaled', 'fitEllipse_unscaled',
                'rgb_comp_contour_unscaled','rgb_comp_contour_scaled', 
                'ellParams', 'fitEllipse_scaled_RGB', 
                'fitEllipse_unscaled_RGB', 'W_comp_contour_scaled',
                'W_comp_contour_unscaled']

# For consistency with the 3-slice export format, add a leading slice axis to each result
# and repeat it `num_repeats` times:
#   (nGridPts_ref, nGridPts_ref, ...)  ->  (num_repeats, nGridPts_ref, nGridPts_ref, ...)
for var_name in results_keys:
    globals()[var_name] = np.repeat(globals()[var_name][None], num_repeats, axis=0)
    
results = {}
results['scaler'] = scaler
for i in results_keys: results[i] = eval(i)

# key string
ext_str = f'_grid{nGridPts_ref}'
# Common payload for this grid size
data_dict = {
    f'sim_thres_CIELab{ext_str}': sim_thres_CIELab,
    f'stim{ext_str}': stim,
    f'results{ext_str}': results
}

if os.path.exists(full_path):
    with open(full_path, 'rb') as f:
        existing_dict = pickled.load(f)

    flag_match_grid_pts = (f'stim{ext_str}' in existing_dict)

    if flag_match_grid_pts:
        flag_overwrite = input(f"The file '{file_name}' already exists. Enter 'y' to overwrite: ")

        if flag_overwrite.lower() == 'y':
            with open(full_path, 'wb') as f:
                pickled.dump(data_dict, f)
        else:
            print("File not overwritten.")
    else:
        existing_dict.update(data_dict)
        with open(full_path, 'wb') as f:
            pickled.dump(existing_dict, f)
else:
    with open(full_path, 'wb') as f:
        pickled.dump(data_dict, f)
    
