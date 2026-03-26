#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:02:33 2024

@author: fangfang

This script derives 3D isothreshold contours in CIELab space from RGB stimuli.

Overview
--------
The script computes color-discrimination isothreshold 3D surface by converting RGB
stimuli to CIELab space using calibrated monitor and visual system models, then
identifying comparison stimuli that produce a fixed perceptual color difference
(ΔE) from a reference.

General procedure
-----------------
1. For each selected reference stimulus, RGB values are converted to CIELab space.
   This conversion uses:
     - The monitor’s spectral power distribution (SPD),
     - The background RGB (adaptation) color,
     - Stockman–Sharpe 2° cone fundamentals,
     - A calibrated transformation from LMS cone responses to CIEXYZ
       (M_LMS_TO_XYZ).

   The monitor calibration data and transformation matrices were output from
   MATLAB code (`t_WFromPlanarGamut.m`).

2. A set of chromatic directions is defined in each 2D color plane. For each
   direction, we search along that direction to find the RGB value of a comparison
   stimulus that produces a target color difference (ΔE = 2.5) relative to the
   reference.

3. The resulting discrete isothreshold contour points are fit with an ellipsoid
   with the ellipsoid center constrained to coincide with the reference stimulus.

"""

import os
import numpy as np
import dill as pickled
from tqdm import tqdm
from dataclasses import replace
from analysis.ellipses_tools import ellParams_to_covMat
from analysis.ellipsoids_tools import UnitCircleGenerate_3D, fit_3d_isothreshold_ellipsoid
from analysis.simulations_CIELab import SimThresCIELab
from plotting.wishart_plotting import PlotSettingsBase            
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization_html,\
    Plot3DPredHTMLSettings
import plotly.graph_objects as go

#%% Set values
ndims = 3  # Number of color dimensions (R, G, B)

# Background RGB used as the normalization/adaptation point (neutral gray)
background_RGB = np.array([0.5, 0.5, 0.5])

# Bounds for the coarse reference grid used for threshold simulation / ellipsoid fitting
lb_RGB_grid = 0.15  # Approx. corresponds to -0.7 in model (W) space
ub_RGB_grid = 0.85  # Approx. corresponds to +0.7 in model (W) space
nGridPts_ref = 7    # Number of reference samples per RGB axis (usually 5 or 7)

# Number of directions on the isoluminant (xy) plane around each reference
numDirPts_xy = 16

# Number of polar (z-axis) samples for 3D directions; fewer due to spherical sampling geometry
numDirPts_z = int(np.ceil(numDirPts_xy / 2)) + 1

# Direction-grid resolution used only for generating smooth ellipsoid surfaces for plotting
nTheta = 200
nPhi = 100

# Optional radial scaling applied about the reference when visualizing fitted ellipsoids
scaler = 1

# Target threshold level (ΔE at 1 JND) used in the threshold search
deltaE_1JND = 2.5

# Choice of color-difference metric for ΔE computations
color_diff_algorithm = 'CIE2000' #or CIE1976, CIE1994, CIE2000

# Unit direction vectors on the sphere (shape: (numDirPts_z, numDirPts_xy, 3) or similar)
grid_xyz = UnitCircleGenerate_3D(numDirPts_xy, numDirPts_z)

# 1D grid of reference RGB coordinates (coarse sampling)
grid_ref = np.linspace(lb_RGB_grid, ub_RGB_grid, nGridPts_ref)

# 3D mesh of reference RGB locations; shape: (nGridPts_ref, nGridPts_ref, nGridPts_ref, 3)
ref_points = np.stack(np.meshgrid(grid_ref, grid_ref, grid_ref, indexing="ij"), axis=-1)

# Threshold-search helper configured with the chosen background
sim_thres_CIELab = SimThresCIELab(background_RGB)

#%%
# Shapes:
# - base_shape1 indexes reference RGB locations on the coarse 3D grid
# - base_shape2 indexes direction samples on the sphere (polar x azimuth)
base_shape1 = (nGridPts_ref, nGridPts_ref, nGridPts_ref)
base_shape2 = (numDirPts_z, numDirPts_xy)

# Preallocate outputs
# opt_vecLen: optimal step length (per ref, per direction) that reaches ΔE = deltaE_1JND
opt_vecLen = np.full(base_shape1 + base_shape2, np.nan)

# Ellipsoid surface points (3 x nSurfacePts)
fitEllipsoid_scaled   = np.full(base_shape1 + (ndims, nTheta * nPhi), np.nan)
fitEllipsoid_unscaled = np.full(base_shape1 + (ndims, nTheta * nPhi), np.nan)

# Discrete threshold samples on the isothreshold surface (3 x nDirs)
rgb_comp_surface_unscaled = np.full(base_shape1 + (ndims, numDirPts_xy * numDirPts_z), np.nan)
rgb_comp_surface_scaled   = np.full(base_shape1 + (ndims, numDirPts_xy * numDirPts_z), np.nan)

# Per-reference dictionary of fitted ellipsoid parameters (center, radii, evecs, etc.)
ellParams = np.full(base_shape1, {})

# Covariance matrices derived from fitted ellipsoid parameters (for later visualization)
covMat_vis = np.full(base_shape1 + (ndims, ndims), np.nan)

# Main loop: for each reference stimulus, find the isothreshold surface
for idx in tqdm(np.ndindex(*base_shape1), total=np.prod(base_shape1), desc="Computing"):
    # Reference RGB at this grid location
    rgb_ref_idx = ref_points[idx]

    # For each sampled direction on the sphere, find the step length (magnitude)
    # that yields the target color difference ΔE = deltaE_1JND.
    for d in np.ndindex(*base_shape2):
        vecDir = grid_xyz[d]  # unit direction vector in RGB space

        opt_vecLen[*idx, *d] = sim_thres_CIELab.find_vecLen(rgb_ref_idx,
                                                            vecDir,
                                                            deltaE_1JND,
                                                            coloralg=color_diff_algorithm
                                                            )

    # Convert (direction, magnitude) into discrete threshold RGB points on the surface:
    # rgb_comp = rgb_ref + vecDir * opt_vecLen
    rgb_comp_surface_unscaled[idx] = np.reshape(
        grid_xyz * opt_vecLen[idx][..., None] + rgb_ref_idx[None, None],
        (-1, ndims)
    ).T

    # Fit an ellipsoid to the discrete threshold points and (optionally) rescale it about ref
    fitEllipsoid_scaled[idx], fitEllipsoid_unscaled[idx], ellParams[idx], rgb_comp_surface_scaled[idx] = \
        fit_3d_isothreshold_ellipsoid(rgb_ref_idx,
                                      rgb_comp_surface_unscaled[idx].T,
                                      nTheta=nTheta,
                                      nPhi=nPhi,
                                      ellipsoid_scaler=scaler,
                                      flag_force_centered_ref=True,
                                      )

    # Convert fitted ellipsoid parameters to a covariance matrix for downstream visualization
    covMat_vis[idx] = ellParams_to_covMat(ellParams[idx]['radii'],
                                          ellParams[idx]['evecs']
                                          )
                        
#%% visualize ellipsoids
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = os.path.join(base_dir,'ELPS_analysis','Simulation_FigFiles', '3D',f'{color_diff_algorithm}')
output_fileDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_DataFiles', '3D',f'{color_diff_algorithm}')
os.makedirs(output_figDir, exist_ok=True)
os.makedirs(output_fileDir, exist_ok=True)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 12)
                                    
figname = f"Isothreshold_ellipsoid_CIELABderived_{color_diff_algorithm}"
pltSettings_html = Plot3DPredHTMLSettings()
pltSettings_html = replace(pltSettings_html,
                           ticks = grid_ref, lim = [0, 1], ell_alpha = 0.4,
                           xlabel = 'R', ylabel = 'G', zlabel = 'B',
                           ) 

# Visualization helper with HTML settings
vis_html = WishartPredictionsVisualization_html(settings=pltSettings_html)

fig = go.Figure()
# Render 3D ellipsoids (mesh surfaces) evaluated on the isoluminant plane
vis_html.plot_ellipsoids_mesh_cov(fig, ref_points, covMat_vis)
# Apply consistent 3D layout (camera, axes, lighting, hover behavior)
vis_html.apply_3d_layout(fig)
# Save interactive HTML
out_html = os.path.join(output_figDir, f"{figname}_grid{nGridPts_ref}.html")
fig.write_html(out_html, include_plotlyjs=True)
    
    
#%% Save data
file_name = f'{figname}.pkl'
full_path = os.path.join(output_fileDir, file_name)

#save all the stim info
stim_keys = ['nGridPts_ref', 'grid_ref', 'ref_points', 'background_RGB',
             'numDirPts_xy', 'numDirPts_z', 'deltaE_1JND']
stim = {}
for i in stim_keys: stim[i] = eval(i)

results_keys = ['opt_vecLen', 'fitEllipsoid_scaled', 'fitEllipsoid_unscaled',
                'rgb_comp_surface_scaled',  'rgb_comp_surface_unscaled',
                'ellParams', 'scaler']
results = {}
for i in results_keys: results[i] = eval(i)    
    
#% check if there is existing file
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
