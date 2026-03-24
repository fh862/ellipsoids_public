#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 23:02:03 2025

@author: fangfang

This script computes threshold surfaces—and their isoluminant-plane slices—at the
reference locations used in the suprathreshold pilot experiment.

Plotting the model-predicted threshold contours alongside equal-geodesic-distance
contours provides an interpretable comparison: it reveals the perceptual scale
of geodesic distances in units of threshold distance.

Important task-design note:
The suprathreshold experiment uses a 2AFC “oddity relative to a pre-cued reference”
task. Participants view two comparison stimuli and decide which one differs more
from the reference. Because the reference is explicitly cued, chance performance
is 0.5 and perfect performance is 1. Thus the threshold is defined at p(correct)=0.75.
To stay consistent with this task structure, the simulation function must change 
from `simulate_oddity` (3-AFC) to `simulate_oddity_reference` (2-AFC with pre-cued
reference).

In this script, we:
1. Regenerate threshold ellipsoids using the updated simulation function and
   0.75-correct threshold criterion.
2. Intersect each threshold ellipsoid with the isoluminant plane (the plane on
   which all suprathreshold reference stimuli lie).
3. Append the results to the loaded pickle file. 

Optional: see eval_equal_geodesic_contour_3D.py for the plot with the resulting 
contours together with equal-geodesic-distance contours.

"""

import jax
jax.config.update("jax_enable_x64", True)
import dill as pickled
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from copy import deepcopy
from dataclasses import replace
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os
script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from analysis.utils_load import select_file_and_get_path
from analysis.ellipsoids_tools import slice_ellipsoid_byPlane
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.sim_CIELab_plotting import CIELabVisualization, Plot3DSettings
from core.model_predictions import rerun_model_pred_wExisting_model
from core.oddity_task import simulate_oddity_reference

#base directory
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

#%%
#---------------------------------------------------------------------------
# SECTION 0: load the model fits to the empirical data
# --------------------------------------------------------------------------
# Select the file containing the model fits
# Navigate to the directory: ELPS_analysis/Experiment_DataFiles/pilot3/sub1/fits
# 'Fitted_ColorDiscrimination_6dExpt_RGBcube_sub1_decayRate0.4_nBasisDeg5.pkl'
input_fileDir_fits, file_name = select_file_and_get_path()

# Construct the full path to the selected file
full_path = os.path.join(input_fileDir_fits, file_name)

# Load the necessary variables from the file
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)

# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict['color_thres_data']
color_thres_data.base_path = base_dir
color_thres_data.load_transformation_matrix()
ndims = color_thres_data.color_dimension #3D

# load model predictions for a grid of reference stimuli (5 x 5 x 5)
model_pred_thres = deepcopy(vars_dict['model_pred_Wishart'])
grid_3D = vars_dict['grid']
grid_3D_trans = vars_dict['grid_trans']
num_grid_pts_3D = vars_dict['NUM_GRID_PTS']

# Create the output directory if it doesn't exist
output_figDir_fits = input_fileDir_fits.replace('DataFiles', 'FigFiles')
os.makedirs(output_figDir_fits, exist_ok=True)

#%%
#---------------------------------------------------------------------------
# SECTION 1: visualize threshold ellipsoids
# --------------------------------------------------------------------------
# Create an instance of the class
pltSettings_base = PlotSettingsBase(fig_dir= output_figDir_fits, fontsize = 11)
Plot3D_settings = replace(Plot3DSettings(), **pltSettings_base.__dict__)
Plot3D_settings = replace(Plot3D_settings,
                          fig_size = (6,6),
                          visualize_thresholdPoints = False,
                          visualize_ellipsoids = True,
                          lim = [-1,1],
                          ticks = np.linspace(-1, 1, 5),
                          surf_alpha = 0.5,
                          flag_input_W = True,
                          title = None,
                          view_angle = [30,-120])

vis = CIELabVisualization(color_thres_data, settings = pltSettings_base)

#create a figure
fig1 = plt.figure(figsize=Plot3D_settings.fig_size, dpi=Plot3D_settings.dpi,
                  constrained_layout=True)
ax1 = fig1.add_subplot(111, projection='3d')
vis.plot_3D(np.reshape(grid_3D_trans,(num_grid_pts_3D**ndims, ndims)),
            np.reshape(model_pred_thres.fitEll_scaled,(num_grid_pts_3D** ndims, ndims,-1)),
            ax = ax1,
            settings = Plot3D_settings)
ax1.legend(bbox_to_anchor=(0.5, -0.2), fontsize= Plot3D_settings.fontsize - 1)
# Save the figure as a PDF
#fig1.savefig(os.path.join(output_figDir_fits, f"Ellipsoids_{file_name[:-4]}.pdf"))    
plt.show()

#%%
#---------------------------------------------------------------------------
# SECTION 2: compute the model predicted threshold ellipsoids
# -------------------------------------------------------------------------- 
#load conditions of the suprathresholds
# 'SUPT_data/sub1'
# 'Suprathres_ColorDiscrimination_2dExpt_Isoluminant plane_sub1_session1.pkl'
input_fileDir_suprathres, file_name_suprathres = select_file_and_get_path()

# Construct the full path to the selected file
full_path_suprathres = os.path.join(input_fileDir_suprathres, file_name_suprathres)

# Load the necessary variables from the file
with open(full_path_suprathres, 'rb') as f:
    vars_dict_suprathres = pickled.load(f)

# `color_thres_data_2D` includes the RGB↔2DW/3DW transforms
color_thres_data_2D = vars_dict_suprathres['color_thres_data']

# Reference locations used in the suprathreshold experiment (in 2DW space, ∈ [-1, 1]^2)
xref_2DW_suprathres = vars_dict_suprathres['ref']
nRefs_suprathres = xref_2DW_suprathres.shape[0]

# Convert 2DW reference points → RGB
xref_RGB_suprathres  = color_thres_data.W2D_to_rgb(xref_2DW_suprathres)

# Convert RGB → 3DW (W-space used by the model)
xref_3DW_suprathres = color_thres_data_2D.N_unit_to_W_unit(xref_RGB_suprathres)

# Update the model's simulation function to match the 2AFC suprathreshold task
model_pred_thres.params['simulation_func'] = simulate_oddity_reference
# Threshold criterion for 2AFC (chance = 0.5 → threshold at p(correct)=0.75)
model_pred_thres.target_pC = 0.75
# Wrap reference list into a grid shape expected by `rerun_model_pred_wExisting_model`
grid_suprathres = xref_3DW_suprathres[None, None]

# Recompute threshold ellipsoids at the suprathreshold reference locations
model_pred_thres, _ = rerun_model_pred_wExisting_model(grid_suprathres,
                                                       model_pred_thres, 
                                                       color_thres_data
                                                       )

#%%
#---------------------------------------------------------------------------
# SECTION 3: slice the threshold ellipsoids by the isoluminant plane
# --------------------------------------------------------------------------
# Corners of the 2D model square (W2D; homogeneous coord will be appended)
corner_points_W = np.array([[-1, -1], [ 1, -1], [ 1,  1], [-1,  1]])

# Map 2D model corners to device RGB (in [0,1]); shape: (3, 4)
corner_points_rgb = color_thres_data.W2D_to_rgb(corner_points_W).T

# Map those RGB points onto the 3D model space (W3D ∈ [-1,1]^3); shape: (3, 4)
corner_points = color_thres_data_2D.N_unit_to_W_unit(corner_points_rgb)

# Mean-center the RGB points (columns are points)
centered_points = corner_points_rgb - np.mean(corner_points_rgb, axis=1)[:, None]

# SVD on transposed (points × dims) to obtain right-singular vectors (Vt)
# The first two rows of Vt are orthonormal in-plane directions
_, _, Vt = np.linalg.svd(np.transpose(centered_points, (1, 0)))

# -----------------------------------------------------------------------
# Slice each 3D ellipsoid by the isoluminant plane defined by Vt[0], Vt[1]
# and collect both the 3D slice and its 2D projection back to the plane
# -----------------------------------------------------------------------
nTheta = 200
sliced_ell_byPlane  = np.full((nRefs_suprathres, 3, nTheta), np.nan)  # (N, 3, θ)
flat_ell_isoluminant = np.full((nRefs_suprathres, 2, nTheta), np.nan)  # (N, 2, θ)

for n in range(nRefs_suprathres):
    # Unpack ellipsoid parameters for stimulus n
    ell_params_n = model_pred_thres.params_ell[0][0][n]
    radii_n  = ell_params_n['radii']
    center_n = np.reshape(ell_params_n['center'], (-1))
    evecs_n  = ell_params_n['evecs']

    # Intersect ellipsoid with the plane spanned by Vt[0], Vt[1]
    sliced_ell_byPlane[n], _ = slice_ellipsoid_byPlane(
        center_n, radii_n, evecs_n, Vt[0], Vt[1], num_grid_pts=nTheta
    )

    # Project the 3D slice back to the 2D model plane for “flat” visualization
    # W3D → RGB → W2D (drop homogeneous row afterward)
    flat_ell = color_thres_data_2D.M_RGBTo2DW @ \
        color_thres_data.W_unit_to_N_unit(sliced_ell_byPlane[n])
    flat_ell_isoluminant[n] = flat_ell[:2]

#create a figure
fig3 = plt.figure(figsize=Plot3D_settings.fig_size, dpi=Plot3D_settings.dpi, 
                  constrained_layout=True)
ax3 = fig3.add_subplot(111, projection='3d')

# draw plane (no label)
plane = Poly3DCollection([corner_points.T], edgecolor= [0.2,0.2,0.2])
plane.set_facecolor(np.array([[0.5, 0.5, 0.5, 0.3]]))  # RGBA as (1,4) array
ax3.add_collection3d(plane)

# draw ellipses and keep the first line handle
line_handle = None
for n in range(nRefs_suprathres):
    h, = ax3.plot(*sliced_ell_byPlane[n], color='k')
    if line_handle is None:
        line_handle = h

# legend proxies
handles = [
    Patch(facecolor=(0.5, 0.5, 0.5, 0.3), edgecolor='none', label='Isoluminant plane'),
    Line2D([0], [0], color='k', label='Threshold ellipses sliced by the plane'),
]
ax3.legend(handles=handles, loc='lower center', 
           bbox_to_anchor=(0.5, -0.2), fontsize= Plot3D_settings.fontsize - 1)

ax3.scatter(*xref_3DW_suprathres.T, marker = '+', color = 'k', s = 20)

vis.plot_3D(np.reshape(grid_suprathres, (nRefs_suprathres, ndims)),
            np.reshape(model_pred_thres.fitEll_scaled, (nRefs_suprathres, ndims,-1)),
            ax = ax3,
            settings = Plot3D_settings)
# Save the figure as a PDF
output_figDir_suprathres = os.path.join(input_fileDir_suprathres,'fits')
os.makedirs(output_figDir_suprathres, exist_ok=True)
fig3.savefig(os.path.join(output_figDir_suprathres,
                          f"Suprathres_refs_at_isoluminant_plane_{file_name[:-4]}.pdf"))    
plt.show()

#%%
#---------------------------------------------------------------------------
# SECTION 4: Append the data to the existing pkl file
# -------------------------------------------------------------------------- 
# Add new variables
vars_dict_suprathres['model_pred_thres'] = model_pred_thres
vars_dict_suprathres['pred_thres_3DW'] = sliced_ell_byPlane
vars_dict_suprathres['pred_thres_2DW'] = flat_ell_isoluminant

# Save back to the same file (overwrite)
with open(full_path_suprathres, 'wb') as f:
    pickled.dump(vars_dict_suprathres, f)


