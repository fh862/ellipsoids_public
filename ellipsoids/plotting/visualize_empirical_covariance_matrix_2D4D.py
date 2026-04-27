#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:37:26 2024

@author: fangfang

This script visualizes the spatial field of covariance matrices implied by the
best-fit Wishart process model.

It produces two types of visualizations:

(1) Covariance-matrix field
    The estimated 2×2 covariance matrix is shown across the stimulus plane as
    four heatmaps:
        • top-left:     σ²_dim1 (variance along stimulus dimension 1)
        • bottom-right: σ²_dim2 (variance along stimulus dimension 2)
        • off-diagonal: σ²_dim1,dim2 (covariance between dimensions)
    In addition, the corresponding covariance ellipses and model-predicted
    discrimination thresholds (66.7% correct) are overlaid at a grid of
    reference locations.

(2) Intermediate weighted-sum components
    Six heatmaps (2 × 3) visualize the weighted sums of Chebyshev basis
    functions that precede the Einstein summation step used to construct the
    covariance-matrix field.

Current limitations:
    • The script is currently restricted to 2D stimulus spaces (yielding a
      4D psychometric field).
    • Extension to 3D stimulus spaces (6D psychometric fields) is left for
      future work.
    • `flag_random_ref` must be set to False: older fitted `.pkl` files were
      generated with an outdated JAX version and cannot recompute predicted
      thresholds at newly sampled (e.g., Sobol) reference locations. Revisit 
      this in the future.

"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import os
import dill as pickled
from dataclasses import replace
import matplotlib.pyplot as plt

#load functions from other modules
from core.model_predictions import rerun_model_pred_wExisting_model
from plotting.wishart_plotting import WishartModelBasicsVisualization,\
    PlotSettingsBase, PlotCovMatSettings, PlotUSettings
from analysis.utils_load import select_file_and_get_path
from analysis.MOCS_thresholds import sim_MOCS_trials

# Define the file name and output directory for model fitting data files
baseDir  = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fig_output_sim = os.path.join(baseDir,'ELPS_analysis','WishartPractice_FigFiles')

#%% 
# -------------------------------------------------------------
# SECTION 1: Load data
# -------------------------------------------------------------
# Prompt user to select the pickled model fit file
# 'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits'
#'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1.pkl'
input_fileDir_fits, file_name = select_file_and_get_path()
full_path_fits = os.path.join(input_fileDir_fits, file_name)

# Load variables from the selected .pkl file
with open(full_path_fits, 'rb') as f:
    vars_dict = pickled.load(f)

# Extract Wishart model predictions and color threshold transformation object
grid_new  = vars_dict['grid']
modelpred = vars_dict['model_pred_Wishart']
W_est     = modelpred.W_est              # Estimated covariance matrices at each location
model     = modelpred.model              # The Wishart model used
ndims     = model.num_dims 
color_thres_data = vars_dict['color_thres_data']
color_thres_data.base_path = baseDir
color_thres_data.load_transformation_matrix(file_date = "02242025")

#%%
# -------------------------------------------------------------
# SECTION 2: Load / compute model predictions on plotting grids
# -------------------------------------------------------------
# Grid bounds and resolution
#   - coarse grid: used for selecting / visualizing reference locations
#   - fine grid:   used for dense covariance-field visualization
grid_1d_bds     = [-0.7, 0.7]
ngrid_pts       = 7
ngrid_pts_fine  = 41

# Build a dense grid in stimulus space (values clipped to [-1, 1] inside compute_U)
xgrid_dim1_fine = jnp.linspace(-1, 1, ngrid_pts_fine)
xgrid_fine = jnp.stack(
    jnp.meshgrid(*[xgrid_dim1_fine for _ in range(ndims)]),
    axis=-1
)  # shape: (ngrid_pts_fine, ngrid_pts_fine, ndims) for 2D

# Evaluate the fitted model on the fine grid to get covariance matrices everywhere
Sigmas_noise_grid_fine = model.compute_Sigmas(model.compute_U(W_est, xgrid_fine))

# -------------------------------------------------------------
# PART 2: Ensure we have model-predicted threshold contours on `grid_new`
# -------------------------------------------------------------
# Two ways to define `grid_new`:
#   (A) Random reference locations (Sobol points) for visualization / sampling
#   (B) A structured linspace grid, reusing cached predictions if already available
flag_random_ref = False

if flag_random_ref:
    # (A) Random (Sobol) reference locations within `grid_1d_bds`
    # Note: force_center=True ensures the grid includes the origin
    dots = sim_MOCS_trials.sample_sobol(
        ngrid_pts**2,
        [grid_1d_bds[0]]*2, [grid_1d_bds[1]]*2,
        force_center=True,
        seed = 0 #for reproducibility
    )
    grid_new = np.reshape(dots, (ngrid_pts, ngrid_pts, ndims))

    # Recompute threshold predictions on the new reference grid using the existing fit
    modelpred_new = rerun_model_pred_wExisting_model(
        grid_new, modelpred, color_thres_data
    )
    
    modelpred_thres = modelpred_new.fitEll_unscaled
    Sigmas_noise_grid_new = modelpred_new.Sigmas_noise_grid
else:
    # (B) Structured grid: reuse predictions if the loaded grid already matches
    loaded_grid_matches = (
        grid_new.shape[0] == ngrid_pts and
        np.min(grid_new) == grid_1d_bds[0] and
        np.max(grid_new) == grid_1d_bds[1]
    )

    if loaded_grid_matches:
        # Grid matches: keep the cached predictions from the loaded `modelpred`
        modelpred_new       = modelpred
        modelpred_thres     = modelpred.fitEll_unscaled
        Sigmas_noise_grid_new = modelpred.Sigmas_noise_grid
    else:
        # Grid differs: create a new structured grid and recompute predictions
        xgrid_dim1 = jnp.linspace(*grid_1d_bds, ngrid_pts)
        grid_new = jnp.stack(
            jnp.meshgrid(*[xgrid_dim1 for _ in range(ndims)]),
            axis=-1
        )

        modelpred_new = rerun_model_pred_wExisting_model(
            grid_new, modelpred, color_thres_data
        )
        
#%% 
# -------------------------------------------------------------
# SECTION 3: Visualize the estimated covariance field (2D)
# -------------------------------------------------------------
# The figure contains:
#   - Four heatmaps (left block): entries of the 2×2 covariance matrix across the plane
#       * upper-left:   Σ11 (variance along dim 1)
#       * upper-right:  Σ12 (covariance)
#       * lower-left:   Σ21 (covariance; same as Σ12 for symmetric matrices)
#       * lower-right:  Σ22 (variance along dim 2)
#   - One ellipse panel (right): covariance matrices and threshold contours (66.7% correct) 
#       at selected reference locations, illustrating smooth changes in the performance field.

# Create output directory for figures 
fig_outputDir = (input_fileDir_fits.replace('DataFiles', 'FigFiles').replace('fits', 'CovarianceMatrix_2d'))
os.makedirs(fig_outputDir, exist_ok=True)

# Base plotting settings shared across figures
pltSettings_base = PlotSettingsBase(fig_dir=fig_outputDir, fontsize=11)

# Plot settings specific to covariance-matrix visualization
pltCovSettings = replace(PlotCovMatSettings(), **pltSettings_base.__dict__)
visualize_sigma2D = WishartModelBasicsVisualization(
    save_fig=True,
    save_format='pdf',
    settings=pltCovSettings
)

# Titles for the heatmap panels (e.g., Σ11, Σ12, Σ21, Σ22)
ttl_list = pltCovSettings.heatmap_title_list

# Compute per-location ellipse colors by mapping 2DW grid points → RGB
grid_reshape = np.reshape(grid_new, (-1, 2))
# Reshape back to an image-like RGB array aligned with grid_new indexing
cmap_ell = np.reshape(color_thres_data.W2D_to_rgb(grid_reshape), (ngrid_pts,ngrid_pts,-1))  # shape: (ngrid_pts, ngrid_pts, 3)

# Figure naming: add suffix if references were randomly sampled
figname = f'CovarianceMatrix_{file_name[:-4]}'
figname_ext = '_randRef' if flag_random_ref else ''

pltCovSettings = replace(pltCovSettings,
                         slc_idx_dim1=ngrid_pts - 1,              # slice indices (used internally for annotations)
                         slc_idx_dim2=ngrid_pts - 1,
                         cmap_bds=[-0.008, 0.008],                
                         ticks_W=np.linspace(*grid_1d_bds, 3),    # axis ticks in 2DW coordinates
                         heatmap_title_list=ttl_list,
                         flag_add_horz_vert_lines=False,
                         flag_remake_cmap = True,
                         covMat_title='Isoluminant plane',
                         cmap_ell=cmap_ell,                        # per-location ellipse color (RGB)
                         scaler_ell=1,
                         fig_name=figname + figname_ext
                         )

# Plot covariance heatmaps (fine grid) and overlay ellipses/contours (coarse grid)
fig, ax, ax_ell = visualize_sigma2D.plot_2D_covMat(grid_new,
                                                   Sigmas_noise_grid_fine,
                                                   Sigmas_noise_grid_new,
                                                   settings=pltCovSettings
                                                   )

# Overlay the model-predicted 66.7% threshold contours on the ellipse panel
for n in range(ngrid_pts):
    for m in range(ngrid_pts):
        ax_ell.plot(*modelpred_thres[n, m], ls=':', color=cmap_ell[n, m], lw=1.2)

plt.show()

# Save the final figure
fig.savefig(os.path.join(fig_outputDir, figname + figname_ext + '.pdf'),
            format='pdf', bbox_inches='tight')
    
#%% 
# -------------------------------------------------------------
# SECTION 4: Visualize U (weighted sum)
# -------------------------------------------------------------
# compute the weighted sum of basis functions given sampled W
U_est = model.compute_U(W_est, xgrid_fine)

pltSettings_base2 = PlotSettingsBase(fig_dir=fig_outputDir, fontsize=9)
pltUSettings = replace(PlotUSettings(), **pltSettings_base2.__dict__)
pltUSettings = replace(pltUSettings, 
                       cmap_bds = [-0.08, 0.08],
                       flag_remake_cmap = True,
                       ticks = np.linspace(*grid_1d_bds, 3),
                       fig_name = f'U_{figname}',
                       fig_name_ext = f'{figname_ext}'
                       )
visualize_sigma2D.plot_U_2D(U_est, settings = pltUSettings)
