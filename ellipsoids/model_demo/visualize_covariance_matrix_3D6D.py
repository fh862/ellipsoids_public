#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:37:26 2024

@author: fangfang

Goal
----
Visualize the covariance field for color discrimination data in the full 3D RGB cube.

Context
-------
In this experiment, both reference and comparison stimuli vary within the entire
3D RGB cube, rather than being restricted to an isoluminant plane (as in the
2D/4D threshold experiments).

Stimulus space (3D):
    dim1, dim2, dim3  — RGB coordinates of the reference stimulus

Covariance field (6D parameterization):
    dim1, dim2, dim3  — RGB coordinates of the reference
    dim4, dim5, dim6 — delta values added to the reference to obtain the comparison
                        (i.e., the RGB offset of the comparison stimulus)

Interpretation
--------------
The underlying Chebyshev basis functions are defined over a 3D space (the
reference RGB coordinates). At each reference location in this space, the model
produces a 3×3 covariance matrix describing the local geometry of color
discrimination.

For visualizing covariance fields in other experimental designs (e.g., 2D/4D or
3D/5D threshold experiments), see:
    - Visualize_covariance_matrix_2D4D.py
    - Visualize_covariance_matrix_3D5D.py
    - Visualize_chebyshev_basis.py

"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
from analysis.utils_load import get_path
from dataclasses import replace
#load functions from other modules
from core.wishart_process import WishartProcessModel
from plotting.wishart_plotting import WishartModelBasicsVisualization,\
    PlotSettingsBase, PlotBasis3DSettings, PlottingTools
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization_html,\
    Plot3DPredHTMLSettings
import plotly.graph_objects as go

# Base directory for saving figures
baseDir = get_path("dropbox_root_mac")

# Directory for all Wishart practice figure files
fig_output_sim = os.path.join(baseDir, 'ELPS_analysis', 'WishartPractice_FigFiles')
    
#%%
# -----------------------------------------------------------
# SECTION 1: Randomly draw from prior (2D)
# -----------------------------------------------------------
ndims = 3
# Number of grid points in the coarse (for ellipses) and fine (for heatmaps) grids.
num_grid_pts      = 5
num_grid_pts_fine = 41

# PRNG key for sampling from the Wishart-process prior over W.
seed = 221
W_INIT_KEY   = jax.random.PRNGKey(seed)

# Initialize a 2D WishartProcessModel with ndims×ndims covariance matrices.
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions
    ndims,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    3e-4,  # Scale parameter for prior on `W`.
    0.8,   # decay rate on `W` 
    0,     # Diagonal term setting minimum variance for the ellipsoids.
    num_dims_cov= ndims
)

# Coarse grid of points in the 2D stimulus space (for plotting ellipses).
grid = jnp.stack(jnp.meshgrid(*[jnp.linspace(-0.7, 0.7, num_grid_pts) \
                                 for _ in range(model.num_dims)]), axis=-1)
    
grid_1d_fine = jnp.linspace(-1, 1, num_grid_pts_fine)
grid_fine = jnp.stack(jnp.meshgrid(*[grid_1d_fine \
                                 for _ in range(model.num_dims)]), axis=-1)
    
# Sample Chebyshev coefficients W from the prior.
W_init = model.sample_W_prior(W_INIT_KEY)

# Evaluate the weighted sum U on the fine and coarse grid.
U_fine = model.compute_U(W_init, grid_fine)
U = model.compute_U(W_init, grid)

# Compute covariance matrices Σ(x) on the fine and coarse grids.
Sigmas_grid_fine = model.compute_Sigmas(U_fine)
Sigmas_grid = model.compute_Sigmas(U) * 0.15

#%%
# -----------------------------------------------------------
# SECTION 2: Configure output paths and plotting settings
# -----------------------------------------------------------
# Subdirectory specific to this covariance-matrix visualization
fig_outputDir = os.path.join(fig_output_sim, f'CovarianceMatrix_{ndims}D{ndims * 2}D')

# Initialize base plotting settings (shared across different plot types)
pltSettings_base = PlotSettingsBase(
    fig_dir=fig_outputDir,
    fontsize=9
)

# Create 3D meshgrids from the provided `grid` array for x, y, and z coordinates respectively.
X_mesh, Y_mesh, Z_mesh = np.meshgrid(grid_1d_fine, grid_1d_fine, grid_1d_fine)

# Start from default 3D plotting settings, then override with base settings
pltSettings_3Dsigma = replace(PlotBasis3DSettings(), **pltSettings_base.__dict__)
fig_name = f'CovarianceMatrix_decayRate{model.decay_rate}_seed{seed}'

# Give this particular visualization a descriptive figure name and size
pltSettings_sigma3D = replace(pltSettings_3Dsigma, 
                              fig_name = fig_name,
                              fig_size = (8,8.5) #(9.5,7)
                              )
# Visualization helper for Wishart model quantities
visualize_sigma3D = WishartModelBasicsVisualization(save_fig=False,
                                                    settings=pltSettings_base,
                                                    save_format= 'png'
                                                    )
# Render U as a sequence of 2D slices through the 3D field
visualize_sigma3D.plot_basis_functions_3D(X_mesh,
                                          Y_mesh, 
                                          Z_mesh,
                                          Sigmas_grid_fine, 
                                          settings = pltSettings_sigma3D
                                          )

#save a gif
if visualize_sigma3D.save_format == 'png':
    PlottingTools.save_gif(fig_outputDir, 
                           gif_name=fig_name,
                           fig_name_start = fig_name, 
                           fig_name_end='.png'
                           )


#%%
# -----------------------------------------------------------
# SECTION 2b: look at the weighted sum (U)
# -----------------------------------------------------------
pltSettings_sigma3D = replace(pltSettings_3Dsigma, 
                              fig_name = 'U_given_sampledWeightMatrix',
                              fig_size = (9.5, 8)
                              )
visualize_sigma3D.save_fig = False

# Render U as a sequence of 2D slices through the 3D field
visualize_sigma3D.plot_basis_functions_3D(X_mesh,
                                        Y_mesh, 
                                        Z_mesh,
                                        U_fine, 
                                        settings = pltSettings_sigma3D
                                        )

#%% 
# -----------------------------------------------------------
# SECTION 3: visualize the internal noise ellipsoids in html
# -----------------------------------------------------------
pltSettings_html = Plot3DPredHTMLSettings()
pltSettings_html = replace(pltSettings_html, ell_scaler = 3) #enlarge it by a scaler of 3 for better visibility

# Visualization helper with HTML settings
vis_html = WishartPredictionsVisualization_html(settings=pltSettings_html)

fig1 = go.Figure()
# Render 3D ellipsoids (mesh surfaces) evaluated on the isoluminant plane
vis_html.plot_ellipsoids_mesh_cov(fig1, grid, Sigmas_grid)
# Apply consistent 3D layout (camera, axes, lighting, hover behavior)
vis_html.apply_3d_layout(fig1)
# Save interactive HTML
out_html = os.path.join(fig_outputDir, f"{fig_name}.html")
vis_html.write_html(fig1, out_html, include_plotlyjs=True)
