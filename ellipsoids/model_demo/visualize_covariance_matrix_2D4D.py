#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:37:26 2024

@author: fangfang

Goal
----
Visualize the field of covariance matrices representing the internal noise that
limits performance in a 2D color-discrimination task.

Context
-------
In this 2D/4D threshold experiment, both the reference and comparison stimuli
vary within a 2D chromatic plane.

Stimulus space (2D):
    dim1, dim2 — chromatic coordinates of the reference and comparison stimuli
                 within the chosen 2D plane.

Covariance field (4D parameterization):
    dim1, dim2 — coordinates of the reference stimulus on the plane
    dim3, dim4 — chromatic offsets applied to the reference to obtain the
                 comparison stimulus (i.e., Δ values defining the comparison direction)

Interpretation
--------------
The underlying Chebyshev basis is defined over a 2D space, and at each reference
location it yields a 2×2 covariance matrix.

For visualization tools tailored to other experiment designs (e.g., 3D/6D threshold
                                                              mappings), see:
    - Visualize_covariance_matrix_3D6D.py
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
    PlotSettingsBase, PlotCovMatSettings, PlotUSettings, PlottingTools
    
#%%
# -----------------------------------------------------------
# SECTION 1: Randomly draw from prior (2D)
# -----------------------------------------------------------
ndims = 2
# Number of grid points in the coarse (for ellipses) and fine (for heatmaps) grids.
num_grid_pts      = 7
num_grid_pts_fine = 100  

# PRNG key for sampling from the Wishart-process prior over W.
W_INIT_KEY   = jax.random.PRNGKey(221)

# Initialize a 2D WishartProcessModel with ndims×ndims covariance matrices.
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions
    ndims, # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    3e-4,  # Scale parameter for prior on `W`.
    0.3,   # decay rate on `W` 
    0,     # Diagonal term setting minimum variance for the ellipsoids.
    num_dims_cov= ndims
)

# Coarse grid of points in the 2D stimulus space (for plotting ellipses).
grid = jnp.stack(jnp.meshgrid(*[jnp.linspace(-0.7, 0.7, num_grid_pts) \
                                 for _ in range(model.num_dims)]), axis=-1)
    
grid_fine = jnp.stack(jnp.meshgrid(*[jnp.linspace(-0.7, 0.7, num_grid_pts_fine) \
                                 for _ in range(model.num_dims)]), axis=-1)
    
# Sample Chebyshev coefficients W from the prior.
W_init = model.sample_W_prior(W_INIT_KEY)

# Evaluate the weighted sum U on the fine and coarse grid.
U_fine = model.compute_U(W_init, grid_fine)
U = model.compute_U(W_init, grid)

# Compute covariance matrices Σ(x) on the fine and coarse grids.
Sigmas_test_grid_fine = model.compute_Sigmas(U_fine)
Sigmas_test_grid = model.compute_Sigmas(U)

#%%
# -----------------------------------------------------------
# SECTION 2: Visualize 2D covariance matrix
# -----------------------------------------------------------
# Base directory for saving figures
baseDir = get_path("dropbox_root_mac")

# Directory for all Wishart practice figure files
fig_output_sim = os.path.join(baseDir, 'ELPS_analysis', 'WishartPractice_FigFiles')

# Subdirectory specific to this covariance-matrix visualization
fig_outputDir = os.path.join(fig_output_sim, f'CovarianceMatrix_{ndims}D{ndims * 2}D')
os.makedirs(fig_outputDir, exist_ok=True)

# Initialize base plotting settings (shared across different plot types)
pltSettings_base = PlotSettingsBase(
    fig_dir=fig_outputDir,
    fontsize=9
)

# Specialize plotting settings for covariance-matrix figures
pltCovSettings = replace(PlotCovMatSettings(), **pltSettings_base.__dict__)

# Visualization helper for Wishart model covariances
visualize_sigma2D = WishartModelBasicsVisualization(
    save_fig=True,
    save_format='png',
    settings=pltCovSettings
)

# Titles for the 2×2 covariance components:
#   [0,0]: σ_dim1², [0,1] and [1,0]: σ_dim1,dim2, [1,1]: σ_dim2²
ttl_list = [
    [r'$\sigma^2_{\mathrm{dim1}}$', r'$\sigma_{\mathrm{dim1},\mathrm{dim2}}$'],
    [r'$\sigma_{\mathrm{dim1},\mathrm{dim2}}$', r'$\sigma^2_{\mathrm{dim2}}$']
]

#figure name
fig_name = f'CovarianceMatrix_decayRate{model.decay_rate}_seed{W_INIT_KEY[1]}'

# Loop over coarse grid locations and plot
for p in range(num_grid_pts):
    for q in range(num_grid_pts):
        pltCovSettings_randW = replace(
            pltCovSettings,
            slc_idx_dim1=p,
            slc_idx_dim2=q,
            cmap_bds=[-0.003, 0.003],
            ticks_W=np.around(np.linspace(-0.7, 0.7, 3), 1),
            cmap_ell='k',
            heatmap_title_list=ttl_list,
            covMat_title='2D plane',
            scaler_ell=5, #enlarge it by 5 times for visualization
            fig_name=fig_name,
            plane_2D='2D plane'
        )

        # Plot heatmaps of Σ(x) components (fine grid) and ellipses (coarse grid)
        visualize_sigma2D.plot_2D_covMat(
            grid,
            Sigmas_test_grid_fine,
            Sigmas_test_grid,
            settings=pltCovSettings_randW
        )
        
if visualize_sigma2D.save_format == 'png':
    PlottingTools.save_gif(fig_outputDir, 
                           gif_name=fig_name,
                           fig_name_start = fig_name, 
                           fig_name_end='.png'
                           )

#%%
# -----------------------------------------------------------
# SECTION 2b: look at the weighted sum (U)
# -----------------------------------------------------------
pltUSettings = replace(PlotUSettings(), **pltSettings_base.__dict__)
pltUSettings = replace(pltUSettings,
                       cmap_bds = [-0.05, 0.05],
                       ticks = np.linspace(-0.7, 0.7, 3),
                       fig_name = 'U_given_sampledWeightMatrix',
                       fig_name_ext = f'_seed{W_INIT_KEY[1]}'
                       )
visualize_sigma2D.fig_dir = os.path.join(fig_output_sim, 'U_given_estimatedWeightMatrix')
os.makedirs(visualize_sigma2D.fig_dir, exist_ok=True)
visualize_sigma2D.plot_U_2D(U_fine, settings = pltUSettings)
