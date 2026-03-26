#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:13:50 2024

@author: fangfang

This script demonstrates (and sanity-checks) a dimension-agnostic construction
of Chebyshev polynomial basis functions in 1D, 2D, and 3D in the model space
bounded between [-1, 1].

Part 1 — Chebyshev basis visualization
  • Build a 1D grid (lb..ub with nbins samples) and evaluate the first `degree`
    Chebyshev polynomials T_0..T_{degree-1}.
  • Extend to 2D and 3D by forming tensor-product bases:
        φ_2D(x,y) = T_i(x) T_j(y)
        φ_3D(x,y,z) = T_i(x) T_j(y) T_k(z)
  • Visualize the basis functions using the plotting helpers in
    `WishartModelBasicsVisualization` (1D curves, 2D surfaces, 3D slices/volumes).

Part 2 — Wishart-process prior weights and order-dependent structure
  • Instantiate `WishartProcessModel` with the hyperparameters used in the eLife
    paper (variance scale, geometric decay rate, etc.) and sample an initial
    weight tensor W from the Wishart-process prior.
  • Compute `basis_orders`, where each tensor-product term is assigned its
    *total* polynomial order:
        order(i,j,k) = i + j + k
  • Visualize (i) a selected slice of W and (ii) the distribution of sampled
    weights aggregated by polynomial order, alongside the prior ±2 SD envelope:
        SD(order) = 2 * sqrt(variance_scale * decay_rate^order)

"""

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import replace
from core import chebyshev
from core.wishart_process import WishartProcessModel
from plotting.wishart_plotting import WishartModelBasicsVisualization, PlotSettingsBase,\
    PlotBasis1DSettings, PlotBasis2DSettings, PlotBasis3DSettings, PlotWSettings,\
    PlotWAllSettings

# Set the output directory for figures
baseDir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'
                        
#%% 
# ---------------------------------------------------------------------
# Visualize the chebyshev basis functions (1D & 2D)
# ---------------------------------------------------------------------
# Degree of the Chebyshev polynomial
degree = 5 
# Define the bounds and number of bins for the grid
lb, ub, nbins = -1, 1, 41
# Generate a linear grid
grid = np.linspace(lb, ub, nbins)
# Get basis coefficients for Chebyshev polynomials
basis_coeffs = chebyshev.chebyshev_basis(degree)
 # Evaluate Chebyshev polynomials at the grid points
lg = chebyshev.evaluate(basis_coeffs, grid)

# Create settings instance with custom fig_dir
fig_basis_dir = os.path.join(baseDir,
                             'WishartPractice_FigFiles',
                             'Chebyshev_basis_functions'
                             )
pltSettings_base = PlotSettingsBase(fig_dir = fig_basis_dir)
os.makedirs(fig_basis_dir, exist_ok=True)

# Specialize per plot type
pltSettings_1D = replace(PlotBasis1DSettings(), **pltSettings_base.__dict__)
visualize_basis = WishartModelBasicsVisualization(settings=pltSettings_base,
                                                  save_format= 'pdf',
                                                  save_fig=True
                                                  )
# 1D Chebyshev basis functions
visualize_basis.plot_basis_function_1d(degree,
                                       grid,
                                       lg, 
                                       settings = pltSettings_1D
                                       )

# 2D Chebyshev basis functions
pltSettings_2D = replace(PlotBasis2DSettings(), **pltSettings_base.__dict__)
visualize_basis.plot_basis_function_2D(degree, 
                                       grid, 
                                       settings=pltSettings_2D
                                       )

#%%
# ---------------------------------------------------------------------
# Visualize the chebyshev basis functions (3D)
# ---------------------------------------------------------------------
# Create 3D meshgrids from the provided `grid` array for x, y, and z coordinates respectively.
X_mesh, Y_mesh, Z_mesh = np.meshgrid(grid, grid, grid)
# Stack the flattened x, y, and z arrays into a single matrix and then transpose it,
# resulting in a matrix where each row represents a point (x, y, z) in the 3D space.
XYZ_mesh = np.transpose(np.stack((X_mesh.flatten(), 
                                  Y_mesh.flatten(), 
                                  Z_mesh.flatten())),(1,0)
                        )

# Evaluate Chebyshev polynomials for each coordinate (x, y, z) and compute their outer product.
# This results in a high-dimensional array where each element represents the product of
# Chebyshev polynomial values at each point in the 3D space.
phi = (chebyshev.evaluate(basis_coeffs, XYZ_mesh[:,0])[:,:,None,None] *\
       chebyshev.evaluate(basis_coeffs, XYZ_mesh[:,1])[:,None,:,None] *\
       chebyshev.evaluate(basis_coeffs, XYZ_mesh[:,2])[:,None,None,:])

# Reshape the resulting tensor `phi` to have dimensions corresponding to bins and degrees
# for x, y, z coordinates, and polynomial degrees, making it easier to handle and interpret.
phi_org = np.reshape(phi, (nbins, nbins, nbins, degree, degree, degree))

#update the plotting data class
pltSettings_3D = replace(PlotBasis3DSettings(), **pltSettings_base.__dict__)

# Loop through each degree of the Chebyshev polynomials.
for i in range(degree): 
    current_settings = replace(pltSettings_3D, 
                               fig_name=f'Chebyshev_3D_basis_function_degree{i+1}')
    # Plot 3D basis functions for each degree using the calculated values in `phi_org`,
    # and save the plots and GIFs to a specified directory.
    visualize_basis.plot_basis_functions_3D(X_mesh, 
                                            Y_mesh, 
                                            Z_mesh,
                                            phi_org[...,i], 
                                            settings = pltSettings_3D
                                            )

#%% 
# ---------------------------------------------------------------------
# Visualize weights
# ---------------------------------------------------------------------

# Stimulus lives in a 3D space 
ndims = 3
# let the cov dimension matches the stimulus dimensionality
ndims_cov = 3
model = WishartProcessModel(
    5,      # Degree of the polynomial basis functions
    3,      # Number of stimulus dimensions
    1,      # Number of extra inner dimensions in `U`.
    3e-4,   # Scale parameter for prior on `W`.
    0.4,    # Geometric decay rate on `W`. default = 0.4
    0,      # Diagonal term setting minimum variance for the ellipsoids.
    num_dims_cov = ndims_cov
)

# specify a seed
seed = 201
W_INIT_KEY = jax.random.PRNGKey(seed)  # Key to initialize `W_est`. 
OPT_KEY = jax.random.PRNGKey(seed)

# Sample an initial weight field from the prior
W_init = model.sample_W_prior(W_INIT_KEY) 
        
# basis_orders[d1, d2, d3] = total polynomial order at that grid location
basis_orders = (
    jnp.arange(degree)[:, None, None] +
    jnp.arange(degree)[None, :, None] + 
    jnp.arange(degree)[None, None, :]
)

# Set up output directory and plotting settings for W slices
fig_W_dir = os.path.join(baseDir,
                         'WishartPractice_FigFiles',
                         'Estimated_W_matrix'
                         )
pltSettings_base_W = PlotSettingsBase(fig_dir = fig_W_dir, fontsize = 9)
pltSettings_W = replace(PlotWSettings(), **pltSettings_base_W.__dict__)

# Visualize one slice of the weight tensor W_init
visualize_basis.plot_W_selected_slice(W_init,
                                      settings = pltSettings_W,
                                      basis_orders= basis_orders,
                                      slc_slice = [3]
                                      )

# Expand basis_orders to match the last two dimensions of W:
# shape → (degree, degree, degree, num_dims_cov, num_dims_cov + extra_dims)
basis_orders_full = np.tile(basis_orders[...,None, None],
                      (1,1,1, model.num_dims_cov, model.num_dims_cov + model.extra_dims))

#max polynomial order
max_order = np.max(basis_orders)
prior_weight_sd = 2*np.sqrt(model.variance_scale * (model.decay_rate ** np.arange(0,max_order+1)))
            
# Plot settings for the “weights vs polynomial order” summary figure           
pltSettings_W_all = replace(PlotWAllSettings(), **pltSettings_base_W.__dict__)
pltSettings_W_all = replace(pltSettings_W_all,
                            xlabel = f'Polynomial order of the {model.num_dims}D chebyshev basis function',
                            ybds = [-0.04, 0.04],
                            yticks = np.linspace(-0.04, 0.04, 5)
                            )
fig, ax = plt.subplots(1, 1, figsize= pltSettings_W_all.fig_size, 
                       dpi = pltSettings_W_all.dpi)

# Plot prior ±1 SD envelope for comparison
ax.plot(np.arange(0,max_order+1), prior_weight_sd, color = 'k', ls = '--', 
        label = 'Prior on weights (2 SD)')
ax.plot(np.arange(0,max_order+1), -prior_weight_sd, color = 'k', ls = '--')

# Overlay empirical weights (W_init) aggregated by maximum polynomial order
visualize_basis.plot_W_all(W_init,
                           basis_orders= basis_orders_full,
                           settings = pltSettings_W_all,
                           ax = ax
                           )
