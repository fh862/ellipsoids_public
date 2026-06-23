#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:15:31 2026

@author: fangfang

This script loads a fitted Wishart model and Unity trial CSV files for one
participant, converts reference/comparison RGB stimuli into model space,
computes Mahalanobis distance for each trial, removes within-bin log-RT
outliers, fits an exponential decay model to log response time as a function
of distance, and visualizes the binned data together with the fitted curve.
"""

import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import dill as pickled
import numpy as np
np.random.seed(None)
import os
import pandas as pd
from copy import deepcopy
from scipy.optimize import curve_fit
from analysis.utils_load import select_file_and_get_path, extract_sub_number

#%%
# -----------------------------------------------------------
# SECTION 1: load model fits
# -----------------------------------------------------------    
# 'Experiment_DataFiles/pilot2/sub1/fits'
# 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub11_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl'
input_fileDir_fits, fits_file_name = select_file_and_get_path()
full_path_fits = os.path.join(input_fileDir_fits, fits_file_name)

# Load the fitted model and color-space transforms.
with open(full_path_fits, 'rb') as f:
    vars_dict = pickled.load(f)

# Helpers for transforming RGB into model space and evaluating the fit.
color_thres_data = vars_dict['color_thres_data']
model_pred = deepcopy(vars_dict['model_pred_Wishart'])
model = deepcopy(model_pred.model)
W_est = model_pred.W_est

#%%
# -----------------------------------------------------------
# SECTION 2: load response time
# -----------------------------------------------------------
# Example selected folder: ELPS_analysis/Experiment_DataFiles/pilot2/sub1
# 'Unity_trial_data_sub1_CH_session1__2025-02-25_17-41-28_copy.csv'
input_fileDir_csv, csv_file_name = select_file_and_get_path()
subN = extract_sub_number(csv_file_name)

# Find all trial CSVs for the selected participant in the same folder.
matching_csv_files = sorted(
    f for f in os.listdir(input_fileDir_csv)
    if f.startswith(f"Unity_trial_data_sub{subN}") and f.endswith(".csv")
)

print("Matching CSV files:")
if matching_csv_files:
    for csv_file in matching_csv_files:
        print(csv_file)
else:
    print("No matching CSV files found.")
    
# Accumulate reference stimuli, comparison stimuli, and response times.
xref_list, x1_list, RT_list = [], [], []

# Extract linear RGB triplets from strings like "R..._G..._B...".
def _extract_rgb_triplets(rgb_series):
    rgb_values = rgb_series.astype(str).str.extract(
        r'R(?P<R>[-\d.]+)_G(?P<G>[-\d.]+)_B(?P<B>[-\d.]+)'
    )
    return rgb_values.astype(float).to_numpy()

# Load and pool all trials across the participant's CSV files.
for mf in matching_csv_files:
    # Read one CSV and standardize column names.
    file_path = os.path.join(input_fileDir_csv, mf)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Append trial-wise reference RGB, comparison RGB, and RT.
    xref_list.append(_extract_rgb_triplets(df["Ref"]))
    x1_list.append(_extract_rgb_triplets(df["Comp"]))
    RT_list.append(pd.to_numeric(df["RT"], errors="coerce").to_numpy())

# Concatenate pooled trials into single arrays.
xref_rgb = np.vstack(xref_list) 
x1_rgb = np.vstack(x1_list) 
RT = np.concatenate(RT_list) 

print(f"Loaded {len(RT)} trials from {len(matching_csv_files)} CSV file(s).")

#%%
# -----------------------------------------------------------
# SECTION 3: compute mahalanobis distance
# -----------------------------------------------------------
# Map RGB stimuli into the 2D Wishart model space.
if model.num_dims == 2:
    xref_jnp = jnp.array(color_thres_data.rgb_to_2DW(xref_rgb))
    x1_jnp = jnp.array(color_thres_data.rgb_to_2DW(x1_rgb))
elif model.num_dims == 3:
    xref_jnp = color_thres_data.W_unit_to_N_unit(xref_rgb)
    x1_jnp = color_thres_data.W_unit_to_N_unit(x1_rgb)

# Evaluate the local noise covariance at each reference and comparison.
sigma_xref = model.compute_Sigmas(model.compute_U(W_est, xref_jnp))
sigma_x1 = model.compute_Sigmas(model.compute_U(W_est, x1_jnp))

# Compute trial-wise Mahalanobis distance predicted by the fitted model.
model_pred.compute_Mahalanobis_distance_batch(xref_jnp, x1_jnp, sigma_xref, sigma_x1)

# Sort distances so the RT relationship can be visualized more clearly.
sort_idx = np.argsort(np.asarray(model_pred.mahalanobis_distances))
dM_sorted = np.asarray(model_pred.mahalanobis_distances)[sort_idx]
logRT_sorted = np.log(RT[sort_idx])

# Bin Mahalanobis distance and reject extreme log-RT values within each bin.
bin_size = np.ceil(np.max(dM_sorted)) / 21
outlier_SD = 3

M = np.ceil(np.max(dM_sorted) / bin_size) * bin_size
bin_edges = np.arange(0, M, bin_size)

# Keep full, outlier-only, and outlier-removed copies for plotting and fitting.
dM_binned, dM_binned_outliers, dM_binned_rmO = [], [], []
logRT_binned, logRT_binned_outliers, logRT_binned_rmO = [], [], []
bin_ranges = []

for edge in bin_edges:
    # Select trials whose distances fall inside the current bin.
    idx_bin = (
        (dM_sorted >= edge) &
        (dM_sorted < edge + bin_size)
    )
    
    # Extract distances and log RT values for this bin.
    dM_bin = dM_sorted[idx_bin]
    logRT_bin = logRT_sorted[idx_bin]

    # Store the full contents of the bin.
    dM_binned.append(dM_bin)
    logRT_binned.append(logRT_bin)
    bin_ranges.append((edge, edge + bin_size))

    # Mark bin-wise log-RT outliers using a symmetric SD threshold.
    if logRT_bin.size == 0:
        idx_outlier = np.array([], dtype=bool)
    else:
        # Estimate the bin mean and spread from finite values.
        rt_mean = np.nanmean(logRT_bin)
        rt_std = np.nanstd(logRT_bin)
        if np.isnan(rt_std) or rt_std == 0:
            idx_outlier = np.zeros(logRT_bin.shape, dtype=bool)
        else:
            idx_outlier = np.abs(logRT_bin - rt_mean) > outlier_SD * rt_std

    # Save outliers separately for visualization.
    dM_binned_outliers.append(dM_bin[idx_outlier])
    logRT_binned_outliers.append(logRT_bin[idx_outlier])

    # Keep the non-outlier trials for model fitting.
    idx_keep = ~idx_outlier
    dM_binned_rmO.append(dM_bin[idx_keep])
    logRT_binned_rmO.append(logRT_bin[idx_keep])

#%%
# -----------------------------------------------------------
# SECTION 4: fit an exponential model
# -----------------------------------------------------------
# Exponential decay model for log RT as a function of Mahalanobis distance.
def RT_model(dM, t_0, a, b):
    return t_0 + a * np.exp(-b * dM)

# Pool all non-outlier trials across bins before fitting.
dM_fit = np.concatenate(dM_binned_rmO)
logRT_fit = np.concatenate(logRT_binned_rmO)
idx_finite = np.isfinite(dM_fit) & np.isfinite(logRT_fit)
dM_fit = dM_fit[idx_finite]
logRT_fit = logRT_fit[idx_finite]

# Fit the decay model with simple bounds for stability.
popt, _ = curve_fit(
    RT_model,
    dM_fit,
    logRT_fit,
    p0=[-1, 1, 1],
    bounds=([-10, 0, 1e-6], [10, 10, 10]),
    maxfev=20000,
)

t_0_fit, a_fit, b_fit = popt
logRT_pred = RT_model(dM_fit, *popt)

print("Fitted log RT(dM) = t_0 + a * exp(- b * dM)")
print(f"t_0 = {t_0_fit:.6f}")
print(f"a   = {a_fit:.6f}")
print(f"c   = {b_fit:.6f}")

#%%
# -----------------------------------------------------------
# SECTION 5: Visualize
# -----------------------------------------------------------
plt.rcParams["font.family"] = "Arial"
fig, ax = plt.subplots(1, 1, figsize=(6,4), dpi=1024)    
for idx, (m_o, rt_o, m, rt) in enumerate(zip(dM_binned_outliers,
                                             logRT_binned_outliers, 
                                             dM_binned_rmO, 
                                             logRT_binned_rmO
                                             )
                                         ):
    ax.scatter(m_o, rt_o, s = 5, alpha = 0.15, color = 'red', marker = '.', 
               label = 'outlier (> ± 3SD)' if idx == 0 else None)
    ax.scatter(m, rt, s = 5, alpha = 0.1, color = 'gray', marker = '.')

#plot separately so that the binned avg can be on top
for idx, (m, rt) in enumerate(zip(dM_binned_rmO, logRT_binned_rmO)):
    if m.size > 0:
        # Use larger markers for per-bin means, scaled by sample count.
        ax.scatter(np.mean(m), np.mean(rt), s = rt.shape[0]/15+5, alpha = 1, 
                   color = 'k', marker = 'o', edgecolor = 'white', 
                   label = 'binned average' if idx == 0 else None)

dM_plot = np.linspace(np.min(dM_fit), np.max(dM_fit), 400)
fit_label = (
    'fitted curve: $t_0 + a \cdot \\exp(-b \cdot d_M)$'
    f'\n$t_0={t_0_fit:.2f},\\ a={a_fit:.2f},\\ b={b_fit:.2f}$'
)
ax.plot(dM_plot, RT_model(dM_plot, *popt), color = 'g', linewidth = 2,
        label = fit_label)
ax.set_xlim([0, M])
ax.set_xticks(np.linspace(0, M, 5))
ax.set_ylim([np.min(logRT_fit)-1, np.max(logRT_fit)+1])
ax.set_xlabel('Mahalanobis distance between reference and comparison stimuli')
ax.set_ylabel('log Response time (s)')
legend = ax.legend(frameon = True, fontsize = 9, loc = 'lower right')
legend.legend_handles[0].set_sizes([100])
legend.legend_handles[1].set_sizes([100])
# Save the figure as a PDF
output_figDir_fits = os.path.join(deepcopy(input_fileDir_csv), 'trial_placement')
output_figDir_fits = output_figDir_fits.replace('DataFiles', 'FigFiles')
os.makedirs(output_figDir_fits, exist_ok=True)
fig.savefig(
    os.path.join(output_figDir_fits, f"RTs_sub{subN}.pdf"),
    format='pdf',
    bbox_inches='tight'
)
plt.show()
