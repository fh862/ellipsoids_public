#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:22:16 2026

@author: fangfang

This script analyzes how AEPsych distributes trials across stimulus directions
and whether the resulting performance matches the intended target level.

For each trial, we compute the direction from reference to comparison stimulus:
    d  = x1 - xref
    du = d / ||d||                          (unit direction vector)
    theta = atan2(du_y, du_x)               azimuth angle in [-180°, 180°]
    theta = mod(theta_raw, 2*pi)            azimuth angle in [0°, 360°]

We then:
  1. Bin trials by polar angle (theta) using evenly spaced edges from 0° to 360°.
  2. Count how many trials fall into each angular bin.
  3. Compute the mean percent-correct (pC) within each bin.

If AEPsych is placing trials efficiently around threshold, the binned pC
should be close to the target performance level (~66.7%) for all directions.
Deviations may indicate directional biases in trial placement or model behavior.

We perform this analysis separately for different trial types:
  - Initial Sobol trials (first 900 AEPsych trials; non-adaptive space-filling)
  - Adaptive EAVC trials (remaining 5100 trials; model-guided placement)
  - Pre-generated Sobol trials (used when waiting for AEPsych to return the next query)

Comparing these subsets helps reveal how adaptive sampling changes directional
coverage and whether it concentrates trials in specific regions of stimulus space.
    
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from analysis.utils_load import load_expt_data
from analysis.bin_pCorrect import BinnedPC

#%%
# -----------------------------------------------------------
# SECTION 1: set directories
# -----------------------------------------------------------
#specify the file name
subN = 11
stim_dims = 2
psyfield_dims = 4
nSessions = 12 #selected session

# Base directory where data lives. On HPC, prefer paths relative to the script.
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
path_str  = os.path.join(base_dir,'ELPS_analysis','Experiment_DataFiles',
                         'pilot2',f'sub{subN}') 
                         #4D_Expt_varyingBackground, 
                         #4D_Expt_dichromats
                         #pilot2

# -----------------------------------------------------------
# SECTION 2: Load and organize pilot data
# -----------------------------------------------------------
# Collect file paths for all sessions for this subject
session_files, session_file_name_part1 = load_expt_data.get_all_sessions_file_names(
    subN, nSessions, path_str,
    exptCond = '_4dExpt_Isoluminant plane',#'_4dExpt_LSisolating plane',
    #str_ext = '_gray_copy'
)

# Load raw session structs
data_allSessions = load_expt_data.load_data_all_sessions(session_files)

try:
    # Concatenate MOCS trials across sessions (if present)
    xref_MOCS_list, x1_MOCS_list, y_MOCS_list, xref_MOCS, x1_MOCS, y_MOCS = \
        load_expt_data.load_MOCS_data(data_allSessions)

    # Group MOCS trials by unique reference stimulus condition
    xref_unique_MOCS, nRefs_MOCS, refStimulus_MOCS, compStimulus_MOCS, \
        responses_MOCS, nLevels_MOCS, nTrials_MOCS, _, _, _ = \
            load_expt_data.org_MOCS_by_condition(xref_MOCS, x1_MOCS, y_MOCS)
except:
    print("MOCS trials not found in this dataset.")

# Concatenate AEPsych trials across sessions, and also load any pre-generated Sobol block
aepsych_data, sobol_data, combined_data = \
    load_expt_data.load_combine_AEPsych_pregSobol(data_allSessions)

xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list, \
    xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed = aepsych_data

# Total number of AEPsych trials
nTrials_AEPsych = y_AEPsych.shape[0]

# Number of trials per strategy block (from the session metadata).
# Convention here: all blocks except the last are the initial Sobol phase;
# the last block corresponds to the adaptive EAVC phase.
nTrials_strat = data_allSessions[0]["NTRIALS_STRAT"]
nTrials_aeSobol = sum(nTrials_strat[:-1])

# Convert per-session trial lists into a single (N, ndim) / (N,) array for easy slicing
xref_ae = np.concatenate(xref_AEPsych_list)
x1_ae = np.concatenate(x1_AEPsych_list)
y_ae = np.concatenate(y_AEPsych_list)

# -----------------------------
# Split AEPsych trials by type
# -----------------------------
# Initial Sobol block (non-adaptive / space-filling)
xref_aeSobol = xref_ae[:nTrials_aeSobol]
x1_aeSobol = x1_ae[:nTrials_aeSobol]
y_aeSobol = y_ae[:nTrials_aeSobol]

# Adaptive EAVC block (model-driven trial placement)
xref_aeEAVC = xref_ae[nTrials_aeSobol:]
x1_aeEAVC = x1_ae[nTrials_aeSobol:]
y_aeEAVC = y_ae[nTrials_aeSobol:]

# Some datasets include an additional pre-generated Sobol chunk (separate from AEPsych).
# If present, extract (xref, x1, y); otherwise set to None.
if sobol_data is not None:
    xref_pregenSobol, x1_pregenSobol, y_pregenSobol = sobol_data[-3:]
else:
    xref_pregenSobol, x1_pregenSobol, y_pregenSobol = None, None, None

#%%
# -----------------------------------------------------------
# SECTION 3: Direction-binning sanity check (pC vs azimuth theta)
# -----------------------------------------------------------
# Goal:
#   Check whether the trial placement is efficent by binning the percent correct.
#   If it's efficient, then the binned percent correct should be around the target
#   which in our case is 66.7%.
#
# What we do:
#   For each trial subset (AEPsych initial Sobol, AEPsych adaptive EAVC, and any
#   pre-generated Sobol block), we compute the direction from reference to
#   comparison for every trial:
#       d  = x1 - xref
#       du = d / ||d||                       (skip trials with ||d|| == 0)
#       theta = atan2(du_y, du_x)            azimuth in [-pi, pi)
#   Then we bin trials by theta using a chosen bin width (step_deg) and compute,
#   per bin:
#       - pC: mean(y) within that bin
#       - nTrials: number of trials in the bin
#       - idx: indices of trials assigned to the bin
#
# Note:
#   Different subsets can use different theta resolutions (e.g., finer bins for
#   EAVC if there are more trials / denser sampling).
step_deg = [10, 4, 10]

trial_type_str = ['AEPsych Sobol', 'AEPsych EAVC', 'Pregenerated Sobol']

# Trial subsets to analyze
xref_for_tests = [xref_aeSobol, xref_aeEAVC, xref_pregenSobol]
x1_for_tests   = [x1_aeSobol,   x1_aeEAVC,   x1_pregenSobol]
y_for_tests    = [y_aeSobol,    y_aeEAVC,    y_pregenSobol]

# Create one BinnedPC object per subset and run theta-binning
binner_list = []
for ii, (xref_ii, x1_ii, y_ii) in enumerate(zip(xref_for_tests, x1_for_tests, y_for_tests)):
    # Some datasets may not include the pre-generated Sobol block; skip if missing
    if xref_ii is None:
        continue

    # Initialize binner for this subset
    binner = BinnedPC(xref_ii, x1_ii, y_ii)

    # Define theta bin edges/centers (stored as attributes on the binner)
    # and compute pC / counts / indices per theta bin.
    binner.edges_theta_deg(step_deg[ii])
    binner.bin_2d()
    
    binner_list.append(binner)
        
#%% visualizse
output_fig_path = os.path.join(path_str.replace('Experiment_DataFiles',
                                                'Experiment_FigFiles'), 'binnedPC')
os.makedirs(output_fig_path, exist_ok= True)

for ii in range(len(xref_for_tests)):
    # Skip subsets that are not present (e.g., pre-generated Sobol might be None)
    if xref_for_tests[ii] is None:
        continue

    binner_ii = binner_list[ii]

    # Color mapping: encode nTrials per bin using a truncated colormap
    nTrials_bin = np.asarray(binner_ii.perBin_data["nTrials"], dtype=float)

    # Normalize nTrials so colormap spans [min, max] across bins
    norm = mpl.colors.Normalize(vmin=np.min(nTrials_bin),
                                vmax=np.max(nTrials_bin))

    # Use a truncated range of a base colormap to keep colors subtle
    base_cmap = plt.cm.PiYG_r
    cmap_trunc = mpl.colors.LinearSegmentedColormap.from_list(
        "bone_trunc",
        base_cmap(np.linspace(0.3, 0.5, 256))
    )

    # Map trial counts to RGBA colors, one per theta bin
    colors = cmap_trunc(norm(nTrials_bin))

    # Figure / polar axis
    fig = plt.figure(dpi = 1024)
    ax = fig.add_subplot(111, projection="polar")

    # Background bars: show bin occupancy (all bars same height, colored by nTrials)
    ax.bar(binner_ii.theta_edges_rad[:-1],
           np.ones_like(binner_ii.theta_edges_rad[:-1]),
           width=np.deg2rad(step_deg[ii]),
           bottom=0.0,
           align="edge",
           color=colors,
           edgecolor="none",
           zorder=0)

    # pC curve + markers at bin centers
    #   - line: pC vs theta
    #   - dots: size encodes nTrials per bin
    theta_centers = binner_ii.theta_centers_rad
    pC_bin = np.asarray(binner_ii.perBin_data["pC"], dtype=float)

    ax.plot(theta_centers, pC_bin, color="k", linewidth=1, zorder=2)

    # Dot size proportional to nTrials (area scaling).
    # Choose a reasonable visual range [s_min, s_max] and scale linearly.
    s_min, s_max = 5.0, 50.0
    n_min, n_max = np.nanmin(nTrials_bin), np.nanmax(nTrials_bin)
    if n_max > n_min:
        s = s_min + (nTrials_bin - n_min) / (n_max - n_min) * (s_max - s_min)
    else:
        s = np.full_like(nTrials_bin, (s_min + s_max) / 2.0)

    ax.scatter(theta_centers, pC_bin, color="k", s=s, zorder=3)

    # Optional: close the curve visually (connect last center back to first)
    ax.plot([theta_centers[0], theta_centers[-1]],
            [pC_bin[0], pC_bin[-1]],
            color="k", linewidth=1, zorder=2)

    # Axis formatting
    ax.set_rlim(0, 1)
    ax.set_rticks([0, 0.333, 0.667, 1.0])

    # Style the radial grid + border
    c_pink = np.array([125, 125, 125]) / 255
    ax.tick_params(axis="y", colors=c_pink)
    ax.yaxis.grid(True, color=c_pink, linewidth=1)
    ax.xaxis.grid(False)

    ax.spines["polar"].set_color(c_pink)
    ax.spines["polar"].set_linewidth(1.5)

    # Polar convention: zero at East, increasing counterclockwise
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(trial_type_str[ii])

    # Colorbar: trial count per bin
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_trunc)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.1, label="Trials per bin")
    plt.tight_layout()
    #save
    fig.savefig(os.path.join(output_fig_path, f'binned_pC_{trial_type_str[ii]}.pdf'))