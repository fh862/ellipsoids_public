#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 15:32:26 2025

@author: fangfang

Pre-generate Sobol trials for color discrimination experiments.

Purpose
-------
This script generates Sobol-sampled stimulus pairs (reference and comparison)
to be used as fallback trials when AEPsych sampling is slow or unavailable.

Key features
------------
- Supports multiple experiment configurations via `PregenSobolConfig`
- Generates Sobol samples within specified bounds
- Applies scaling factors to control trial difficulty
- Optionally inserts catch trials with fixed offsets
- Saves results in a backward-compatible pickle format

Outputs
-------
A pickle file containing:
    - Sobol_xref: reference stimuli
    - Sobol_x1: comparison stimuli
    - catch trial indices and values (if enabled)
    - config parameters (via legacy dict)

Notes
-----
- More trials are generated than needed; unused trials are simply ignored.
- Designed to integrate with existing analysis pipelines without modification.

"""

import matplotlib.pyplot as plt
import dill as pickled
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import sys
import os
script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from analysis.color_thres import color_thresholds
from analysis.MOCS_thresholds import sim_MOCS_trials
from dconfig.config_pregenSobol import PregenSobolConfig


# --------------------------------------------------------------------------
# SECTION 1: Configuration and Sobol Trial Generation
# --------------------------------------------------------------------------

# Select experiment configuration (only one active at a time)

# scfg = PregenSobolConfig.rgbcube_3D_dichromat()
# scfg = PregenSobolConfig.isoluminant_2D4D()
# scfg = PregenSobolConfig.LSisolating_dichromat()
# scfg = PregenSobolConfig.LSisolating_dichromat_expanded()
# scfg = PregenSobolConfig.adaptation_round1()
# scfg = PregenSobolConfig.adaptation_round2()

scfg = PregenSobolConfig.adaptation_round2()

# Print configuration summary for sanity check
scfg.print_summary()

# Base directory for data and calibration files
baseDir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

# Initialize color transformation helper
color_thres_data = color_thresholds(
    scfg.stim_dims,
    baseDir,
    plane_2D=scfg.plane_2D
)

# Load calibration / transformation matrix (only for 2D experiments)
if scfg.stim_dims == 2:
    color_thres_data.load_transformation_matrix(file_date=scfg.file_date)

# Random seed for reproducibility across sessions
# This seed is usually 100 * subN or 10000 * subN. 
# It can be found in our preregistration documents
sobol_seed = 1500

# Preallocate arrays for Sobol-generated stimuli
Sobol_xref = np.full(
    (scfg.nSessions, scfg.nTrials_sobol_perSession, scfg.stim_dims),
    np.nan
)
Sobol_x1 = np.full_like(Sobol_xref, np.nan)

# Toggle visualization of generated trials
flag_debugplots = True

# Catch trial setup (optional)
if scfg.flag_addCatchTrials:
    # number of total catch trials
    nTotal_catchTrials = int(scfg.nTrials_sobol_perSession * scfg.percent_catchTrials)

    # Preallocate arrays for catch trial bookkeeping
    catch_idx_all = np.full((scfg.nSessions, nTotal_catchTrials), np.nan)
    choice_unique_catch_all = np.full_like(catch_idx_all, np.nan)
    delta_catch_all = np.full(
        (scfg.nSessions, nTotal_catchTrials, scfg.stim_dims),
        np.nan
    )

# Main loop: generate Sobol trials per session
for n in range(scfg.nSessions):

    # Independent RNG per session for reproducibility
    rng = np.random.default_rng(sobol_seed + n)

    # Generate Sobol samples in specified bounds
    Sobol_samples = sim_MOCS_trials.sample_sobol(
        scfg.nTrials_sobol_perSession,
        lb=scfg.lb_sobol_trials,
        ub=scfg.ub_sobol_trials,
        force_center=False,
        seed=sobol_seed + n
    )

    # Shuffle scaling factors (ensures balanced difficulty)
    sobol_scaler_n = np.concatenate([
        rng.permutation(scfg.sobol_scaler)
        for _ in range(scfg.num_repeats)
    ])

    # Construct reference (xref) and comparison (x1) stimuli
    if scfg.psyfield_dims in [4, 6]:
        # Split Sobol samples into reference and direction components
        Sobol_xref[n] = Sobol_samples[:, :scfg.stim_dims]
        Sobol_x1[n] = (
            Sobol_xref[n]
            + sobol_scaler_n[:, None] * Sobol_samples[:, scfg.stim_dims:]
        )
    else:
        # Direct scaling (e.g., 3D RGB cube)
        Sobol_xref[n] = 0
        Sobol_x1[n] = Sobol_xref[n] + sobol_scaler_n[:, None] * Sobol_samples

    # Catch trials: replace subset of trials with fixed offsets
    if scfg.flag_addCatchTrials and nTotal_catchTrials > 0:

        # Randomly select trial indices to replace
        catch_idx = rng.choice(
            scfg.nTrials_sobol_perSession,
            size=nTotal_catchTrials,
            replace=False
        )
        catch_idx_all[n] = catch_idx

        # Randomly select predefined delta offsets
        choose = rng.integers(
            0, len(scfg.delta_catchTrials_unique),
            size=nTotal_catchTrials
        )
        choice_unique_catch_all[n] = choose

        delta_catch = scfg.delta_catchTrials_unique[choose]
        delta_catch_all[n] = delta_catch

        # Overwrite comparison stimuli for catch trials
        Sobol_x1[n, catch_idx] = Sobol_xref[n, catch_idx] + delta_catch

    # Debug visualization (optional)
    if flag_debugplots:

        if scfg.stim_dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot reference and comparison points
            for m in range(scfg.nTrials_sobol_perSession):
                ax.scatter(*Sobol_xref[n, m], marker='.', s=5,
                           c=color_thres_data.W_unit_to_N_unit(Sobol_xref[n, m]))
                ax.scatter(*Sobol_x1[n, m], marker='o', s=5,
                           c=color_thres_data.W_unit_to_N_unit(Sobol_x1[n, m]))

            # Draw connecting lines (stimulus differences)
            for m in range(scfg.nTrials_sobol_perSession):
                ax.plot(
                    *[[Sobol_xref[n, m, i], Sobol_x1[n, m, i]]
                      for i in range(scfg.stim_dims)],
                    c=color_thres_data.W_unit_to_N_unit(Sobol_xref[n, m]),
                    alpha=0.5
                )

            plt.show()

        elif scfg.stim_dims == 2:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

            for m in range(scfg.nTrials_sobol_perSession):
                rgb = np.clip(color_thres_data.W2D_to_rgb(Sobol_xref[n, m]), 0, 1)
                ax.plot(
                    *[[Sobol_xref[n, m, i], Sobol_x1[n, m, i]]
                      for i in range(scfg.stim_dims)],
                    c=rgb, alpha=0.5, linewidth=0.6
                )

            # Highlight catch trials in black
            if scfg.flag_addCatchTrials and nTotal_catchTrials > 0:
                ax.plot(
                    *[[Sobol_xref[n, catch_idx, i], Sobol_x1[n, catch_idx, i]]
                      for i in range(scfg.stim_dims)],
                    c='k', alpha=0.5, linewidth=0.8
                )

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.show()

#%%
# --------------------------------------------------------------------------
# SECTION 2: Save generated data
# --------------------------------------------------------------------------
# Determine color space label for filename
cspace = (
    'RGBcube'
    if scfg.stim_dims == 3
    else color_thres_data.plane_2D.replace(" ", "_")
)

# Construct output filename
output_file = (
    f'Sim{scfg.psyfield_dims}dTask_colorDiscrimination_{cspace}_'
    f'pregeneratedSobol_seed{sobol_seed}.pkl'
)

# Output directory
output_fileDir = os.path.join(
    baseDir,
    'ELPS_analysis',
    'Simulation_DataFiles',
    f'{scfg.stim_dims}D',
    'pregenerated_Sobol'
)
os.makedirs(output_fileDir, exist_ok=True)

full_path2 = os.path.join(output_fileDir, output_file)

# Build dictionary in legacy-compatible format
vars_dict = scfg.to_legacy_dict()

# Append generated arrays
vars_dict.update({
    'Sobol_xref': Sobol_xref,
    'Sobol_x1': Sobol_x1,
    'catch_idx_all': catch_idx_all if scfg.flag_addCatchTrials else None,
    'choice_unique_catch_all': choice_unique_catch_all if scfg.flag_addCatchTrials else None,
    'delta_catch_all': delta_catch_all if scfg.flag_addCatchTrials else None,
})

# Save to pickle
with open(full_path2, 'wb') as f:
    pickled.dump(vars_dict, f)