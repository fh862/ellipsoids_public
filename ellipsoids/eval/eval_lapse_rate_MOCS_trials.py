#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:35:29 2025

@author: fangfang

The goal of this short script is to compute the **percent correct (pC)**
for easy (catch) trials. 

In the experiment reported in the eLife paper, each session consists of **600 
or 500 trials, with 50 or 41-42 catch trials.

In the experiment that measures the entire 3D RGB cube, every two sessions consist
of 20 catch trials. There were 400 catch trials in total.

If the computed percent correct (pC) for catch trials falls below a 
specified cutoff (default: 0.9), we may consider excluding that subject.

"""

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from analysis.utils_load import load_expt_data 
#from dconfig.config_6Ddata import DatasetConfig_6D
from dconfig.config_4Ddata import DatasetConfig_4D
from analysis.utils_load import get_path

# ---------------------------------------------------------------------------
# Load Data from All Sessions
# ---------------------------------------------------------------------------
# Base directory where experiment data is stored
base_dir = get_path("dropbox_root_mac")

subN = 2

# choose dataset
#dcfg = DatasetConfig_6D.human_fullcube(base_dir, subN)
dcfg = DatasetConfig_4D.human_isoluminant(base_dir, subN)

#retrieve all the session files
session_files, session_file_name_part1 = \
    load_expt_data.get_all_sessions_file_names(subN, 
                                               dcfg.nSession, 
                                               dcfg.path_str,
                                               exptCond = dcfg.exptCond
                                               )

# Load session data from the files
data_allSessions = load_expt_data.load_data_all_sessions(session_files)

# ---------------------------------------------------------------------------
# Extract MOCS trials
# ---------------------------------------------------------------------------
xref_all_list = []   # reference stimuli
x1_all_list = []     # comparison stimuli
y_all_list = []      # binary responses

# Extract xref, x1, and binary response from each session
for session_data in data_allSessions:
    mocs_trials = session_data['sim_interleaved_trial_sequence'].data_MOCS
    num_mocs_trials = len(mocs_trials)

    # Stack trial-wise values into arrays of shape (N_trials, dcfg.stim_dims) or (N_trials, 1)
    xref_n = np.vstack([mocs_trials[trial]['xref'] for trial in range(num_mocs_trials)])
    x1_n   = np.vstack([mocs_trials[trial]['x1'] for trial in range(num_mocs_trials)])
    y_n    = np.vstack([mocs_trials[trial]['binaryResp'] for trial in range(num_mocs_trials)])

    # append to the list
    xref_all_list.append(xref_n)
    x1_all_list.append(x1_n)
    y_all_list.append(y_n)

# Convert session-wise lists into arrays:
#   xref_all, x1_all: (nSessions, nTrials, dcfg.stim_dims)
#   y_all:            (nSessions, nTrials, 1)
xref_all = np.array(xref_all_list)
x1_all = np.array(x1_all_list)
y_all = np.array(y_all_list)

# ---------------------------------------------------------------------------
# Find the Euclidean distance between each ref and its associated catch trial
# ---------------------------------------------------------------------------
# Flatten across sessions so we can identify unique reference stimuli
xref_all_flat = xref_all.reshape(-1, dcfg.stim_dims)
x1_all_flat = x1_all.reshape(-1, dcfg.stim_dims)

# Unique reference stimuli across all MOCS trials
unique_xref = np.unique(xref_all_flat, axis=0)

# For each reference stimulus, store the largest x1-xref distance observed
# across all sessions. This defines the catch/easy trial for that reference.
l2_norm_easy_trials = np.full((unique_xref.shape[0],), np.nan)

for idx, ref_stimulus in enumerate(unique_xref):
    # Find all trials using this reference stimulus
    matching_indices = np.where(np.all(xref_all_flat == ref_stimulus, axis=1))[0]

    # Comparison stimuli paired with this reference
    x1_subset = x1_all_flat[matching_indices]

    # Euclidean distance between reference and comparison
    l2_norm_differences = np.linalg.norm(x1_subset - ref_stimulus, axis=1)

    # Define the catch trial as the largest available difference
    l2_norm_easy_trials[idx] = np.max(l2_norm_differences)

# ---------------------------------------------------------------------------
# Compute catch-trial performance for each session
# ---------------------------------------------------------------------------
# Per-session summary statistics for catch trials
pC_easy_trials = np.full((dcfg.nSession,), np.nan)      # proportion correct
num_easy_trials = np.full((dcfg.nSession,), np.nan)     # number of catch trials
num_correct_trials = np.full((dcfg.nSession,), np.nan)  # number correct on catch trials

# Loop over each session
for session_idx in range(dcfg.nSession):
    # Responses from trials identified as catch/easy trials in this session
    y_easy_trials = []

    # Loop over each reference stimulus and find its catch trial(s)
    for r, ref_stimulus in enumerate(unique_xref):
        # Trials in this session with the current reference stimulus
        matching_indices = np.where(
            np.all(xref_all[session_idx] == ref_stimulus, axis=1)
        )[0]

        # Comparison stimuli for those trials
        x1_subset = x1_all[session_idx, matching_indices]

        # Distances from comparison to reference
        l2_norm_differences_n = np.linalg.norm(x1_subset - ref_stimulus, axis=1)

        # Identify trial(s) whose distance matches the precomputed catch-trial distance
        max_l2_norm_indices = np.where(
            l2_norm_differences_n == l2_norm_easy_trials[r]
        )[0]

        # Append the corresponding binary responses
        y_easy_trials.extend(
            y_all[session_idx, matching_indices[max_l2_norm_indices]].flatten().tolist()
        )

    # Summarize catch-trial performance for this session
    num_correct_trials[session_idx] = np.sum(y_easy_trials)
    num_easy_trials[session_idx] = len(y_easy_trials)

    # print out the results
    if num_easy_trials[session_idx] != 0:
        pC_easy_trials[session_idx] = (
            num_correct_trials[session_idx] / num_easy_trials[session_idx]
        )
        print(
            f'Session {session_idx+1}: Catch trial pC = '
            f'{int(num_correct_trials[session_idx])}/'
            f'{int(num_easy_trials[session_idx])} = '
            f'{pC_easy_trials[session_idx]:.4f}'
        )
    else:
        print(f'Session {session_idx+1} does not include any Sobol catch trials.')

# ---------------------------------------------------------------------------
# Aggregate catch-trial performance across sessions
# ---------------------------------------------------------------------------
num_correct_total = np.nansum(num_correct_trials)
pC_easy_trials_avg = num_correct_total / np.nansum(num_easy_trials)

print(
    f'All sessions: Catch trial pC = '
    f'{num_correct_total}/{np.sum(num_easy_trials)} = {pC_easy_trials_avg:.4f}'
)
print(f'Range: [{np.nanmin(pC_easy_trials):.4f}, {np.nanmax(pC_easy_trials):.4f}]')

