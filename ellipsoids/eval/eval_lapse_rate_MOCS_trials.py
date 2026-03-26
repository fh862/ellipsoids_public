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
import os
from analysis.utils_load import load_expt_data, select_file_and_get_path, extract_sub_number

# ---------------------------------------------------------------------------
# Load Data from All Sessions
# ---------------------------------------------------------------------------
# Base directory where experiment data is stored
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

# Construct path to the subject's data directory
# e.g., ELPS_analysis/Experiment_DataFiles/6D_Expt/sub1'
# 'ColorDiscrimination_6dExpt_RGBcube_sub1_session1_copy.pkl'
# OR
# ELPS_analysis/Experiment_DataFiles/pilot2/sub1'
# 'ColorDiscrimination_4dExpt_Isoluminant plane_sub1_session1_copy.pkl'
input_fileDir, file_name = select_file_and_get_path()
full_path = os.path.join(input_fileDir, file_name)

#extract subject number
subN = extract_sub_number(file_name)

# Number of sessions to analyze (selected session)
# all subjects completed the same number of trials, but sub1 did them in less sessions
###################### NEED TO MODIFY ######################
ndims = 2 #2 or 3
nSessions = 12 #10 or 12 for 2D/4D; 40 for 3D/6D
############################################################
exptCond = '_6dExpt_RGBcube' if ndims == 3 else '_4dExpt_Isoluminant plane'

#retrieve all the session files
session_files, session_file_name_part1 = \
    load_expt_data.get_all_sessions_file_names(subN, 
                                               nSessions, 
                                               input_fileDir,
                                               exptCond = exptCond
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

    # Stack trial-wise values into arrays of shape (N_trials, ndims) or (N_trials, 1)
    xref_n = np.vstack([mocs_trials[trial]['xref'] for trial in range(num_mocs_trials)])
    x1_n   = np.vstack([mocs_trials[trial]['x1'] for trial in range(num_mocs_trials)])
    y_n    = np.vstack([mocs_trials[trial]['binaryResp'] for trial in range(num_mocs_trials)])

    # append to the list
    xref_all_list.append(xref_n)
    x1_all_list.append(x1_n)
    y_all_list.append(y_n)

# Convert session-wise lists into arrays:
#   xref_all, x1_all: (nSessions, nTrials, ndims)
#   y_all:            (nSessions, nTrials, 1)
xref_all = np.array(xref_all_list)
x1_all = np.array(x1_all_list)
y_all = np.array(y_all_list)

# ---------------------------------------------------------------------------
# Find the Euclidean distance between each ref and its associated catch trial
# ---------------------------------------------------------------------------
# Flatten across sessions so we can identify unique reference stimuli
xref_all_flat = xref_all.reshape(-1, ndims)
x1_all_flat = x1_all.reshape(-1, ndims)

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
pC_easy_trials = np.full((nSessions,), np.nan)      # proportion correct
num_easy_trials = np.full((nSessions,), np.nan)     # number of catch trials
num_correct_trials = np.full((nSessions,), np.nan)  # number correct on catch trials

# Loop over each session
for session_idx in range(nSessions):
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

