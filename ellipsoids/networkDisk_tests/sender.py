# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:57:13 2025

@author: brainardlab-adm
"""

import sys
import os
import numpy as np
import random
import pickle
sys.path.append('c:\\users\\brainardlab\\documents\\github\\ellipsoids\\ellipsoids')
from analysis.utils_communication import CommunicateViaTextFile,\
    ExperimentFileManager, get_experiment_info_custom, get_comment_after_session

#%% Prompt the user for experiment information
# Use a custom Tkinter-based popup to collect subject ID, initials, and session number
subject_id, subject_init, session_today = get_experiment_info_custom()

# Define the main shared network disk path where files will be stored
networkDisk_path = 'c:\Shares\BrainardLab'
# Create the path for the subject's directory
path_sub = os.path.join(networkDisk_path, f'sub{subject_id}')

# Attempt to load the experiment manager state from a pickle file
try:
    # Define the metadata file name and full path
    expt_info = f'sub{subject_id}_{subject_init}_expt_record.pkl'
    path_metadata = os.path.join(path_sub, expt_info)
    # Load the existing state of the experiment file manager
    expt_file_manager = ExperimentFileManager.load_state(path_metadata)
except:
    # If loading fails (e.g., file not found), initialize a new ExperimentFileManager
    expt_file_manager = ExperimentFileManager(subject_id, 
                                              subject_init,
                                              networkDisk_path)
# Create a new session file for the current session
file_path, file_name = expt_file_manager.create_session_file(session_today)
# List all files created for this subject
expt_file_manager.list_files()

#%% Initialize communication class
communicator = CommunicateViaTextFile(expt_file_manager.path_sub)
communicator.check_and_handle_file(file_name)

# Step 1: Initialize
print("Initializing communication...")
communicator.initialize_communication()
print("Initialization complete.")
#update the communication status
expt_file_manager.status_updates('Confirmed')

#generate random RGB values or load
flag_load_rgb = True
if not flag_load_rgb:
    # Step 2: Send 10 sets of RGB values
    # Generate MOCS and AEPsych trial types
    MOCS_trial_type = [f'MOCS_{i}' for i in range(1, 6)]
    AEPsych_trial_type = [f'AEPsych_{i}' for i in range(1, 6)]
    trial_type_both = MOCS_trial_type + AEPsych_trial_type
    random.shuffle(trial_type_both)
    trial_type_final = [f"Trial_{i + 1}_{item}" for i, item in enumerate(trial_type_both)]
    print(trial_type_final)

    ref_rgb_values = np.random.rand(10, 3)  # Generate 10 random RGB values
    comp_rgb_values = np.random.rand(10, 3)
else:
    stim_at_thres_path = r'c:\Users\brainardlab\Aguirre-Brainard Lab Dropbox\Fangfang Hong\ELPS_analysis\Experiment_DataFiles\pilot2\sub1\fits\Stim_at_thres_for_image_generation_sub1.pkl'
    # Load the dictionary from the pickle file
    with open(stim_at_thres_path, 'rb') as f:
        stim_at_thres_dict = pickle.load(f)
    ref_rgb_values = stim_at_thres_dict['MOCS_trials_RGB']['MOCS_xref_shuffled']
    comp_rgb_values = stim_at_thres_dict['MOCS_trials_RGB']['MOCS_x1_shuffled']
    
    trial_type_final = [f'MOCS_{i}' for i in range(1, ref_rgb_values.shape[0]+1)]
    
#run it
for i, (trial, ref_rgb, comp_rgb) in enumerate(zip(trial_type_final, ref_rgb_values, comp_rgb_values), start=1):
    print(f"Sending reference and comparison pair {i}...")
    communicator.send_RGBvals(trial, ref_rgb.tolist(), comp_rgb.tolist())
    print(f"RGB values {i} confirmed.")

# Step 3: Finalize
print("Finalizing communication...")
communicator.finalize()
print("Communication finalized.")
#update the communication status
expt_file_manager.status_updates('Done')
    
#%%add a comment at the end of an experiment
expt_file_manager.add_comments(get_comment_after_session())





