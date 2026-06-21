#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:10:09 2024

@author: fangfang
"""
import tkinter as tk
from tkinter import filedialog
import os
import re
import subprocess
import sys
from numbers import Integral
import dill as pickled
import jax.numpy as jnp
import numpy as np
import glob
from scipy.io import loadmat
import pandas as pd
from colour.models import XYZ_to_xyY
from .ellipses_tools import PointsOnEllipseQ

#%%
def find_files_with_prefix(path, prefix, file_type="csv"):
    """
    Search for files in a given directory that start with a specific prefix 
    and have the given file extension.

    If no files are found, returns None.
    If only one file is found, returns that file path as a string.
    If multiple files are found, prompts the user to choose one or return all.

    Parameters
    ----------
    path : str
        Directory path to search in.
    prefix : str
        File name prefix to match (case-sensitive).
    file_type : str, optional
        File extension (default is 'csv').

    Returns
    -------
    str or list of str or None
        - Single file path (str) if one file is selected
        - List of file paths if 'all' is selected
        - None if no match or invalid choice
    """

    # Build the search pattern (e.g., "/path/prefix*.csv")
    pattern = os.path.join(path, f"{prefix}*.{file_type}")

    # Use glob to find matching files
    matched_files = glob.glob(pattern)

    # Case 1: No matching files found
    if not matched_files:
        print("No files found.")
        return None

    # Case 2: Exactly one matching file — return immediately
    if len(matched_files) == 1:
        return matched_files[0]

    # Case 3: Multiple matches — ask user which file(s) to choose
    print("Multiple files found:")
    for i, file in enumerate(matched_files, 1):
        print(f"{i}: {os.path.basename(file)}")

    # Prompt user input: file number or "all"
    choice = input(
        "Enter the number of the file to choose, "
        "or type 'all' for all matches: "
    ).strip()

    # If user types 'all', return the whole list
    if choice.lower() == "all":
        return matched_files

    # If user selects a specific file number
    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(matched_files):
            return matched_files[choice_index]
        else:
            print("Invalid choice.")
            return None
    except ValueError:
        # User entered something that's not an integer or 'all'
        print("Invalid input.")
        return None

def _select_file_with_macos_dialog():
    script = 'POSIX path of (choose file with prompt "Select a File")'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        if "User canceled" in result.stderr:
            return ""
        return None

    return result.stdout.strip()


def select_file_and_get_path():
    if sys.platform == "darwin":
        file_path = _select_file_with_macos_dialog()
        if file_path == "":
            return None, None
        if file_path:
            directory, file_name = os.path.split(file_path)
            return directory, file_name

    # Create a hidden Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.update_idletasks()

    try:
        # Open the file dialog
        file_path = filedialog.askopenfilename(
            title="Select a File",
            filetypes=[("CSV Files", "*.csv"), ("Pickle Files", "*.pkl"), ("Mat Files", "*.mat")]
        )

        # If a file is selected, split its path into directory and file name
        if file_path:
            directory, file_name = os.path.split(file_path)
            return directory, file_name
        else:
            return None, None
    finally:
        root.destroy()

def select_files_and_get_paths():
    # Create a hidden Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the file dialog for multiple selection
    file_paths = filedialog.askopenfilenames(
        title="Select One or More Files",
        filetypes=[("CSV Files", "*.csv"), ("Pickle Files", "*.pkl"), ("Mat Files", "*.mat")]
    )

    # Return a list of full file paths
    return list(file_paths) if file_paths else []

def select_multiple_files_across_folders():
    root = tk.Tk()
    root.withdraw()

    all_selected_files = []

    while True:
        file_paths = filedialog.askopenfilenames(
            title="Select files (Cancel to stop selecting more)",
            filetypes=[("CSV Files", "*.csv"), ("Pickle Files", "*.pkl"), ("Mat Files", "*.mat")]
        )

        if not file_paths:
            break  # User pressed cancel

        all_selected_files.extend(file_paths)

    return all_selected_files
    
def extract_sub_number(input_string):
    """
    Extracts the integer following 'sub' in the input string.
    
    Parameters:
        input_string (str): The string containing 'sub' followed by an integer.
    
    Returns:
        int: The integer following 'sub' in the input string.
        None: If no match is found.
    """
    match = re.search(r'sub(\d+)', input_string)
    if match:
        return int(match.group(1))
    return None

def extract_between_patterns(str_list, str_prefix, str_suffix):
    """
    Extracts substrings between a given prefix and suffix from each string in the list.

    Parameters
    ----------
    str_list : List[str]
        List of strings to search.
    str_prefix : str
        The substring that comes before the target.
    str_suffix : str
        The substring that comes after the target.

    Returns
    -------
    List[Optional[str]]
        List of extracted substrings. Returns None for strings that do not match.
    """
    pattern = re.escape(str_prefix) + r'(.*?)' + re.escape(str_suffix)
    return [re.search(pattern, s).group(1) if re.search(pattern, s) else None for s in str_list]


def values_disagree(existing, loaded):
    """
    Return True when two values appear to disagree.
    """
    if type(existing) is not type(loaded):
        return True

    if isinstance(existing, np.ndarray):
        return not np.array_equal(existing, loaded, equal_nan=True)

    if isinstance(existing, (list, tuple)):
        if len(existing) != len(loaded):
            return True
        return any(values_disagree(x, y) for x, y in zip(existing, loaded))

    if isinstance(existing, dict):
        if existing.keys() != loaded.keys():
            return True
        return any(values_disagree(existing[k], loaded[k]) for k in existing)

    try:
        return existing != loaded
    except Exception:
        return False

#%%
class load_expt_data:
    def get_all_sessions_file_names(subN, nSessions, path_str, 
                                    exptCond = '_4dExpt_Isoluminant plane',
                                    str_ext = '_copy'):
        """
        Generate a list of file paths for all session data files of a given subject.

        Parameters:
        subN (int or str): Subject number or identifier.
        nSessions (int or iterable of int): Total number of sessions, or the
            explicit session numbers to import.
        path_str (str): Base directory path where session files are stored.

        Returns:
        tuple: A list of file paths (session_files) and the common filename part (session_file_name_part1).
        """

        # Construct the common part of the session file names
        session_file_name_part1 = f'ColorDiscrimination{exptCond}_sub{subN}'

        if isinstance(nSessions, Integral):
            session_ids = range(1, nSessions + 1)
        else:
            session_ids = nSessions

        # Generate full file paths for all session files
        session_files = [
            os.path.join(path_str, f'{session_file_name_part1}_session{session_id}{str_ext}.pkl')
            for session_id in session_ids
        ]
        
        return session_files, session_file_name_part1
    
    def load_data_all_sessions(session_files):
        """
        Load data from all session files.

        Parameters:
        session_files (list of str): List of file paths to session data files.

        Returns:
        list: A list containing loaded data for each session.
        """

        # Load and deserialize data from all session files
        data_all_sessions = []   
        for file in session_files:
            try:
                with open(file, 'rb') as f:
                    data_all_sessions.append(pickled.load(f))
            except (FileNotFoundError, EOFError) as e:
                print(f"Warning: Skipping {file} due to error: {e}")

        return data_all_sessions
    
    def load_MOCS_data(data_allSessions):
        """
        Extract and preprocess Method of Constant Stimuli (MOCS) trial data from all sessions.

        Parameters:
        data_allSessions (list of dict): List containing data from multiple sessions.

        Returns:
        tuple: 
            - xref_MOCS_list (list of jnp.ndarray): List of reference stimuli arrays from each session.
            - x1_MOCS_list (list of jnp.ndarray): List of comparison stimuli arrays from each session.
            - y_MOCS_list (list of jnp.ndarray): List of participant response arrays from each session.
            - xref_MOCS (np.ndarray): Concatenated reference stimuli across all sessions.
            - x1_MOCS (np.ndarray): Concatenated comparison stimuli across all sessions.
            - y_MOCS (np.ndarray): Concatenated participant responses across all sessions.
        """

        # Extract MOCS trial data from each session and store in separate lists
        xref_MOCS_list = [jnp.array(d['data_vis_MOCS'].xref_all) for d in data_allSessions]
        x1_MOCS_list   = [jnp.array(d['data_vis_MOCS'].x1_all) for d in data_allSessions]
        y_MOCS_list    = [jnp.array(d['data_vis_MOCS'].y_all) for d in data_allSessions]

        # Concatenate data across all sessions along axis 0
        xref_MOCS, x1_MOCS, y_MOCS = map(lambda lst: np.concatenate(lst, axis=0), 
                                         [xref_MOCS_list, x1_MOCS_list, y_MOCS_list])
        
        return xref_MOCS_list, x1_MOCS_list, y_MOCS_list, xref_MOCS, x1_MOCS, y_MOCS
        
    def org_MOCS_by_condition(xref_MOCS, x1_MOCS, y_MOCS, leave_out_conditions = []):
        """
        Organize MOCS data by unique reference stimulus conditions.

        Parameters:
        xref_MOCS (np.ndarray): Concatenated reference stimuli across all sessions.
        x1_MOCS (np.ndarray): Concatenated comparison stimuli across all sessions.
        y_MOCS (np.ndarray): Concatenated participant responses across all sessions.

        Returns:
        tuple:
            - xref_unique_MOCS (np.ndarray): Unique reference stimuli.
            - nRefs_MOCS (int): Number of unique reference stimulus conditions.
            - refStimulus_MOCS (list of np.ndarray): Grouped reference stimuli for each unique condition.
            - compStimulus_MOCS (list of np.ndarray): Grouped comparison stimuli for each unique condition.
            - responses_MOCS (list of np.ndarray): Grouped participant responses for each unique condition.
            - nLevels_MOCS (int): Number of unique levels per condition.
            - nTrials_MOCS (int): Total number of trials.
        """

        # Identify unique reference stimulus conditions in MOCS trials
        xref_unique_MOCS = np.unique(xref_MOCS, axis=0)

        # Count the number of unique reference stimulus conditions
        nRefs_MOCS = xref_unique_MOCS.shape[0]

        # Initialize lists to store grouped data by reference stimulus condition
        refStimulus_MOCS, compStimulus_MOCS, responses_MOCS = [], [], []
        
        #initialize lists to save leaved out data
        refStimulus_MOCS_leaveout, compStimulus_MOCS_leaveout, responses_MOCS_leaveout = [], [], []

        # Iterate through each unique reference stimulus and collect matching trials
        for i in range(nRefs_MOCS):
            matching_indices = np.where((xref_MOCS == xref_unique_MOCS[i]).all(axis=1))[0]
            if i not in leave_out_conditions:
                refStimulus_MOCS.append(xref_MOCS[matching_indices])
                compStimulus_MOCS.append(x1_MOCS[matching_indices])
                responses_MOCS.append(y_MOCS[matching_indices])
            else:
                refStimulus_MOCS_leaveout.append(xref_MOCS[matching_indices])
                compStimulus_MOCS_leaveout.append(x1_MOCS[matching_indices])
                responses_MOCS_leaveout.append(y_MOCS[matching_indices])                

        # Determine the number of unique comparison stimulus levels per condition
        nLevels_MOCS = np.unique(x1_MOCS[matching_indices], axis=0).shape[0]

        # Total number of trials in the dataset
        nTrials_MOCS = y_MOCS.shape[0]
        
        return xref_unique_MOCS, nRefs_MOCS, refStimulus_MOCS, compStimulus_MOCS, \
            responses_MOCS, nLevels_MOCS, nTrials_MOCS, refStimulus_MOCS_leaveout, \
            compStimulus_MOCS_leaveout, responses_MOCS_leaveout
    
    def load_AEPsych_data(data_allSessions):
        """
        Extract and preprocess AEPsych trial data from all sessions.

        Parameters:
        data_allSessions (list of dict): List containing data from multiple experimental sessions.

        Returns:
        tuple: 
            - xref_AEPsych_list (list of jnp.ndarray): List of reference stimuli arrays from each session.
            - x1_AEPsych_list (list of jnp.ndarray): List of comparison stimuli arrays from each session.
            - y_AEPsych_list (list of jnp.ndarray): List of binary response arrays from each session.
            - time_elapsed_list (list of np.ndarray): List of time elapsed arrays from each session.
            - xref_AEPsych (np.ndarray): Concatenated reference stimuli across all sessions.
            - x1_AEPsych (np.ndarray): Concatenated comparison stimuli across all sessions.
            - y_AEPsych (np.ndarray): Concatenated binary responses across all sessions.
            - time_elapsed (np.ndarray): Concatenated time elapsed across all sessions.
        """

        # Extract relevant data for AEPsych trials from each session
        time_elapsed_list = [d['expt_trials'].time_elapsed for d in data_allSessions]
        xref_AEPsych_list = [jnp.array(d['expt_trials'].xref_all) for d in data_allSessions]
        x1_AEPsych_list   = [jnp.array(d['expt_trials'].x1_all) for d in data_allSessions]
        y_AEPsych_list    = [jnp.array(d['expt_trials'].binaryResp_all) for d in data_allSessions]

        # Concatenate data across all sessions along axis 0
        xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed = \
            map(lambda lst: np.concatenate(lst, axis=0), 
                [xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list])

        return xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list, \
               xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed
               
    def load_pregSobol_data(data_allSessions):
        """
        Extract and preprocess pregenerated Sobol trial data from multiple sessions.

        Unlike AEPsych generated Sobol trials—which are always 900 in number and 
        appear at the beginning of each session—the pregenerated Sobol trials in this 
        dataset are dynamically interleaved throughout a session. Their number varies 
        both across sessions and across subjects.

        These pregenerated trials are inserted whenever the MOCS trials get more than 
        4 trials ahead of the AEPsych trials. Instead of advancing MOCS further, we 
        insert one of the pregenerated Sobol trials. Although we generated 1,200 
        Sobol trials per session, only ~50 are typically used in each session.

        Parameters:
        data_allSessions (list of dict): List containing data from multiple experimental sessions.

        Returns:
        tuple: 
            - xref_pregSobol_list (list of jnp.ndarray): List of reference stimuli arrays from each session.
            - x1_pregSobol_list (list of jnp.ndarray): List of comparison stimuli arrays from each session.
            - y_pregSobol_list (list of jnp.ndarray): List of binary response arrays from each session.
            - xref_pregSobol (np.ndarray): Concatenated reference stimuli across all sessions.
            - x1_pregSobol (np.ndarray): Concatenated comparison stimuli across all sessions.
            - y_pregSobol (np.ndarray): Concatenated binary responses across all sessions.
        """
        # Initialize lists to store pregenerated Sobol data from each session
        xref_pregSobol_list, x1_pregSobol_list, y_pregSobol_list = [], [], []
             
        # Loop through each session's data
        for i in range(len(data_allSessions)):
            d = data_allSessions[i]
            # Access the pregenerated Sobol data for that session
            d_pregSobol = d['sim_interleaved_trial_sequence'].pregenerated_Sobol

            # Extract binary responses and stimuli
            y_pregSobol_i = jnp.array(d_pregSobol['binaryResp'])   # shape: (n_trials,)
            xref_pregSobol_i = jnp.array(d_pregSobol['xref'])      # shape: (n_trials, dim)
            x1_pregSobol_i = jnp.array(d_pregSobol['x1'])          # shape: (n_trials, dim)

            # Identify indices where the binary response is not NaN
            mask = ~jnp.isnan(y_pregSobol_i)
            float_indices = jnp.where(mask)[0]  # Indices with valid float values       
            
            # Append only valid (non-NaN) trials to the corresponding lists
            xref_pregSobol_list.append(xref_pregSobol_i[float_indices])
            x1_pregSobol_list.append(x1_pregSobol_i[float_indices])
            y_pregSobol_list.append(y_pregSobol_i[float_indices])

        # Concatenate all valid trials from each session into full arrays
        xref_pregSobol, x1_pregSobol, y_pregSobol = \
            map(lambda lst: np.concatenate(lst, axis=0), 
                [xref_pregSobol_list, x1_pregSobol_list, y_pregSobol_list])     
        
        return xref_pregSobol_list, x1_pregSobol_list, y_pregSobol_list,\
            xref_pregSobol, x1_pregSobol, y_pregSobol
            
    def load_combine_AEPsych_pregSobol(data_allSessions):
        """
        Loads and combines AEPsych trials with pre-generated Sobol trials, if available.
    
        For most subjects, both AEPsych and Sobol trials are included and can be combined.
        However, for **pilot subject #1**, Sobol trials were not included in the experiment.
        This function handles that exception gracefully using a try-except block.
    
        If subject #1 ever re-runs the experiment using the same version as the other subjects
        (which includes Sobol trials), this exception will no longer be needed.

        """
        # Load AEPsych data
        (
            xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list,
            xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed
        ) = load_expt_data.load_AEPsych_data(data_allSessions)
    
        try:
            # Load pre-generated Sobol data
            (
                xref_pregSobol_list, x1_pregSobol_list, y_pregSobol_list,
                xref_pregSobol, x1_pregSobol, y_pregSobol
            ) = load_expt_data.load_pregSobol_data(data_allSessions)
    
            # Combine AEPsych and Sobol data
            xref_combined = np.concatenate([xref_AEPsych, xref_pregSobol], axis=0)
            x1_combined   = np.concatenate([x1_AEPsych,   x1_pregSobol],   axis=0)
            y_combined    = np.concatenate([y_AEPsych,    y_pregSobol],    axis=0)
    
            return (
                # AEPsych data
                [xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list, 
                 xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed],
                
                # Sobol data
                [xref_pregSobol_list, x1_pregSobol_list, y_pregSobol_list, 
                 xref_pregSobol, x1_pregSobol, y_pregSobol],
    
                # Combined data
                [xref_combined, x1_combined, y_combined]
            )
    
        except:
            print("No pre-generated Sobol trials found for this participant.")
    
            # Return AEPsych data only
            return (
                [xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list, 
                 xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed],
    
                # No Sobol data
                None,
    
                # Combined data is just the AEPsych data
                [xref_AEPsych, x1_AEPsych, y_AEPsych]
            )
        
    def bootstrap_AEPsych_data(xref, x1, y, trials_split=[900], seed=None):
        """
        Bootstraps AEPsych trials by splitting the data into chunks defined by `trials_split`,
        then resampling each chunk with replacement independently.
    
        Parameters
        ----------
        xref : np.ndarray, Reference stimuli of shape (n_trials, 2).
        x1 : np.ndarray, Comparison stimuli of shape (n_trials, ...).
        y : np.ndarray, Observed responses of shape (n_trials, ...).
        trials_split : list of int, optional
            List of trial indices where the data should be split.
            Example: [900] will split trials into [0:900) and [900:end).
        seed : int, optional
            Seed for the random number generator for reproducibility.
    
        Returns
        -------
        xref_btst : np.ndarray, Bootstrapped xref data.
        x1_btst : np.ndarray, Bootstrapped x1 data.
        y_btst : np.ndarray, Bootstrapped y data.
        """
        np.random.seed(seed)
    
        # Validate trials_split
        if len(trials_split) > 1:
            if not all(trials_split[i] < trials_split[i + 1] for i in range(len(trials_split) - 1)):
                raise ValueError("trials_split must be in strictly increasing order.")
    
        #append the first and the last indices    
        split_points = [0] + trials_split + [xref.shape[0]]
    
        # Preallocate arrays for bootstrapped results
        xref_btst = np.full(xref.shape, np.nan)
        x1_btst = np.full(x1.shape, np.nan)
        y_btst = np.full(y.shape, np.nan)
        sampled_indices_btst = np.full((xref.shape[0],), np.nan)
    
        # Resample within each segment defined by trials_split
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            n = end_idx - start_idx
            
            #sample indices with replacement 
            sample_indices = np.random.randint(start_idx, end_idx, size=n)
            sampled_indices_btst[start_idx:end_idx] = sample_indices
    
            #store bootstrapped dataset
            xref_btst[start_idx:end_idx] = xref[sample_indices]
            x1_btst[start_idx:end_idx] = x1[sample_indices]
            y_btst[start_idx:end_idx] = y[sample_indices]
    
        return jnp.array(xref_btst), jnp.array(x1_btst), jnp.array(y_btst), sampled_indices_btst
               
    def load_AEPsych_data_before_last_MOCS(data_allSessions):
        """
        Extract and preprocess AEPsych trial data from all sessions,
        **excluding any AEPsych trials that occurred after the last MOCS trial** 
        within each session.

        This function is a specialized version of `load_AEPsych_data`, designed 
        specifically for subject #1 during the pilot study. In this early version 
        of the experiment, MOCS trials were presented first and not interleaved 
        with AEPsych trials near the end of the session. This may have caused 
        biases or inconsistencies in threshold estimates.

        By trimming the AEPsych trials that came after the last MOCS trial, this 
        method allows testing the hypothesis that these late AEPsych trials 
        contributed to the mismatch in thresholds between the two methods.

        Important:
        - This method is intended **only for subject #1**.
        - For all other subjects, AEPsych and MOCS trials were properly interleaved,
          so this function is not needed.
        - If subject #1 re-runs the experiment using the final interleaved version, 
          this function will no longer be necessary.
    
        Parameters:
        data_allSessions (list of dict): List containing data from multiple experimental sessions.
    
        Returns: check load_AEPsych_data
        """
    
        # Initialize lists to store results per session
        last_idx_included_AEPsych = []  # Index of the last AEPsych trial before the final MOCS trial
        nTrials_included_AEPsych = []   # Number of AEPsych trials included
        time_elapsed_list = []          # List of time elapsed arrays per session
        xref_AEPsych_list = []          # List of reference stimuli arrays per session
        x1_AEPsych_list = []            # List of comparison stimuli arrays per session
        y_AEPsych_list = []             # List of response arrays per session
    
        # Iterate through all sessions
        for idx, d in enumerate(data_allSessions):
            # Extract the trial sequence array for the session
            sequence_array = d['sim_interleaved_trial_sequence'].final_sequence[0]
    
            # Find the index of the last 'MOCS' trial in the sequence
            last_mocs_index = next((i for i in range(len(sequence_array) - 1, -1, -1) 
                                    if sequence_array[i].startswith('MOCS')), None)
    
            # Define the number of AEPsych trials to include (all trials up to the last MOCS trial)
            last_idx_included_AEPsych.append(last_mocs_index + 1)
    
            # Extract the trial number from the last included AEPsych trial
            match = re.search(r'AEPsych_(\d+)', sequence_array[last_idx_included_AEPsych[-1]])
            trial_num = int(match.group(1)) if match else None  # Convert to integer if found, otherwise None
    
            # Store the number of AEPsych trials included for this session
            nTrials_included_AEPsych.append(trial_num)
    
            # Extract and truncate AEPsych trial data up to the last included trial
            time_elapsed_list.append(d['expt_trials'].time_elapsed[:trial_num])
            xref_AEPsych_list.append(jnp.array(d['expt_trials'].xref_all[:trial_num]))
            x1_AEPsych_list.append(jnp.array(d['expt_trials'].x1_all[:trial_num]))
            y_AEPsych_list.append(jnp.array(d['expt_trials'].binaryResp_all[:trial_num]))
    
        # Concatenate data across all sessions along axis 0
        xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed = map(
            lambda lst: np.concatenate(lst, axis=0), 
            [xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list]
        )
    
        return xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list, \
               xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed
    
    def combine_AEPsych_MOCS(data_AEPsych, data_MOCS, flag_combine=True):
        """
        Combine AEPsych and MOCS datasets if the flag is set to True. 
        Otherwise, return only AEPsych data.

        Parameters:
        data_AEPsych (tuple of jnp.ndarray): AEPsych data containing (xref, x1, y).
        data_MOCS (tuple of jnp.ndarray): MOCS data containing (xref, x1, y).
        flag_combine (bool, optional): Whether to combine AEPsych and MOCS data. Defaults to True.

        Returns:
        tuple:
            - xref_jnp (jnp.ndarray): Combined or AEPsych-only reference stimuli.
            - x1_jnp (jnp.ndarray): Combined or AEPsych-only comparison stimuli.
            - y_jnp (jnp.ndarray): Combined or AEPsych-only response data.
        """

        if flag_combine:
            # Stack reference stimuli (xref), comparison stimuli (x1), and responses (y) from both datasets
            xref_jnp = jnp.vstack((data_AEPsych[0], data_MOCS[0]))  # Vertical stack for 2D data
            x1_jnp   = jnp.vstack((data_AEPsych[1], data_MOCS[1]))  # Vertical stack for 2D data
            y_jnp    = jnp.hstack((data_AEPsych[2], data_MOCS[2]))  # Horizontal stack for 1D response data
        else:
            # Use only AEPsych data without modification
            xref_jnp, x1_jnp, y_jnp = data_AEPsych

        return xref_jnp, x1_jnp, y_jnp

#%%        
class load_util_files:
    def load_matchingFunc_monitorSPD(file_dir, cal_file_date):    
        # Load the 1931 XYZ color matching functions (shape: 3 x 201)
        file_T_xyz = 'T_xyz1931.mat'
        T_xyz = loadmat(os.path.join(file_dir, file_T_xyz))['T_xyz1931']
        
        # Load the monitor's spectral primaries (shape: 201 x 3)
        file_P_device = f'B_monitor_dell_{cal_file_date}.mat'
        P_device = loadmat(os.path.join(file_dir, file_P_device))['B_monitor']
        
        # Compute the transformation matrix
        M_RGBToXYZ = T_xyz @ P_device  # shape: (3, 3)
        
        # Compute white point in XYZ assuming equal-energy white (R=G=B=1)
        rgb_white = np.array([1, 1, 1])
        XYZ_white = M_RGBToXYZ @ rgb_white
        xyY_white = XYZ_to_xyY(XYZ_white)
        
        # Print white point for sanity check (expect x and y around 0.3)
        print("White point (xyY):", xyY_white)
        
        # Compute chromaticity coordinates of the monitor primaries (gamut vertices)
        x_gamut = M_RGBToXYZ[0] / np.sum(M_RGBToXYZ, axis=0)
        y_gamut = M_RGBToXYZ[1] / np.sum(M_RGBToXYZ, axis=0)
        
        return x_gamut, y_gamut, T_xyz, P_device, M_RGBToXYZ, rgb_white, XYZ_white, xyY_white
    
    # Load MacAdam ellipse parameters
    def load_MacAdam_ellipses(file_dir, scaler=1/100, num_ell_pts=200):
        """
        Load MacAdam ellipse parameters from an Excel file and generate finely
        sampled ellipse contours in the CIE 1931 chromaticity diagram.
    
        Parameters
        ----------
        file_dir : str
            Path to the Excel file containing ellipse parameters.
        scaler : float, optional
            Scaling factor applied to the semi-major and semi-minor axes
            (MacAdam units → chromaticity units). Default = 1/100.
        num_ell_pts : int, optional
            Number of points used to sample each ellipse contour.
    
        Returns
        -------
        ellipses_fine : ndarray
            Ellipse contours with shape (n_ellipses, 2, num_ell_pts).
        macadam_df : pandas.DataFrame
            Original dataframe loaded from the Excel file.
        xc_array, yc_array : ndarray
            Ellipse center coordinates.
        semi_major, semi_minor : ndarray
            Scaled semi-major and semi-minor axis lengths.
        angles_deg : ndarray
            Ellipse rotation angles in degrees.
        n_ellipses : int
            Number of ellipses.
        """
    
        # Load table (skip first two rows containing metadata)
        macadam_df = pd.read_excel(file_dir, skiprows=2)
    
        # Extract ellipse parameters: xc, yc, a, b, theta
        vals = macadam_df.iloc[:, :5].to_numpy(dtype=float)
        xc_array, yc_array, semi_major, semi_minor, angles_deg = vals.T
    
        # Convert MacAdam axis units to chromaticity units
        semi_major *= scaler
        semi_minor *= scaler
    
        n_ellipses = len(semi_major)
    
        # Container for sampled ellipse coordinates: (ellipse, xy, points)
        ellipses_fine = np.full((n_ellipses, 2, num_ell_pts), np.nan)
    
        # Generate finely sampled points on each ellipse
        for idx, (a, b, theta, xc, yc) in enumerate(
            zip(semi_major, semi_minor, angles_deg, xc_array, yc_array)
        ):
            x_fine, y_fine = PointsOnEllipseQ(a, b, theta, xc, yc, nTheta=num_ell_pts)
            ellipses_fine[idx] = np.stack((x_fine, y_fine))
    
        return (
            ellipses_fine,
            macadam_df,
            xc_array,
            yc_array,
            semi_major,
            semi_minor,
            angles_deg,
            n_ellipses,
        )
        
