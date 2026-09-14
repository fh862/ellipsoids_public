#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:10:09 2024

@author: fangfang
"""
import tkinter as tk
from tkinter import filedialog
import json
import os
import platform
import re
import subprocess
import sys
from numbers import Integral
from pathlib import Path
from typing import Any
import dill as pickled
import numpy as np
import glob
from scipy.io import loadmat
import pandas as pd
from colour.models import XYZ_to_xyY
from .ellipses_tools import PointsOnEllipseQ

#%%
class PathConfig:
    """Load machine-specific filesystem paths from repository JSON files."""

    _config_cache: dict[str, Any] | None = None
    _config_dir = Path(__file__).resolve().parents[2] / "config"

    @classmethod
    def default_path(cls) -> Path:
        platform_name = os.environ.get("ELLIPSOIDS_PATH_PLATFORM", platform.system()).lower()
        if platform_name.startswith("win"):
            return cls._config_dir / "hardcoded_paths_windows.json"
        return cls._config_dir / "hardcoded_paths_mac.json"

    @classmethod
    def active_path(cls) -> Path:
        override = os.environ.get("ELLIPSOIDS_PATH_CONFIG")
        return Path(override).expanduser() if override else cls.default_path()

    @classmethod
    def load(cls) -> dict[str, Any]:
        if cls._config_cache is None:
            cls._config_cache = cls._load_json(cls.active_path())
        return cls._config_cache

    @staticmethod
    def _load_json(config_path: Path) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get(cls, name: str) -> str:
        paths = cls.load()
        if name in paths:
            return paths[name]

        if "ELLIPSOIDS_PATH_CONFIG" not in os.environ:
            for config_path in (
                cls._config_dir / "hardcoded_paths_mac.json",
                cls._config_dir / "hardcoded_paths_windows.json",
            ):
                if config_path == cls.active_path():
                    continue
                fallback_paths = cls._load_json(config_path)
                if name in fallback_paths:
                    return fallback_paths[name]

        raise KeyError(f"Path key {name!r} is not defined in {cls.active_path()}")


def get_path(name: str) -> str:
    """Return one configured path by key."""
    return PathConfig.get(name)


def load_pickle_with_jax_arrays_as_numpy(path):
    """Load a dill pickle while reconstructing serialized JAX arrays as NumPy."""
    import jax._src.array as jax_array

    reconstruct_jax_array = jax_array._reconstruct_array

    def reconstruct_as_numpy(fun, args, arr_state, aval_state):
        del aval_state
        np_value = fun(*args)
        np_value.__setstate__(arr_state)
        return np_value

    jax_array._reconstruct_array = reconstruct_as_numpy
    try:
        with open(path, "rb") as f:
            return pickled.load(f)
    finally:
        jax_array._reconstruct_array = reconstruct_jax_array


def _jnp():
    import jax.numpy as jnp

    return jnp


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
        jnp = _jnp()

        # Extract MOCS trial data from each session and store in separate lists
        xref_MOCS_list = [jnp.array(d['data_vis_MOCS'].xref_all) for d in data_allSessions]
        x1_MOCS_list   = [jnp.array(d['data_vis_MOCS'].x1_all) for d in data_allSessions]
        y_MOCS_list    = [jnp.array(d['data_vis_MOCS'].y_all) for d in data_allSessions]

        # Concatenate data across all sessions along axis 0
        xref_MOCS, x1_MOCS, y_MOCS = map(lambda lst: np.concatenate(lst, axis=0), 
                                         [xref_MOCS_list, x1_MOCS_list, y_MOCS_list])
        
        return xref_MOCS_list, x1_MOCS_list, y_MOCS_list, xref_MOCS, x1_MOCS, y_MOCS

    def assign_trials_to_conditions(xref_trials, x1_trials,
                                    xref_by_condition, x1_by_condition,
                                    atol = 1e-6):
        """Assign trials to fixed conditions using their reference and comparison #1.

        Parameters
        ----------
        xref_trials : array-like, shape (n_trials, n_dims)
            Reference stimulus for each trial.
        x1_trials : array-like, shape (n_trials, n_dims)
            Fixed comparison #1 for each trial.
        xref_by_condition : array-like, shape (n_conditions, n_dims)
            Reference stimulus defining each condition. Row order defines the
            returned zero-based condition indices.
        x1_by_condition : array-like, shape (n_conditions, n_dims)
            Fixed comparison #1 defining each condition.
        atol : float, optional
            Absolute tolerance used to match stimulus coordinates. Relative
            tolerance is fixed at zero. Default is 1e-6.

        Returns
        -------
        np.ndarray, shape (n_trials,)
            Zero-based condition index for every trial.

        Raises
        ------
        ValueError
            If any trial matches zero or multiple conditions.
        """
        xref_trials = np.asarray(xref_trials)
        x1_trials = np.asarray(x1_trials)
        xref_by_condition = np.asarray(xref_by_condition)
        x1_by_condition = np.asarray(x1_by_condition)

        matches = (
            np.all(np.isclose(xref_trials[:, None, :],
                              xref_by_condition[None, :, :],
                              atol = atol, rtol = 0), axis = -1)
            &
            np.all(np.isclose(x1_trials[:, None, :],
                              x1_by_condition[None, :, :],
                              atol = atol, rtol = 0), axis = -1)
        )
        match_counts = matches.sum(axis = 1)

        if np.any(match_counts != 1):
            unmatched = np.where(match_counts == 0)[0]
            ambiguous = np.where(match_counts > 1)[0]
            raise ValueError(
                f"Each trial must match exactly one condition; found "
                f"{unmatched.size} unmatched and {ambiguous.size} ambiguous trials."
            )

        return matches.argmax(axis = 1)
        
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
    
    def load_AEPsych_data(data_allSessions, include_x2=False):
        """Load recorded AEPsych trials, optionally including comparison #2.

        Parameters
        ----------
        data_allSessions : list of dict
            Nonempty list of loaded session dictionaries, in the desired order.
        include_x2 : bool, optional
            False preserves the original eight-output interface and does not
            require x2_all. True also loads x2_all for suprathreshold trials:
            x1 is the fixed comparison for each condition, and x2 is the varying
            comparison. A response of 1 means x2 was judged more different from
            xref; 0 means x1 was chosen.

        Returns
        -------
        tuple
            With include_x2=False:
            (xref_list, x1_list, y_list, time_list, xref, x1, y, time).
            With include_x2=True:
            (xref_list, x1_list, x2_list, y_list, time_list,
             xref, x1, x2, y, time).
            Stimulus and response lists contain one JAX array per session;
            time_list retains each session's original time_elapsed array.
            Concatenated outputs are NumPy arrays, in input session order and
            stored trial order. Stimuli have shape (n_trials, n_dims), and
            responses have shape (n_trials,). No trials are filtered here.
            Pregenerated fallback Sobol trials are loaded separately with
            load_pregSobol_data.

        Raises
        ------
        ValueError
            If no sessions are supplied.
        AttributeError
            If a required trial attribute is missing, including x2_all when
            include_x2=True.
        """
        if len(data_allSessions) == 0:
            raise ValueError("At least one session is required to load AEPsych data.")

        jnp = _jnp()
        # Keep the original field order unless comparison #2 is requested.
        fields = ["xref_all", "x1_all"]
        if include_x2:
            fields.append("x2_all")
        fields.append("binaryResp_all")
        data_lists = [[] for _ in fields]
        time_elapsed_list = []

        for session in data_allSessions:
            trials = session["expt_trials"]
            for field, values in zip(fields, data_lists):
                values.append(jnp.array(getattr(trials, field)))
            time_elapsed_list.append(trials.time_elapsed)

        # Return per-session lists followed by their concatenated arrays.
        data_lists.append(time_elapsed_list)
        combined = [np.concatenate(values, axis=0) for values in data_lists]
        return (*data_lists, *combined)

    def load_pregSobol_data(data_allSessions, include_x2=False):
        """Load pregenerated fallback Sobol trials with recorded responses.

        These trials are separate from Sobol trials generated by AEPsych. Their
        scheduling and number depend on the experiment. Unused pool entries have
        NaN responses and are excluded using the same mask for every stimulus.

        Parameters
        ----------
        data_allSessions : list of dict
            Nonempty list of loaded session dictionaries, in the desired order.
        include_x2 : bool, optional
            If False (default), preserve the original six-output interface and
            do not require an x2 field. If True, also load comparison #2 from
            every session. For suprathreshold trials, x1 is the fixed comparison
            for each condition and x2 is the varying comparison. Responses are
            1 when x2 is judged more different from xref, and 0 when x1 is chosen.

        Returns
        -------
        tuple
            With include_x2=False:
            (xref_list, x1_list, y_list, xref, x1, y).
            With include_x2=True:
            (xref_list, x1_list, x2_list, y_list, xref, x1, x2, y).
            Each *_list contains one JAX array per session; the remaining values
            are NumPy arrays concatenated along the trial axis. Stimulus arrays
            have shape (n_trials, n_dims), and responses have shape (n_trials,).
            Sessions without recorded responses retain empty arrays in the lists.
            Ordering follows the input sessions and their stored pool indices,
            rather than reconstructing the interleaved presentation sequence.

        Raises
        ------
        ValueError
            If no sessions are supplied.
        KeyError
            If a required data field is missing, including x2 when requested.
        """
        if len(data_allSessions) == 0:
            raise ValueError("At least one session is required to load Sobol data.")

        jnp = _jnp()
        # Field order determines both the per-session and concatenated outputs.
        fields = ["xref", "x1", "x2", "binaryResp"] if include_x2 else [
            "xref", "x1", "binaryResp"
        ]
        data_lists = [[] for _ in fields]

        for session in data_allSessions:
            sobol = session["sim_interleaved_trial_sequence"].pregenerated_Sobol
            responses = jnp.array(sobol["binaryResp"])
            valid_indices = jnp.where(~jnp.isnan(responses))[0]

            # Apply identical trial selection to responses and all comparisons.
            for field, values in zip(fields, data_lists):
                array = responses if field == "binaryResp" else jnp.array(sobol[field])
                values.append(array[valid_indices])

        combined = [np.concatenate(values, axis=0) for values in data_lists]
        return (*data_lists, *combined)

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
        jnp = _jnp()
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
        
