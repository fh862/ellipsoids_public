#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:54:10 2025

@author: fangfang
"""

import numpy as np

#%% create a new file
class ExperimentTrialSequence:
    def __init__(self, nTrials_AEPsych, pregenerated_MOCS = None, nBlocks = 1, 
                 break_trials = [], pregenerated_Sobol = None):
        """
        Initializes the ExperimentTrialSequence with the given number of trials and blocks.
        
        Parameters:
        - nTrials_AEPsych (int): The total number of trials for AEPsych.
        - pregenerated_MOCS (dict): A dictionary of pre-generated MOCS trials, including xref and x1.
        - nBlocks (int): The number of experimental blocks. This is required to ensure an even division 
                         of AEPsych and MOCS trials across all blocks.

        Attributes:
        - nTrials_MOCS (int): The total number of MOCS trials (based on the length of 'xref' in pregenerated_MOCS).
        - nTrials_total (int): The total number of trials across all blocks, combining AEPsych and MOCS trials.

        Functionality:
        - Ensures that AEPsych and MOCS trials are evenly divisible by the number of blocks.
        - Initializes a dictionary to store data for MOCS trials.
        """
        
        self.nTrials_AEPsych = nTrials_AEPsych
        self.pregenerated_MOCS = pregenerated_MOCS
        self.nBlocks = nBlocks
        if self.pregenerated_MOCS is not None: #if we do want to interleave MOCS trials
            self.nTrials_MOCS = pregenerated_MOCS['xref'].shape[-2]  # Total number of MOCS trials.
            
            #initialize a dictionary that saves the data for MOCS trials
            self._initialize_data_MOCS()
        else: #if not
            self.nTrials_MOCS = 0 
            
        #other info that's unrelated to the input MOCS trials
        # Check if the trials for AEPsych and MOCS are divisible by the number of blocks.
        self._check_divisible()
        self.nTrials_total = nTrials_AEPsych + self.nTrials_MOCS  # Total number of trials across all blocks.
        self.nBumpUp_MOCS = 0 #the number of times that a MOCS trial is bumped up 
        self.break_trials = break_trials
        self.pregenerated_Sobol = pregenerated_Sobol
        
    def _check_divisible(self):
        """
        Ensures that AEPsych trials and MOCS trials are divisible by the number of 
        experimental blocks. If any of these conditions are not met, an exception is raised.
    
        Raises:
        - ValueError: If AEPsych trials, or MOCS trials are not divisible by the number of blocks.
        """
        
        if self.nTrials_AEPsych % self.nBlocks != 0:
            # Raise an error if the number of AEPsych trials is not divisible by the number of blocks.
            raise ValueError(
                f"The number of AEPsych trials ({self.nTrials_AEPsych}) is NOT divisible by"+\
                    f" the number of blocks ({self.nBlocks}).")
        else:
            self.nTrials_AEPsych_perBlock = self.nTrials_AEPsych // self.nBlocks
        
        if self.nTrials_MOCS % self.nBlocks != 0:
        # Raise an error if the number of MOCS trials is not divisible by the number of blocks.
            raise ValueError(
                f"The number of MOCS trials ({self.nTrials_MOCS}) is NOT divisible by"+\
                    " the number of blocks ({self.nBlocks}).")
        else:
            self.nTrials_MOCS_perBlock = self.nTrials_MOCS // self.nBlocks
            
    def _initialize_trial_status(self, nExpt):
        """
        Initializes the trial_status list.
        """
        trial_status = []
        for i in range(nExpt):
            trial_status.append([[f'Trial_{n}_'+self.original_sequence[i][n]] for n in range(self.nTrials_total)])
        self.trial_status = trial_status
            
    def _initialize_data_MOCS(self):
        """
        Initialize the `data_MOCS` dictionary to store trial-specific data.

        If this is a simulation, we initialize additional fields (`Uref`, `U1`, 
        `signed_diff`, `pX1`) since we have access to the ground truth and can compute them.
        For actual experiments, these fields are not initialized because they cannot be computed.
        
        """
        self.data_MOCS = {
            trial: {
                'xref': None,  # Reference stimulus value
                'x1': None,    # Comparison stimulus value
                'Uref': None,  # Weighted sum of basis functions for the reference stimulus
                'U1': None,    # Weighted sum of basis functions for the comparison stimulus
                'signed_diff': None,  # Signed difference of squared mahalanobis distance
                'pX1': None,   # Probability of identifying the comparison stimulus as the odd stimulus
                'binaryResp': None  # Binary response (1 for correct, 0 for incorrect)
            } for trial in range(self.nTrials_MOCS)
        }        
           
    def set_trial_status(self, expt_idx, trial_idx, status):
        """
        Updates the status of a specific trial.
        e.g., ["Completed_in_time"]
              ["Timed_out", "Stepped_down_toXX", "Completed"]
              ["Stepped_up_toXX", "Completed"]

        Parameters:
        - expt_idx (int): Index of the configured experiment (we might interleave several different expts)
        - trial_idx (int): Index of the trial to update.
        - status (str): New status for the trial. Must be one of the allowed statuses.
        """
        if not (0 <= trial_idx < self.nTrials_total):
            raise IndexError(f"Trial index {trial_idx} is out of range (0 to {self.nTrials_total - 1}).")
        else:
            new_status_list = self.trial_status[expt_idx][trial_idx] + [status]
            self.trial_status[expt_idx][trial_idx] = new_status_list

    def update_data_MOCS(self, trial_idx, xref, x1, binaryResp, 
                         Uref = None, U1 = None, signed_diff = None, pX1 = None):
        """
        Update the `data_MOCS` dictionary for a specific trial.

        For simulations, this method updates all fields, including computed values (`Uref`, `U1`, 
        `signed_diff`, `pX1`). For actual experiments, only `xref`, `x1`, and `binaryResp` are updated.

        Args:
            trial_idx (int): The trial index to update.
            xref (float): Reference stimulus value.
            x1 (float): Comparison stimulus value.
            binaryResp (int): Binary response (e.g., 1 for correct, 0 for incorrect).
            Uref (float, optional): Utility value for the reference stimulus (simulation only).
            U1 (float, optional): Utility value for the comparison stimulus (simulation only).
            signed_diff (float, optional): Signed difference between stimuli (simulation only).
            pX1 (float, optional): Probability associated with the comparison stimulus (simulation only).
        """
        self.data_MOCS[trial_idx] = {
            'xref': xref,
            'x1': x1,
            'Uref': Uref,
            'U1': U1,
            'signed_diff': signed_diff,
            'pX1': pX1,
            'binaryResp': binaryResp
        }        
                
    def bump_up_one_MOCS_trial(self, expt_idx, trial_idx):
        """
        Replaces the current trial with the next available MOCS trial 
        in the updated sequence and updates the trial status accordingly.
    
        Parameters:
        ----------
        expt_idx : int
            Index of the configuration in the updated sequence.
        trial_idx : int
            Index of the trial to be replaced in the mixed list.
    
        Returns:
        -------
        tuple
            A tuple containing:
            - The trial number in the MOCS list (int).
            - The trial number in the mixed list (int).
            - The trial identity + trial number (str).

        Raises:
        -------
        ValueError
            If no MOCS trial is found after the given trial index.
        """
        if trial_idx >= self.nTrials_total:
            print("There are no more trials in the list.")

        # Retrieve the current trial
        current_trial = self.updated_sequence[expt_idx][trial_idx]

        # Look for the next available MOCS trial after the current trial index
        for n_search in range(trial_idx + 1, self.nTrials_total):
            # trial_next is the trial identity + trial number (e.g., 'MOCS_1')
            trial_next = self.updated_sequence[expt_idx][n_search]
            
            # Check that the MOCS trial has not been completed
            if trial_next.startswith('MOCS') and not self.trial_status[expt_idx][n_search][-1].startswith('Completed'):
                # Update trial statuses
                self.set_trial_status(expt_idx, trial_idx, f"Stepped_down_to_{trial_next}")
                self.set_trial_status(expt_idx, n_search, f"Stepped_up_to_{current_trial}")

                # Split the string by the underscore '_'
                _, trial_idx = trial_next.split('_') #the omitting part must be 'MOCS'

                # Return the trial number in the MOCS list, the trial number in the mixed list, and the trial identity
                return int(trial_idx), n_search, trial_next 

        # If no MOCS trial was found, raise an error
        print(f"No available MOCS trial found after trial_idx {trial_idx}.")
        return None, None, None
        
    def generate_init_sequence(self, nExpt=1, seed=None):
        """
        Generate a randomized sequence of trial labels for AEPsych and MOCS trials,
        ensuring that trials are evenly distributed across experimental blocks.

        Parameters
        ----------
        nExpt : int, optional
            Number of experiment configurations to generate sequences for. Default is 1.
        seed : int, optional
            Random seed for reproducible randomization. Default is None.

        Returns
        -------
        None
            The method updates the following attributes:
            - `self.original_sequence`: A list of randomized trial sequences for all experiments.
            - `self.updated_sequence`: A copy of the original sequences, which can be modified later.
            - Initializes `self.trial_status` for tracking the status of each trial.

        Notes
        -----
        - Each experimental block contains a balanced split of AEPsych and MOCS trials.
        - Trial labels are formatted as 'AEPsych_{n}' or 'MOCS_{n}', where `{n}` is the trial index.
        """
        # Set up the random seed for reproducibility
        rng = np.random.default_rng(seed)

        # Initialize sequences
        self.original_sequence = []
        self.final_sequence = [[]] * nExpt

        for n in range(nExpt):
            # Create string labels for AEPsych and MOCS trials
            list_str_AEPsych = [f'AEPsych_{n}' for n in range(self.nTrials_AEPsych)]
            list_str_MOCS = [f'MOCS_{n}' for n in range(self.nTrials_MOCS)]

            # Initialize a randomized list to store trial assignments
            list_randomized = []
            for n in range(self.nBlocks):
                # Ensure AEPsych and MOCS trials are evenly split across blocks
                temp_list = [0] * self.nTrials_AEPsych_perBlock + [1] * self.nTrials_MOCS_perBlock
                rng.shuffle(temp_list)  # Shuffle the trial assignments within the block
                list_randomized += temp_list

            # Initialize `shuffled_labels` with the appropriate size and type
            shuffled_labels = np.empty((self.nTrials_total,), dtype=object)

            # Map randomized trial assignments to their corresponding labels
            idx_match0 = np.where(np.array(list_randomized) == 0)[0]
            shuffled_labels[idx_match0] = list_str_AEPsych

            idx_match1 = np.where(np.array(list_randomized) == 1)[0]
            shuffled_labels[idx_match1] = list_str_MOCS

            # Append the randomized sequence to the original sequence list
            self.original_sequence.append(shuffled_labels)

        # Create a shallow copy of the original sequence for updates
        self.updated_sequence = list(self.original_sequence)

        # Initialize trial status tracking for all experiments
        self._initialize_trial_status(nExpt)
            
    @staticmethod
    def indices_trial_type(trial_sequence, match_trial):
        """
        Retrieve indices of all elements in the trial sequence that start with the specified trial type.

        Parameters
        ----------
        trial_sequence : list of str
            A list of trial identifiers, for example:
            ['MOCS_190', 'MOCS_191', 'AEPsych_190', 'AEPsych_191', 'AEPsych_192', ...]

        match_trial : str
            The trial type to match, e.g., 'MOCS' or 'AEPsych'.

        Returns
        -------
        indices : list of int
            A list of indices where elements in the trial sequence start with the specified trial type.
        """
        # Use list comprehension for simplicity and efficiency
        return [idx for idx, item in enumerate(trial_sequence) if item.startswith(match_trial)]

#%%
@staticmethod
def shuffle_sobol_trials(val_scaler, nTrials_strat, shuffle_val_scaler_max_strat, seed=None):
    """
    Randomly shuffles Sobol trial scalers for a subset of strategies while keeping
    the rest unchanged.

    Parameters
    ----------
    val_scaler : list of float
        A list of scalers applied to scale trial values up or down.
        
    nTrials_strat : list of int
        A list specifying the number of trials assigned to each scaler in `val_scaler`.
        Must have the same length as `val_scaler`.

    shuffle_val_scaler_max_strat : int
        The number of strategies (i.e., first N elements in `val_scaler`) 
        that should be involved in the shuffling process.
        Example:
            If `val_scaler = [1/4, 2/4, 3/4, 1]` and `shuffle_val_scaler_max_strat = 2`, 
            only `1/4` and `2/4` will be shuffled.

    seed : int, optional
        Random seed for reproducibility. Default is None (randomized each run).

    Returns
    -------
    shuffled_val_scaler : np.ndarray
        A 1D array containing the shuffled values of the first `shuffle_val_scaler_max_strat`
        scalers, followed by the remaining scalers in their original order.
    """

    # Validate that val_scaler and nTrials_strat have matching lengths
    if len(val_scaler) != len(nTrials_strat):
        raise ValueError("Mismatch: val_scaler and nTrials_strat must have the same length.")

    # Validate shuffle_val_scaler_max_strat
    if not isinstance(shuffle_val_scaler_max_strat, int):
        raise ValueError("Invalid type: shuffle_val_scaler_max_strat must be an integer.")

    if shuffle_val_scaler_max_strat < 1:
        raise ValueError("Invalid value: shuffle_val_scaler_max_strat must be at least 1. "
                         "No scalers will be shuffled.")
    
    if shuffle_val_scaler_max_strat >= len(val_scaler):
        raise ValueError(f"Invalid value: shuffle_val_scaler_max_strat cannot exceed {len(val_scaler) - 1}. "
                         "No scalers will be shuffled.")

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Create the array of scalers to be shuffled
    # Example: If val_scaler = [1/4, 2/4, 3/4] and nTrials_strat = [300, 300, 300],
    #          This expands into [1/4, 1/4, ..., 2/4, 2/4, ..., 3/4, 3/4, ...]
    val_scaler_all = np.repeat(val_scaler[:shuffle_val_scaler_max_strat], 
                               nTrials_strat[:shuffle_val_scaler_max_strat])

    # Shuffle the selected subset
    shuffled_val_scaler = rng.permutation(val_scaler_all)

    # Append the remaining unchanged scalers
    shuffled_val_scaler = np.concatenate([
        shuffled_val_scaler, 
        np.repeat(val_scaler[shuffle_val_scaler_max_strat:], 
                  nTrials_strat[shuffle_val_scaler_max_strat:])
    ])

    return shuffled_val_scaler
    
    
#%%        
class ExperimentTrialSequence_suprathres(ExperimentTrialSequence):
    """
    Suprathreshold variant of ExperimentTrialSequence.

    Behaves identically to ExperimentTrialSequence, except for methods
    that are explicitly overridden in this subclass.

    In this suprathreshold version, each MOCS trial involves:
    - One reference stimulus (xref)
    - Two comparison stimuli (x1, x2)
    - A binary response indicating which comparison is judged more different
      from the reference (e.g., 1 for choosing comp#2, 0 for choosing comp#1).

    Parameters
    ----------
    nExpt : int
        Identifier or index for the experiment (e.g., which suprathreshold
        design / condition set is being used).
    Other parameters
        Same as ExperimentTrialSequence.
    """

    def __init__(self, nTrials_AEPsych, pregenerated_MOCS=None,
                 nBlocks=1, break_trials=None, pregenerated_Sobol=None, nExpt = 1):
        
        # Store the experiment index / label
        self.nExpt = nExpt

        if break_trials is None:
            break_trials = []

        # Call parent constructor to do all the standard setup
        super().__init__(
            nTrials_AEPsych=nTrials_AEPsych,
            pregenerated_MOCS=pregenerated_MOCS,
            nBlocks=nBlocks,
            break_trials=break_trials,
            pregenerated_Sobol=pregenerated_Sobol,
        )

        self.nBumpUp_MOCS = [0] * nExpt

    def _initialize_data_MOCS(self):
        """
        Initialize the `data_MOCS` dictionary to store trial-specific data.
    
        If self.nExpt == 1:
            data_MOCS[trial_idx]
        If self.nExpt > 1:
            data_MOCS[(expt_idx, trial_idx)]
        """
        def empty_entry():
            return {
                'xref': None,   # Reference stimulus 
                'x1': None,     # Comparison stimulus #1
                'x2': None,     # Comparison stimulus #2
                'Uref': None,   # Weighted sum of basis functions for reference
                'U1': None,     # Weighted sum of basis functions for comp #1
                'U2': None,     # Weighted sum of basis functions for comp #2
                'signed_diff': None,  # Signed difference of squared Mahalanobis distance
                'pX2': None,    # P(comp #2 is judged more different from ref)
                'binaryResp': None,   # 1 = chose comp #2, 0 = chose comp #1
            }
    
        if getattr(self, "nExpt", 1) == 1:
            # Keys: trial index only
            self.data_MOCS = {
                trial: empty_entry()
                for trial in range(self.nTrials_MOCS)
            }
        else:
            # Keys: (expt index, trial index)
            self.data_MOCS = {
                (expt, trial): empty_entry()
                for expt in range(self.nExpt)
                for trial in range(self.nTrials_MOCS)
            }
            
    def _initialize_trial_status(self, nExpt):
        """
        Initializes the trial_status list.
        """
        trial_status = []
        for i in range(nExpt):
            trial_status.append([[f'Trial_{n}_Cond_{i}_'+self.original_sequence[i][n]] \
                                 for n in range(self.nTrials_total)])
        self.trial_status = trial_status

    def update_data_MOCS(self, expt_idx, trial_idx, xref, x1, x2, binaryResp,
                         Uref=None, U1=None, U2=None, signed_diff=None, pX2=None):
        """
        Update the `data_MOCS` dictionary for a specific trial.

        For simulations, this method updates all fields, including computed values (`Uref`, `U1`, `U2`, 
        `signed_diff`, `pX2`). For actual experiments, only `xref`, `x1`, `x2`, and `binaryResp` are updated.

        Args:
            trial_idx (int): The trial index to update.
            xref (float): Reference stimulus 
            x1 (float): Comparison stimulus #1
            x2 (float): Comparison stimulus #2
            binaryResp (int): Binary response (1 for picking comp#2, 0 for picking comp#1)
            Uref (float, optional): Utility value for the reference stimulus (simulation only).
            U1 (float, optional): Utility value for the comparison stimulus (simulation only).
            U2 (float, optional): Utility value for the comparison stimulus (simulation only).
            signed_diff (float, optional): Signed difference between stimuli (simulation only).
            pX2 (float, optional): Probability of identifying the comp#2 as more different from the reference (simulation only).
        """
        # Choose key type depending on whether there are multiple experiments
        if getattr(self, "nExpt", 1) == 1:
            key = trial_idx
        else:
            key = (expt_idx, trial_idx)
    
        self.data_MOCS[key] = {
            'xref': xref,
            'x1': x1,
            'x2': x2,
            'Uref': Uref,
            'U1': U1,
            'U2': U2,
            'signed_diff': signed_diff,
            'pX2': pX2,
            'binaryResp': binaryResp,
        }
        
    def generate_init_sequence(self, nExpt=1, seed=None):
        """
        Generate a randomized sequence of trial labels for AEPsych and MOCS trials,
        ensuring that trials are evenly distributed across experimental blocks.

        Parameters
        ----------
        nExpt : int, optional
            Number of experiment configurations to generate sequences for. Default is 1.
        seed : int, optional
            Random seed for reproducible randomization. Default is None.

        Returns
        -------
        None
            The method updates the following attributes:
            - `self.original_sequence`: A list of randomized trial sequences for all experiments.
            - `self.updated_sequence`: A copy of the original sequences, which can be modified later.
            - Initializes `self.trial_status` for tracking the status of each trial.

        Notes
        -----
        - Each experimental block contains a balanced split of AEPsych and MOCS trials.
        - Trial labels are formatted as 'AEPsych_{n}' or 'MOCS_{n}', where `{n}` is the trial index.
        """
        # Set up the random seed for reproducibility
        rng = np.random.default_rng(seed)

        # Initialize sequences
        self.original_sequence = []
        self.final_sequence = [[]] * nExpt

        for n in range(nExpt):
            # Create string labels for AEPsych and MOCS trials
            list_str_AEPsych = [f'AEPsych_{n}' for n in range(self.nTrials_AEPsych)]
            list_str_MOCS = [f'MOCS_{n}' for n in range(self.nTrials_MOCS)]

            # Initialize a randomized list to store trial assignments
            list_randomized = []
            for n in range(self.nBlocks):
                # Ensure AEPsych and MOCS trials are evenly split across blocks
                temp_list = [0] * self.nTrials_AEPsych_perBlock + [1] * self.nTrials_MOCS_perBlock
                rng.shuffle(temp_list)  # Shuffle the trial assignments within the block
                list_randomized += temp_list

            # Initialize `shuffled_labels` with the appropriate size and type
            shuffled_labels = np.empty((self.nTrials_total,), dtype=object)

            # Map randomized trial assignments to their corresponding labels
            idx_match0 = np.where(np.array(list_randomized) == 0)[0]
            shuffled_labels[idx_match0] = list_str_AEPsych

            idx_match1 = np.where(np.array(list_randomized) == 1)[0]
            shuffled_labels[idx_match1] = list_str_MOCS
            
            # --- Enforce that the first trial is AEPsych ---
            if not str(shuffled_labels[0]).startswith("AEPsych_"):
                # Find the first AEPsych label and swap it to the front
                swap_idx = next(
                    (i for i, lab in enumerate(shuffled_labels)
                     if str(lab).startswith("AEPsych_")),
                    None
                )
                if swap_idx is not None:
                    shuffled_labels[0], shuffled_labels[swap_idx] = \
                        shuffled_labels[swap_idx], shuffled_labels[0]
            # -----------------------------------------------

            # Append the randomized sequence to the original sequence list
            self.original_sequence.append(shuffled_labels)

        # Create a shallow copy of the original sequence for updates
        self.updated_sequence = list(self.original_sequence)

        # Initialize trial status tracking for all experiments
        self._initialize_trial_status(nExpt)