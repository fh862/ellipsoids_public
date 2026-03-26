#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:32:10 2024

@author: fangfang
"""

import numpy as np
import time  
import threading
import jax.numpy as jnp
import io
import configparser
import warnings
import os
from core import oddity_task

#%%
class SimulateTrialGivenWishart:
    def __init__(self, expt_dim, config_all, gt_Wishart, ref = None, 
                 pseudo_randomize = False, pseudo_randomize_seed = None, 
                 val_scaler = None, customized_val_scaler = None):
        """
        Initializes the trial simulation with specific experimental dimensions and configurations,
        while optionally setting up for pseudo-randomized trials and value scaling.
        
        The initialization also sets up necessary parameters based on the dimensionality of the
        experiment, which dictates the experimental setup and the data required.
    
        Args:
            expt_dim (int): Number of dimensions in the color discrimination experiment. Valid options are:
                            - 2D: Manipulates two dimensions of color.
                            - 3D: Manipulates three dimensions of color.
                            - 4D: Adjusts two dimensions for the reference and their deltas for comparison.
                            - 6D: Adjusts three dimensions for the reference and their deltas for comparison.
            config_all (list of str): Configuration strings for each trial setup.
            gt_Wishart (object): Wishart model predictions.
            ref (list, optional): Reference stimuli configurations, required for 2D and 3D experiments.
            pseudo_randomize (bool): If True, configures trials in a pseudo-random order.
            val_scaler (list, optional): Scaling factors to adjust trial values dynamically.
    
        Raises:
            ValueError: If the experiment dimension is not one of [2, 3, 4, 6], or if reference data is
                        required but not provided, or if the number of scaling factors does not match
                        the expected number based on the trial setup.
        """

        if expt_dim not in (2,3,4,6):
            raise ValueError("Color discrimination experiment must be either 2, 3, 4 or 6d.")
        self.expt_dim   = expt_dim
        self.gt_Wishart = gt_Wishart
        self.config_all = config_all
        self.numConfig  = len(config_all)
        self.ref        = ref
        
        # Validate the need for reference stimuli in 2D and 3D setups
        if (self.expt_dim == 2 or self.expt_dim == 3) and self.ref is None:
            raise ValueError("If the experiment has fixed ref stimulus, you need to specify ref stimuli!")
        
        # Retrieve parameter names, strategy names, and trial numbers
        self.parnames       = self._extract_field_vals('common','parnames', strsplit = True)
        self.strat_names    = self._extract_field_vals('common','strategy_names', strsplit = True)

        # Read per-strategy trial quotas from the config.
        # Exactly ONE of these fields is used in a given config:
        #   - "min_asks" (standard workflow)
        #   - "min_total_tells" (DB-switch / preload workflow)
        # Read per-strategy trial quotas from the config.
        #
        # Normal workflow uses `min_asks` (strategy advances after generating N asks).
        # DB-switch / preload workflow uses `min_total_tells` (strategy advances after
        # accumulating N tells, including preloaded historical trials).
        #
        # We support both by trying `min_asks` first and falling back to
        # `min_total_tells` if the field is not present in the config.
        try:
            self.nTrials_strat = [
                int(self._extract_field_vals(s, "min_asks")) for s in self.strat_names
            ]
        except Exception:
            self.nTrials_strat = [
                int(self._extract_field_vals(s, "min_total_tells")) for s in self.strat_names
            ]
                
        self.nTrials_cumsum = np.cumsum(np.array(self.nTrials_strat))
        self.nTrials        = self.nTrials_cumsum[-1]
        
        # Initialize pseudo-random order of configurations if enabled
        # if pseudo_randomize is false, the return pseudo_order will not be shuffled
        self.pseudo_randomize = pseudo_randomize
        self.pseudo_randomize_seed = pseudo_randomize_seed
        self.pseudo_order = self._create_pseudorandom_order()

        # Initialize lists to store trial data
        self._init_trial_lists()

        # Validate and assign scaling factors, defaulting to 1 if not provided
        if customized_val_scaler is not None: #customized (shuffled) sobol scalers are prioritized
            self.customized_val_scaler = customized_val_scaler
            self.val_scaler = None
        else:
            self.customized_val_scaler = None
            self._validate_val_scaler(val_scaler)
            self.val_scaler = val_scaler

    def _extract_field_vals(self, section, field, strsplit = False):
        """
        Extracts and optionally splits values from config strings using a ConfigParser.
        
        Args:
            section (str): Section name in the configuration.
            field (str): Field name from which to extract the values.
            strsplit (bool): If True, splits the string into a list.
        
        Returns:
            Extracted value, optionally as a list.
        """

        # Create a new ConfigParser object
        config_parser = configparser.ConfigParser()
        # Read the configuration from the string
        config_parser.read_file(io.StringIO(self.config_all[0]))
        #retrieve the field
        result_string = config_parser.get(section, field)
        #if we want to split the string to a list of string
        #e.g., '[strat1, strat2]' -> ['strat1', 'strat2']
        if strsplit:
            # Remove square brackets and split by comma
            result_list = result_string.strip('[]').split(',')
            
            # Strip any leading or trailing whitespace from each element
            result_list = [item.strip() for item in result_list]
            return result_list
        else:
            return result_string

    def _create_pseudorandom_order(self):
        """
        Generate a 2D array specifying the order of trial configurations for each trial set.
    
        If `self.pseudo_randomize` is False, the configurations are repeated in the same
        fixed order across all trial sets (i.e., sequential). If True, each column
        (trial set) is independently shuffled to introduce randomization.
    
        If `self.pseudo_randomize_seed` is provided, it is used to seed the random number
        generator for reproducible shuffling.
    
        Returns:
            np.ndarray: A 2D array of shape (numConfig, nTrials), where each column contains
                        the order of configurations for a given trial set.
        """
        order_temp = np.array(list(range(self.numConfig)))
        pseudo_order = np.tile(order_temp[:, np.newaxis], (1, self.nTrials))
        if not self.pseudo_randomize:
            return pseudo_order
    
        rng = np.random.default_rng(self.pseudo_randomize_seed) \
            if self.pseudo_randomize_seed is not None else np.random.default_rng()
    
        for i in range(pseudo_order.shape[1]):
            rng.shuffle(pseudo_order[:, i])
    
        return pseudo_order
        
    def _validate_val_scaler(self, val_scaler):
        """
        Validates and adjusts the 'val_scaler' list to ensure its length matches 
        'nTrials_cumsum'. If 'val_scaler' is not provided, defaults to a list of 
        1s matching the length of 'nTrials_cumsum'.
        """
        # Check if a val_scaler list was provided
        if val_scaler is not None:
            # Calculate the difference in length between nTrials_cumsum and val_scaler
            len_diff = len(self.nTrials_cumsum) - len(val_scaler)
            # If lengths are equal, no adjustment is needed
            if len_diff == 0:
                self.val_scaler = val_scaler
            else:
                # Adjust val_scaler based on len_diff
                if len_diff > 0:
                    # If there are more trials than scalers, extend val_scaler with 1s
                    self.val_scaler = val_scaler + [1] * len_diff
                    # Use warnings.warn to issue a warning
                    warnings.warn(f"The number of val scalers ({len(val_scaler)}) is less"+\
                                  " than the expected number ({len(self.nTrials_cumsum)}).")
                    print(" Padding with 1's to match the count.")
                else:
                    # If there are fewer trials than scalers, truncate val_scaler
                    self.val_scaler = val_scaler[:len(self.nTrials_cumsum)]
                    warnings.warn(f"The number of val scalers ({len(val_scaler)}) is greater"+\
                                  " than the expected number ({len(self.nTrials_cumsum)}).")
                    print("Trimming the list to match the count.")
        else:
            # Set default val_scaler if none provided
            self.val_scaler = [1] * len(self.nTrials_cumsum)
            
    def _init_trial_lists(self, prefix=""):
        """
        Initializes lists to store data collected during the trial simulation.
    
        Args:
            prefix (str): Prefix for the attribute names (e.g., "pregenerated_").
        """
        # Define the list of attributes to initialize
        attributes = [
            "xref_list",
            "x1_list",
            "Uref_list",
            "U1_list",
            "signed_diff_list",
            "pX1_list",
            "binaryResp_list"
        ]
    
        # Initialize each attribute dynamically with the prefix
        for attr_name in attributes:
            setattr(self, f"{prefix}{attr_name}", [])
            
    def _configure_trial(self, client, trial_counter, config_index):
        """
        Configures or resumes a trial based on its number. If it's the first trial, 
        it configures a new trial. Otherwise, it resumes the existing configuration.
        
        Args:
            client: An instance of the client that communicates with the trial configuration server.
            trial_counter (int): The current trial number, determining if a new configuration or a resume is needed.
            config_index (int): Index to select the specific trial configuration from `self.config_all`.
        """

        if trial_counter == 0:
            client.configure(config_str=self.config_all[config_index],\
                config_name=f"{self.expt_dim}d_colorDiscrimination_idx{config_index}")
        else:
            client.resume(config_name=f"{self.expt_dim}d_colorDiscrimination_idx{config_index}")

    def _derive_xref_x1(self, trial_counter, trial_val, config_index = None):
        """
        Derives normalized reference and comparison stimulus dimensions based on the 
        experiment dimensionality.
        
        Args:
            trial_counter (int): Current trial number to determine the scaling factor.
            trial_val (list): Current trial values that need to be scaled and normalized.
            config_index (int): Index of the current configuration for selecting reference values.
                                Default is None as this is not relevant for 4d/6d experiments.
        
        Returns:
            tuple: 
                xref (jax.numpy.array): Normalized reference stimulus dimensions 
                    (W unit: between -1 and 1).
                x1 (jax.numpy.array): Normalized comparison stimulus dimensions 
                    (W unit: between -1 and 1).
                trial_val_report (list): Adjusted values of the trial used for reporting and 
                    further calculations.
        """

        if self.expt_dim == 2 or self.expt_dim == 3: #interleaved 2D or 3D experiment
            # Apply value scaling to adjust comparison values based on experimental progress.
            trial_val_report = self._apply_val_scaling(trial_counter, trial_val)
            
            # Normalize reference and comparison stimulus dimensions.
            xref = jnp.array(self.ref[config_index])               
            x1 = jnp.array(trial_val_report) + xref
        else: #full 4D or 6D experiment
            # Separate the reference and comparison values for higher dimensions.
            trial_val_ref = trial_val[:self.expt_dim//2]
            trial_val_delta_comp = trial_val[(self.expt_dim//2):]
            trial_val_delta_comp_scaled = self._apply_val_scaling(\
                trial_counter, trial_val_delta_comp)
            #put reference and delta values to a list and report it back to the client
            trial_val_report = trial_val_ref + trial_val_delta_comp_scaled #lists
            xref = jnp.array(trial_val_ref)
            
            # Add deltas on top of the ref stimulus to derive comparison stimulus
            x1 = jnp.array(trial_val_delta_comp_scaled) + xref
        return xref, x1, trial_val_report
    
    def _apply_val_scaling(self, trial_counter, trial_val):
        """
        Scales the values of trial parameters based on a predefined scaling sequence. 
        The scaling adjusts trial values to be closer to or further from the reference values, 
        depending on the trial's progress and the dimensionality of the experiment.
    
        Args:
            trial_counter (int): The current number of the trial, used to determine the scaling factor.
            trial_val (list): List of values from the current trial to be scaled. These represent
                              delta values of the comparison stimulus    
        Returns:
            list: Scaled trial values, where each value has been adjusted according to the 
            specified scaling logic.
        """
        
        # Determine the scaling factor index based on how many trials have been completed.
        if self.customized_val_scaler is None:
            val_scaler_idx = np.searchsorted(self.nTrials_cumsum, trial_counter, side='right')
            val_scaler_i = self.val_scaler[val_scaler_idx]
        else:
            val_scaler_i = self.customized_val_scaler[trial_counter]
        #initialize
        trial_val_scaled =[]
        
        # Apply scaling for 2D and 3D experiments where scaling is relative to a reference value.
        if self.expt_dim == 2 or self.expt_dim == 3:
            len_par = len(self.parnames)
        else:
            # Apply direct scaling for 4D and 6D experiments because they are delta values!
            len_par = len(self.parnames)//2
            
        for i in range(len_par): 
            # Directly scale each parameter, not relative to any reference.
            trial_val_scaled_i = trial_val[i]*val_scaler_i
            trial_val_scaled.append(trial_val_scaled_i)
        return trial_val_scaled
  
    def _predict_probability_correct(self, xref, x1):
        """
        Predicts the probability of correctly identifying the odd stimulus in a 
        trial using a given model. The method utilizes the model's basis functions
        to compute weighted sums for both reference and comparison stimuli, and 
        calculates the probability of a correct response.
        
        Args:
            xref (array-like): Normalized dimensions of the reference stimulus,
                               values ranging from -1 to 1.
            x1 (array-like): Normalized dimensions of the comparison stimulus,
                             similar scale as xref.
        """
        # compute weighted sum of basis function at the reference
        Uref = self.gt_Wishart.model.compute_U(self.gt_Wishart.W_est, xref)
        # compute weighted sum of basis function at the comparison
        U1   = self.gt_Wishart.model.compute_U(self.gt_Wishart.W_est, x1)
        
        # Simulate the decision-making process for identifying the odd stimulus.
        signed_diff = oddity_task.simulate_oddity_one_trial(
            (xref, x1, Uref, U1), self.gt_Wishart.opt_key, 
            self.gt_Wishart.opt_params['mc_samples'],
            self.gt_Wishart.model.diag_term)
        # Compute the probability of correctly identifying the odd stimulus using the signed difference.
        pX1 = oddity_task.approx_cdf_one_trial(0.0, signed_diff,
                                               self.gt_Wishart.opt_params['bandwidth'])
        
        # Generate a random response based on the predicted probability and send feedback to the client
        randNum = np.random.rand() 
        binaryResp = int(randNum < pX1)
        
        return Uref, U1, signed_diff, pX1, binaryResp
    
    def _update_trial_lists(self, xref, x1, binaryResp, 
                            Uref = None, U1 = None, signed_diff = None, pX1 = None, 
                            prefix=""):
        """
        Append elements to the lists to store data collected during the trial simulation.
    
        Args:
            xref, x1, Uref, U1, signed_diff, pX1, binaryResp: Data to append.
            prefix (str): Prefix for the attribute names (e.g., "pregenerated_").
        """
        # Define the attribute names and their corresponding values
        attributes = [
            (f"{prefix}xref_list", xref),
            (f"{prefix}x1_list", x1),
            (f"{prefix}Uref_list", Uref),
            (f"{prefix}U1_list", U1),
            (f"{prefix}signed_diff_list", signed_diff),
            (f"{prefix}pX1_list", pX1),
            (f"{prefix}binaryResp_list", binaryResp)
        ]
    
        # Append the values to the corresponding attributes
        for attr_name, value in attributes:
            getattr(self, attr_name).append(value)

    def _stack_them_all(self, stacking_ax=0, prefix=""):
        """
        Aggregates all the trial data lists into single arrays for further analysis or storage.
        
        Args:
            stacking_ax (int): Axis along which to stack the data.
            prefix (str): Prefix for the attribute names (e.g., "pregenerated_").
        """
        # Define the mapping of attributes to be stacked
        attributes = {
            "xref_list": "xref_all",
            "x1_list": "x1_all",
            "Uref_list": "Uref_all",
            "U1_list": "U1_all",
            "signed_diff_list": "signed_diff_all",
            "pX1_list": "pX1_all",
            "binaryResp_list": "binaryResp_all"
        }
    
        # Loop through attributes and process dynamically
        for list_attr, array_attr in attributes.items():
            list_attr_name = f"{prefix}{list_attr}"
            array_attr_name = f"{prefix}{array_attr}"
            
            # Retrieve the list values
            list_values = getattr(self, list_attr_name)
    
            # Handle binaryResp_list (uses jnp.array directly)
            if list_attr == "binaryResp_list":
                setattr(self, array_attr_name, jnp.array(list_values))
            else:
                # Stack only if there are no None values in the list
                if all(v is not None for v in list_values):
                    setattr(self, array_attr_name, jnp.stack(list_values, axis=stacking_ax))
                else:
                    # Skip stacking if any None is found
                    continue

    def _monitor_time_insert_pregenerated_trials(self, start_time, max_wait_time,
                                                 pregenerated_trials, stop_event):
        # Function to monitor elapsed time and perform side tasks
        while not stop_event.is_set():  # Exit if stop_event is set
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time:
                print(f"Deadline exceeded ({elapsed_time:.2f}s). Running a pre-generated trial...")
                
                # Get the stimulus information
                xref = pregenerated_trials['xref'][self.pregenerated_trial_counter]
                x1 = pregenerated_trials['x1'][self.pregenerated_trial_counter]
                
                # Simulate response based on the reference and comparison stimuli
                Uref, U1, signed_diff, pX1, binaryResp = self._predict_probability_correct(xref, x1)
                print(f"Simulated responses (#trial {self.pregenerated_trial_counter}): {binaryResp}")
                
                #store simulated responses
                self._update_trial_lists(xref, x1, binaryResp, 
                                         Uref, U1, signed_diff, pX1, 
                                         prefix = "pregenerated_")
                
                # Increase trial counter
                self.pregenerated_trial_counter += 1
                
                # Reset the start time for the next pre-generated trial
                start_time = time.time()
    
                # Break the loop if all pre-generated trials are exhausted
                if self.pregenerated_trial_counter >= len(pregenerated_trials['xref']):
                    print("All pre-generated trials have been used.")
                    break
            time.sleep(0.05)  # Check every 50 ms
   
    def run_simulation(self, client, pregenerated_trials = None, max_wait_time = None):
        """
        Simulates color-discrimination responses using AEPsych and ground truth data. 
        Ground truth can be based on Wishart fits to pilot data or CIELAB thresholds.

        This method interleaves AEPsych trials with pregenerated trials. If 
        AEPsych takes too long to generate a trial, a pregenerated trial is used instead. 
        Unlike the run_simulation_wMOCSinserted method, this approach does not require 
        MOCS trials as pregenerated trials. The pregenerated trials can include trials 
        near thresholds or other predefined configurations. Additionally, this method 
        does not rely on a pseudorandomized pregenerated trial sequence but inserts 
        pregenerated trials dynamically when needed.

        Args:
            client (object): AEPsych client instance used to configure and query trials.
            pregenerated_trials (dict): A dictionary containing pregenerated trial pairs 
                                        of xref and x1.
            max_wait_time (float): Maximum allowed wait time for AEPsych trial generation. 
                                   If exceeded, a pregenerated trial is used.
        """
        # Check if a time limit for trial generation is specified
        if max_wait_time is not None:
            # If a time limit is provided, ensure pregenerated_trials are also provided
            if pregenerated_trials is None:
                raise ValueError(
                    "A max_wait_time was specified, but no pregenerated_trials were provided. "
                    "Please supply pregenerated_trials to use during long wait times."
                )
            else:
                # Initialize a counter to track the number of pregenerated trials used
                self.pregenerated_trial_counter = 0
                # Initialize lists to store data specifically for pregenerated trials
                self._init_trial_lists(prefix="pregenerated_")     
                # Set a flag to indicate that pregenerated trials are available for use
                flag_insert_pregen = True
        else:
            # If no time limit is provided, pregenerated trials are not used
            flag_insert_pregen = False
    
        trial_counter = 0
        time_elapsed = []
        finished = False
        while not finished:
            print(trial_counter)
            
            for i in range(self.numConfig):
                ii = self.pseudo_order[i, trial_counter]
                
                # Start the timer for the trial
                start_time = time.time()
                
                self._configure_trial(client, trial_counter, ii)
                
                if flag_insert_pregen:
                    # Create an event for stopping the monitoring thread
                    stop_event = threading.Event()
                    # Start monitoring in a separate thread
                    pregenerated_trial_counter_before = self.pregenerated_trial_counter
                    monitor_thread = threading.Thread(target=self._monitor_time_insert_pregenerated_trials,
                                                      args=(start_time, 
                                                            max_wait_time, 
                                                            pregenerated_trials,
                                                            stop_event))  # Pass stop_event
                    monitor_thread.start()
                
                # Request a new trial configuration from the AEPsych client
                trial_AEPsych = client.ask()
                
                if flag_insert_pregen:
                    stop_event.set()  # Signal that `client.ask()` has completed
                    monitor_thread.join()  # Join the monitoring thread to clean up
                #End the timer
                end_time = time.time()
                
                # Extract stimulus dimensions from the trial configuration
                trial_val = []
                for s in self.parnames:
                    trial_val.append(trial_AEPsych["config"][s][0])
                
                # Compute xref and x1
                xref, x1, trial_val_report = self._derive_xref_x1(trial_counter, trial_val, ii)
                
                # Simulate one single trial
                Uref, U1, signed_diff, pX1, binaryResp = self._predict_probability_correct(xref, x1)
                
                # Report back to AEPsych client
                client.tell(config=dict(zip(self.parnames, trial_val_report)),
                            outcome=binaryResp)
                            
                self._update_trial_lists(xref, x1, binaryResp, Uref, U1, signed_diff, pX1)
                
                # calculate elapsed time
                if flag_insert_pregen:
                    pregenerated_trial_counter_after = self.pregenerated_trial_counter
                    used_pregenerated_trial = pregenerated_trial_counter_after - pregenerated_trial_counter_before
                    trial_duration = end_time - start_time - used_pregenerated_trial * max_wait_time
                else:
                    trial_duration = end_time - start_time
                time_elapsed.append(trial_duration)  # Record the time for this trial
                
                # Check if the experiment is finished based on the AEPsych server's response
                finished = trial_AEPsych["is_finished"]
                
            # After all reference stimuli were tested
            trial_counter += 1
        
        # Aggregate all the trial data lists into single arrays
        self._stack_them_all()
        if flag_insert_pregen:
            self._stack_them_all(prefix="pregenerated_")
            
        # Record time
        self.time_elapsed = time_elapsed

#%%            
    def _monitor_time_insert_MOCS_trials(self, start_time, max_wait_time,
                                         trial_sequence, expt_counter, trial_counter, 
                                         stop_event, mocs_triggered):
        """
        Monitors elapsed time and performs tasks such as running MOCS trials 
        if the deadline is exceeded.

        Parameters:
        - start_time: Time when the monitoring started.
        - max_wait_time (list): Time limits for AEPsych trial generation. The first 
                        missed presentation uses max_wait_time[0]; subsequent misses use 
                        max_wait_time[1] (accounting for response and inter-trial interval delays).
        - trial_sequence: The trial sequence object to be updated with trial results.
        - expt_counter (int): Counter to track the experiment configuration index.
        - trial_counter (int): Counter to track the current trial index.
        - stop_event: An event to signal the thread to stop monitoring.
        - mocs_triggered: A threading event to indicate a MOCS trial was executed.

        Behavior:
        - Checks if the elapsed time exceeds the current threshold in `max_wait_time`.
        - If exceeded, it runs a MOCS trial by updating the trial sequence with pre-generated data.
        - Updates the trial status and flags the MOCS trial as completed.
        - Resets the start time for the next monitoring cycle and increments the count of MOCS trials executed.

        Returns:
        - None. Updates are made to the `trial_sequence` object directly.
        """      
        num_bumped_up_MOCS = 0
        while not stop_event.is_set():  # Exit if stop_event is set
            elapsed_time = time.time() - start_time
            max_wait_time_ii = max_wait_time[0] if num_bumped_up_MOCS == 0 else max_wait_time[-1]
            
            #if the time elapsed exceeds the max wait time
            if elapsed_time > max_wait_time_ii:
                #find the next available MOCS trial in the list
                print(f"Deadline exceeded ({elapsed_time:.2f}s). Running a pre-generated MOCS trial...")
                trial_replacement_idx_MOCSlist, trial_placement_idx_originallist, trial_placement_id = \
                    trial_sequence.bump_up_one_MOCS_trial(expt_counter, trial_counter) 
    
                # Get the stimulus information
                xref = trial_sequence.pregenerated_MOCS['xref'][trial_replacement_idx_MOCSlist]
                x1 = trial_sequence.pregenerated_MOCS['x1'][trial_replacement_idx_MOCSlist]
    
                # Simulate response based on the reference and comparison stimuli
                _, _, _, _, binaryResp = self._predict_probability_correct(xref, x1)
                print(f"Simulated responses (#trial {trial_replacement_idx_MOCSlist}): {binaryResp}")
    
                # Store simulated responses
                trial_sequence.update_data_MOCS(trial_replacement_idx_MOCSlist,
                                                xref, x1, binaryResp)
                # Set the status of the MOCS trial to 'completed'
                # it is trial counter because the sequence has been updated 
                trial_sequence.set_trial_status(expt_counter, trial_placement_idx_originallist, "Completed")
                
                # Set the flag to indicate a MOCS trial was run
                mocs_triggered.set() 

                # Reset the start time for the next pre-generated trial
                start_time = time.time()
                num_bumped_up_MOCS += 1
                trial_sequence.nBumpUp_MOCS += 1
                
                trial_sequence.final_sequence[expt_counter].append(trial_placement_id)
            time.sleep(0.05)  # Check every 50 ms
        
        #%%
    def run_simulation_wMOCSinserted(self, client, trial_sequence, max_wait_time=[2.4, 3.6]):
        """
        Simulates color-discrimination responses using AEPsych and ground truth data. 
        Ground truth can be based on Wishart fits to pilot data or CIELAB thresholds.

        This method interleaves AEPsych trials with pre-generated MOCS trials. If 
        AEPsych takes too long to generate a trial, a MOCS trial is inserted as a 
        fallback. The interleaving sequence is pre-generated in the `trial_sequence` class.

        Args:
            client (object): AEPsych client instance used to configure and query trials.
            trial_sequence (object): Contains trial sequence information for interleaving
                AEPsych and pre-generated MOCS trials.
            max_wait_time (list): Time limits for AEPsych trial generation. The first 
                missed presentation uses max_wait_time[0]; subsequent misses use 
                max_wait_time[1] (accounting for response and inter-trial interval delays).
        """    
        time_elapsed = []  # List to store elapsed time for AEPsych trials
        trial_counter = 0  # Counter to track the current trial number

        while trial_counter < trial_sequence.nTrials_total:            
            for expt_idx in range(self.numConfig):  
                
                # Check if the trial is already completed
                current_trial_status = trial_sequence.trial_status[expt_idx][trial_counter]
                if "Completed" in current_trial_status or "Completed_in_time" in current_trial_status:
                    # If already completed, mark as skipped and move to the next trial
                    print(f"Skipping trial {trial_counter} as it is already completed.")
                    new_status = current_trial_status + ["Skipped"]
                    trial_sequence.trial_status[expt_idx][trial_counter] = list(new_status)
                    trial_counter += 1
                    continue
                
                # Retrieve trial identity (e.g., 'MOCS_1', 'AEPsych_1')
                trial_identity = trial_sequence.updated_sequence[expt_idx][trial_counter]
                # Extract trial type ('MOCS' or 'AEPsych') and trial index
                trial_type, trial_idx = trial_identity.split('_')
                trial_idx = int(trial_idx)
                print(f"Trial #{trial_counter}: {trial_identity}")
                
                if trial_type == 'MOCS': 
                    # Retrieve xref and x1 for MOCS trials
                    xref = trial_sequence.pregenerated_MOCS['xref'][trial_idx]
                    x1 = trial_sequence.pregenerated_MOCS['x1'][trial_idx]
                    
                    # Simulate the trial and record the binary response
                    _, _, _, _, binaryResp = self._predict_probability_correct(xref, x1)
                    # Update the MOCS trial data with the simulated response
                    trial_sequence.update_data_MOCS(trial_idx, xref, x1, binaryResp)
                    # Mark the trial as completed within the time window
                    trial_sequence.set_trial_status(expt_idx, trial_counter, "Completed_in_time")
                    
                elif trial_type == 'AEPsych':
                    # Retrieve the trial order for AEPsych
                    ii = self.pseudo_order[expt_idx, trial_idx]
                    # Configure the trial for the AEPsych client
                    self._configure_trial(client, trial_idx, ii)
                    
                    # Flag to track if a MOCS trial was inserted
                    mocs_triggered = threading.Event()
                    # Start timing the AEPsych trial
                    start_time = time.time()  
                    # Create an event to stop the monitoring thread
                    stop_event = threading.Event()
                    # Start monitoring trial generation in a separate thread
                    monitor_thread = threading.Thread(target=self._monitor_time_insert_MOCS_trials,
                                                      args=(start_time, 
                                                            max_wait_time, 
                                                            trial_sequence,
                                                            expt_idx,
                                                            trial_counter,
                                                            stop_event,
                                                            mocs_triggered))  
                    monitor_thread.start()
                
                    # Request a new trial configuration from AEPsych
                    trial_AEPsych = client.ask()
                    
                    # Once AEPsych finishes, stop the monitoring thread
                    stop_event.set()
                    monitor_thread.join()  # Ensure the thread finishes before continuing
                    end_time = time.time()
                    
                    # Extract stimulus dimensions for the trial
                    trial_val = [trial_AEPsych["config"][s][0] for s in self.parnames]
                    # Simulate the trial
                    xref, x1, trial_val_report = self._derive_xref_x1(trial_idx, trial_val, ii)
                    _, _, _, _, binaryResp = self._predict_probability_correct(xref, x1)
                
                    # Report the result back to AEPsych
                    client.tell(config=dict(zip(self.parnames, trial_val_report)),
                                outcome=binaryResp)
                                
                    # Update trial-related data
                    self._update_trial_lists(xref, x1, binaryResp)
                    
                    # Record elapsed time for the trial
                    trial_duration = end_time - start_time
                    time_elapsed.append(trial_duration)
                    
                    # Mark trial as completed with appropriate status
                    if mocs_triggered.is_set():
                        trial_sequence.set_trial_status(expt_idx, trial_counter, 
                                                        f"Elapsed_time_{trial_duration:.4f}")
                        trial_sequence.set_trial_status(expt_idx, trial_counter, 
                                                        "Completed")
                    else:
                        trial_sequence.set_trial_status(expt_idx, trial_counter, 
                                                        "Completed_in_time")
                
                # Record the actual trial sequence that was executed
                trial_sequence.final_sequence[expt_idx].append(trial_identity)
                # Increment trial counter
                trial_counter += 1
  
        # Aggregate trial data lists into single arrays
        self._stack_them_all()

        # Record the total elapsed times for AEPsych trials only
        self.time_elapsed = time_elapsed
    
        # Return the updated trial_sequence with new data
        return trial_sequence
    
#%%
class SimulateTrialGivenWishart_suprathres(SimulateTrialGivenWishart):
    def __init__(self, expt_dim, config_all, gt_Wishart, ref = None, 
                 pseudo_randomize = False, pseudo_randomize_seed = None, 
                 val_scaler = None, customized_val_scaler = None,
                 comp1 = None, nTrials_total = None):
        # Handle subclass-specific stuff
        self.comp1 = comp1
        self.nTrials_total = nTrials_total
        
        # Call base-class init with the shared arguments
        super().__init__(expt_dim, config_all, gt_Wishart,
                         ref=ref,
                         pseudo_randomize=pseudo_randomize,
                         pseudo_randomize_seed=pseudo_randomize_seed,
                         val_scaler=val_scaler,
                         customized_val_scaler=customized_val_scaler)
    
        # For 2D or 3D experiments, a fixed comp1 must be provided
        if self.expt_dim in (2, 3) and self.comp1 is None:
            raise ValueError(
                "comp1 must be provided for 2D or 3D suprathreshold experiments."
            )
   
    def _create_pseudorandom_order(self):
        """
        Generate a 2D array specifying the order of trial configurations for each trial set.
    
        If `self.pseudo_randomize` is False, the configurations are repeated in the same
        fixed order across all trial sets (i.e., sequential). If True, each column
        (trial set) is independently shuffled to introduce randomization.
    
        If `self.pseudo_randomize_seed` is provided, it is used to seed the random number
        generator for reproducible shuffling.
    
        Returns:
            np.ndarray: A 2D array of shape (numConfig, nTrials), where each column contains
                        the order of configurations for a given trial set.
        """
        order_temp = np.array(list(range(self.numConfig)))
        pseudo_order = np.tile(order_temp[:, np.newaxis], (1, self.nTrials_total))
        if not self.pseudo_randomize:
            return pseudo_order
    
        rng = np.random.default_rng(self.pseudo_randomize_seed) \
            if self.pseudo_randomize_seed is not None else np.random.default_rng()
    
        for i in range(pseudo_order.shape[1]):
            rng.shuffle(pseudo_order[:, i])
    
        return pseudo_order

    def _init_trial_lists(self, prefix=""):
        """
        Initializes lists to store data collected during the trial simulation.
    
        Args:
            prefix (str): Prefix for the attribute names (e.g., "pregenerated_").
        """
        # Define the list of attributes to initialize
        attributes = [
            "xref_list",
            "x1_list",
            "x2_list",
            "Uref_list",
            "U1_list",
            "U2_list",
            "signed_diff_list",
            "pX2_list",
            "binaryResp_list"
        ]
    
        # Initialize each attribute dynamically with the prefix
        for attr_name in attributes:
            setattr(self, f"{prefix}{attr_name}", [])   
            
    def _update_trial_lists(self, xref, x1, x2, binaryResp, 
                            Uref = None, U1 = None, U2 = None, 
                            signed_diff = None, pX2 = None, 
                            prefix=""):
        """
        Append elements to the lists to store data collected during the trial simulation.
    
        Args:
            xref, x1, Uref, U1, signed_diff, pX1, binaryResp: Data to append.
            prefix (str): Prefix for the attribute names (e.g., "pregenerated_").
        """
        # Define the attribute names and their corresponding values
        attributes = [
            (f"{prefix}xref_list", xref),
            (f"{prefix}x1_list", x1),
            (f"{prefix}x2_list", x2),
            (f"{prefix}Uref_list", Uref),
            (f"{prefix}U1_list", U1),
            (f"{prefix}U2_list", U2),
            (f"{prefix}signed_diff_list", signed_diff),
            (f"{prefix}pX2_list", pX2),
            (f"{prefix}binaryResp_list", binaryResp)
        ]
    
        # Append the values to the corresponding attributes
        for attr_name, value in attributes:
            getattr(self, attr_name).append(value)

    def _stack_them_all(self, stacking_ax=0, prefix=""):
        """
        Aggregates all the trial data lists into single arrays for further analysis or storage.
        
        Args:
            stacking_ax (int): Axis along which to stack the data.
            prefix (str): Prefix for the attribute names (e.g., "pregenerated_").
        """
        # Define the mapping of attributes to be stacked
        attributes = {
            "xref_list": "xref_all",
            "x1_list": "x1_all",
            "x2_list": "x2_all",
            "Uref_list": "Uref_all",
            "U1_list": "U1_all",
            "U2_list": "U2_all",
            "signed_diff_list": "signed_diff_all",
            "pX2_list": "pX2_all",
            "binaryResp_list": "binaryResp_all"
        }
    
        # Loop through attributes and process dynamically
        for list_attr, array_attr in attributes.items():
            list_attr_name = f"{prefix}{list_attr}"
            array_attr_name = f"{prefix}{array_attr}"
            
            # Retrieve the list values
            list_values = getattr(self, list_attr_name)
    
            # Handle binaryResp_list (uses jnp.array directly)
            if list_attr == "binaryResp_list":
                setattr(self, array_attr_name, jnp.array(list_values))
            else:
                # Stack only if there are no None values in the list
                if all(v is not None for v in list_values):
                    setattr(self, array_attr_name, jnp.stack(list_values, axis=stacking_ax))
                else:
                    # Skip stacking if any None is found
                    continue

    def _derive_xref_x1_x2(self, trial_counter, trial_val, config_index = None):
        """
        Derives normalized reference and comparison stimulus dimensions based on the 
        experiment dimensionality.
        
        Args:
            trial_counter (int): Current trial number to determine the scaling factor.
            trial_val (list): Current trial values that need to be scaled and normalized.
            config_index (int): Index of the current configuration for selecting reference values.
                                Default is None as this is not relevant for 4d/6d experiments.
        
        Returns:
            tuple: 
                xref (jax.numpy.array): Normalized reference stimulus 
                    (W unit: between -1 and 1).
                x1 (jax.numpy.array): Normalized comparison stimulus #1 
                    (W unit: between -1 and 1).
                x2 (jax.numpy.array): Normalized comparison stimulus #2
                    (W unit: between -1 and 1).
                trial_val_report (list): Adjusted values of the trial used for reporting and 
                    further calculations.
        """
        
        if self.expt_dim == 2 or self.expt_dim == 3:
            xref = jnp.array(self.ref[config_index])    
            x1 = jnp.array(self.comp1[config_index])
            trial_val_delta_comp2_scaled = self._apply_val_scaling(trial_counter, trial_val)
            x2 = jnp.array(trial_val_delta_comp2_scaled) + xref
            trial_val_report= list(trial_val_delta_comp2_scaled)
            
        elif self.expt_dim == 4 or self.expt_dim == 6: #full 4D or 6D experiment
            # Normalize reference and comparison stimulus dimensions.
            xref = jnp.array(self.ref[config_index])    
            
            # Separate the reference and comparison values for higher dimensions.
            trial_val_delta_comp1 = trial_val[:self.expt_dim//2]
            trial_val_delta_comp2 = trial_val[(self.expt_dim//2):]
            trial_val_delta_comp2_scaled = self._apply_val_scaling(\
                trial_counter, trial_val_delta_comp2)
            #put reference and delta values to a list and report it back to the client
            trial_val_report = trial_val_delta_comp1 + trial_val_delta_comp2_scaled #lists
            
            # Add deltas on top of the ref stimulus to derive comparison stimulus
            x1 = jnp.array(trial_val_delta_comp1) + xref
            x2 = jnp.array(trial_val_delta_comp2_scaled) + xref
        else: 
            print('not supported right now!')
        return xref, x1, x2, trial_val_report

    def _predict_probability_correct(self, xref, x1, x2):
        """
        Predicts the probability of correctly identifying the odd stimulus in a 
        trial using a given model. The method utilizes the model's basis functions
        to compute weighted sums for both reference and comparison stimuli, and 
        calculates the probability of a correct response.
        
        Args:
            xref (array-like): Normalized dimensions of the reference stimulus,
                               values ranging from -1 to 1.
            x1 (array-like): Normalized dimensions of the comparison stimulus,
                             similar scale as xref.
        """
        
        # compute weighted sum of basis function at the reference
        Uref = self.gt_Wishart.model.compute_U(self.gt_Wishart.W_est, xref)
        # compute weighted sum of basis function at the comparison
        U1   = self.gt_Wishart.model.compute_U(self.gt_Wishart.W_est, x1)
        U2   = self.gt_Wishart.model.compute_U(self.gt_Wishart.W_est, x2)
        
        # Simulate the decision-making process for identifying the odd stimulus.
        signed_diff = oddity_task.simulate_oddity_suprathres_one_trial(
            (xref, x1, x2, Uref, U1, U2), self.gt_Wishart.opt_key, 
            self.gt_Wishart.opt_params['mc_samples'],
            self.gt_Wishart.model.diag_term)
        # Compute the probability of correctly identifying the odd stimulus using the signed difference.
        pX2 = oddity_task.approx_cdf_one_trial(0.0, signed_diff,
                                               self.gt_Wishart.opt_params['bandwidth'])
        
        # Generate a random response based on the predicted probability and send feedback to the client
        randNum = np.random.rand() 
        binaryResp = int(randNum < pX2)
        
        return Uref, U1, U2, signed_diff, pX2, binaryResp
        
    def run_simulation(self, client, pregenerated_trials = None, max_wait_time = None):
        """
        Simulates color-discrimination responses using AEPsych and ground truth data. 
        Ground truth can be based on Wishart fits to pilot data or CIELAB thresholds.

        This method interleaves AEPsych trials with pregenerated trials. If 
        AEPsych takes too long to generate a trial, a pregenerated trial is used instead. 
        Unlike the run_simulation_wMOCSinserted method, this approach does not require 
        MOCS trials as pregenerated trials. The pregenerated trials can include trials 
        near thresholds or other predefined configurations. Additionally, this method 
        does not rely on a pseudorandomized pregenerated trial sequence but inserts 
        pregenerated trials dynamically when needed.

        Args:
            client (object): AEPsych client instance used to configure and query trials.
            pregenerated_trials (dict): A dictionary containing pregenerated trial pairs 
                                        of xref and x1.
            max_wait_time (float): Maximum allowed wait time for AEPsych trial generation. 
                                   If exceeded, a pregenerated trial is used.
        """    
        trial_counter = 0
        time_elapsed = []
        finished = False
        while not finished:
            print(trial_counter)
            
            for i in range(self.numConfig):
                ii = self.pseudo_order[i, trial_counter]
                
                # Start the timer for the trial
                start_time = time.time()
                
                self._configure_trial(client, trial_counter, ii)
                
                # Request a new trial configuration from the AEPsych client
                trial_AEPsych = client.ask()
                
                #End the timer
                end_time = time.time()
                
                # Extract stimulus dimensions from the trial configuration
                trial_val = []
                for s in self.parnames:
                    trial_val.append(trial_AEPsych["config"][s][0])
                
                # Compute xref and x1
                xref, x1, x2, trial_val_report = self._derive_xref_x1_x2(trial_counter, trial_val, ii)
                
                # Simulate one single trial
                Uref, U1, U2, signed_diff, pX2, binaryResp = self._predict_probability_correct(xref, x1, x2)
                
                # Report back to AEPsych client
                #1: pick comp#2 as more different; 0: pick comp#1 as more different
                client.tell(config=dict(zip(self.parnames, trial_val_report)),
                            outcome=binaryResp)
                            
                self._update_trial_lists(xref, x1, x2, binaryResp, Uref, U1, U2, signed_diff, pX2)
                
                # calculate elapsed time
                trial_duration = end_time - start_time
                time_elapsed.append(trial_duration)  # Record the time for this trial
                
                # Check if the experiment is finished based on the AEPsych server's response
                finished = trial_AEPsych["is_finished"]
                
            # After all reference stimuli were tested
            trial_counter += 1
        
        # Aggregate all the trial data lists into single arrays
        self._stack_them_all()
            
        # Record time
        self.time_elapsed = time_elapsed
        

