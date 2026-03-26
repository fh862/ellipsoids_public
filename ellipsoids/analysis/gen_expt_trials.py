#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:10:33 2025

@author: fangfang
"""

import numpy as np
import time  
import threading
import dill as pickled
import os
from analysis.sim_trials import SimulateTrialGivenWishart

#%%
class ExptTrialGeneration(SimulateTrialGivenWishart):
    """
    ExptTrialGeneration extends SimulateTrialGivenWishart to allow for additional 
    functionalities in trial generation, such as custom trial sequences, 
    data storage, or extended experimental configurations.

    Inherits all methods and properties from SimulateTrialGivenWishart.

    Additional parameters and methods can be added here as needed.
    """
    
    def __init__(self, expt_dim, config_all, flag_noMore_MOCS = False, 
                 max_shifts_MOCS = 4, gt_Wishart = None, ref = None, 
                 pseudo_randomize=False, pseudo_randomize_seed = None, 
                 val_scaler=None, customized_val_scaler = None, 
                 communicator = None, color_thres = None):
        """
        Initializes ExptTrialGeneration while inheriting from SimulateTrialGivenWishart.

        Args:
            expt_dim (int): Number of dimensions in the color discrimination experiment.
            flag_noMore_MOCS (bool): whether we would like to use MOCS as fallback trials.
                if we do, set it to False, otherwise, set it to True
            max_shifts_MOCS: the maximum trials ahead of MOCS relative to AEPsych that's
                allowed when AEPsych is taking its time to determine the next trial
            config_all (list of str): Configuration strings for each trial setup.
            ref (list, optional): Reference stimuli configurations.
            pseudo_randomize (bool, optional): If True, configures trials in a pseudo-random order.
            val_scaler (list, optional): Scaling factors to adjust trial values dynamically.
        """
        # Call parent class constructor to inherit initialization
        super().__init__(expt_dim, config_all, gt_Wishart, ref = ref, 
                         pseudo_randomize = pseudo_randomize, 
                         pseudo_randomize_seed = pseudo_randomize_seed,
                         val_scaler = val_scaler, 
                         customized_val_scaler = customized_val_scaler)
        
        # Store additional parameters specific to ExptTrialGeneration
        # set up communicator via a text file in a network disk
        self.communicator = communicator
        self.color_thres = color_thres
        # this variables keep tracks how many MOCS we have bumped up
        # if it exceed the max, then we do not continue bumping up more MOCS trials
        # that would disrupt the balance between MOCS and AEPsych trials too much
        self.max_shifts_MOCS = max_shifts_MOCS
        self.keep_track_trials_finished = 0 #AEPsych and MOCS trials (pregenerated sobol trials are not included)
        self.keep_track_trials_finished_pregenSobol = 0
        self.flag_noMore_MOCS = flag_noMore_MOCS
        
    def _convert_W_to_RGB(self, x_Wspace):
        """
        Convert coordinates from W-space to RGB space.
    
        For experiments:
        - 2D (interleaved) or 4D (full) on the isoluminant plane:
            Append a 1 to the W-space coordinate to form a homogeneous coordinate,
            then apply the 2D W-to-RGB transformation matrix (M_2DWToRGB).
            This maps the quadrilateral isoluminant plane in W-space to a square
            in RGB space.
    
        - 3D (interleaved) or 6D (full) in the RGB cube:
            Apply a direct linear transformation using W_unit_to_N_unit,
            which maps the cube bounded by [-1, 1] in W-space
            to the cube bounded by [0, 1] in RGB space.
        """
        if self.expt_dim == 2 or self.expt_dim == 4:
            # Append 1 for homogeneous coordinate representation
            xWspace_append1 = np.append(x_Wspace, 1)
            x_rgb = self.color_thres.M_2DWToRGB @ xWspace_append1
        else:
            # Direct scaling from [-1, 1] to [0, 1] range
            x_rgb = self.color_thres.W_unit_to_N_unit(x_Wspace)
    
        return x_rgb
        
    def _send_stim_and_wait_for_resp(self, trial_counter, trial_identity, xref, x1,
                                     x2 = None, background = None):
        """
        Convert stimuli from W-space to RGB, write them to the shared file, and
        wait for Unity’s confirmation + response.
        
        Args:
            trial_counter (int): Sequential trial index (for logging).
            trial_identity (str): Short tag describing the trial (used in logs).
            xref (array-like): Reference stimulus in W-space.
            x1 (array-like): First comparison stimulus in W-space.
            x2 (array-like, optional): Second comparison (suprathreshold tasks).
            background (array-like, optional): Background in RGB-space.
        
        Returns:
            binaryResp (int): Parsed response code from Unity (e.g., 0/1).
        
        Notes:
            Paradigms supported:
              (1) Oddity: 3 items; 2×reference, 1×comparison.
              (2) Suprathreshold: reference on top; two distinct comparisons below.
              (3) Oddity + varying background: as (1), with background changed.
        
            This method assumes Unity writes a line ending in "Image_Confirmed"
            followed by a response token that `extract_resp` can parse.
        """
        
        # Write down the RGB values to a text file in a shared disk
        trial_type_details = f"Trial_{trial_counter}_{trial_identity}"
        
        # Convert xref and x1 from W space to RGB space
        xref_rgb = self._convert_W_to_RGB(xref).tolist()
        x1_rgb = self._convert_W_to_RGB(x1).tolist()
        
        #optional
        x2_rgb = self._convert_W_to_RGB(x2).tolist() if x2 is not None else None
        background_rgb = background.tolist() if background is not None else None
    
        # Record the time when "Image_Display" is sent
        display_time = time.time()
        
        # Send the RGB values to the recipient
        self.communicator.send_RGBvals(trial_type_details, xref_rgb, x1_rgb, x2_rgb, background_rgb)
    
        # Wait for confirmation
        while True:
            is_confirmed = self.communicator.check_last_word_in_file("Image_Confirmed")
            if is_confirmed: 
                last_line = self.communicator.extract_last_line()
                binaryResp = self.communicator.extract_resp(last_line, str_idx=-2)
                break
    
            # Check if the timeout duration has been exceeded
            if time.time() - display_time > self.communicator.timeout:
                raise TimeoutError("Timeout: Did not receive 'Image_Confirmed' "+\
                                   f"within {self.communicator.timeout} seconds.")
    
            # Pause for a short period to prevent CPU overload
            time.sleep(self.communicator.retry_delay)
    
        return binaryResp
    
    #%%
    def _monitor_time_insert_MOCS_trials(self, start_time, max_wait_time,
                                         trial_sequence, expt_counter, trial_counter, 
                                         stop_event, event_triggered):
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
                    
                if trial_replacement_idx_MOCSlist is None:
                    self.flag_noMore_MOCS = True
                    stop_event.set()
                    break
    
                # Get the stimulus information
                xref = trial_sequence.pregenerated_MOCS['xref'][trial_replacement_idx_MOCSlist]
                x1 = trial_sequence.pregenerated_MOCS['x1'][trial_replacement_idx_MOCSlist]
    
                # Send the stimulus information and wait for the subject to make a response
                binaryResp = self._send_stim_and_wait_for_resp(trial_placement_idx_originallist, 
                                                               f"{trial_placement_id}",
                                                               xref, x1)
                print(f"Responses (MOCS #trial {trial_replacement_idx_MOCSlist}): {binaryResp}")
    
                # Store simulated responses
                trial_sequence.update_data_MOCS(trial_replacement_idx_MOCSlist,
                                                xref, x1, binaryResp)
                # Set the status of the MOCS trial to 'completed'
                trial_sequence.set_trial_status(expt_counter, trial_placement_idx_originallist,
                                                "Completed")
                
                # Set the flag to indicate a MOCS trial was run
                event_triggered.set() 

                # Reset the start time for the next pre-generated trial
                start_time = time.time()
                num_bumped_up_MOCS += 1
                trial_sequence.nBumpUp_MOCS += 1
                self.keep_track_trials_finished += 1
                
                trial_sequence.final_sequence[expt_counter].append(trial_placement_id)
            time.sleep(0.1)  # Check every 10 ms
            
    def _monitor_time_insert_pregenSobol_trials(self, start_time, max_wait_time,
                                         trial_sequence, expt_counter, trial_counter, 
                                         stop_event, event_triggered):     
        num_bumped_up_Sobol = 0
        while not stop_event.is_set():  # Exit if stop_event is set
            elapsed_time = time.time() - start_time
            
            max_wait_time_ii = max_wait_time[0] if num_bumped_up_Sobol == 0 else max_wait_time[-1]
            
            #if the time elapsed exceeds the max wait time
            if elapsed_time > max_wait_time_ii:
                print('Reached the maximum shift for the MOCS trials within a block!')
                #find the next available MOCS trial in the list
                print(f"Deadline exceeded ({elapsed_time:.2f}s). Running a pre-generated Sobol trial...")
                sobol_idx = self.keep_track_trials_finished_pregenSobol
    
                # Get the stimulus information
                xref = trial_sequence.pregenerated_Sobol['xref'][sobol_idx]
                x1 = trial_sequence.pregenerated_Sobol['x1'][sobol_idx]
    
                # Send the stimulus information and wait for the subject to make a response
                binaryResp = self._send_stim_and_wait_for_resp(trial_sequence.nTrials_total + sobol_idx, 
                                                               f"Sobol_{sobol_idx}",
                                                               xref, x1)
                print(f"Responses (Sobol #trial {sobol_idx}): {binaryResp}")
    
                # Store simulated responses
                trial_sequence.pregenerated_Sobol['binaryResp'][sobol_idx] = binaryResp
                
                # Set the status of the MOCS trial to 'completed'
                trial_sequence.set_trial_status(expt_counter, trial_counter,
                                                f"Insert_Sobol_{sobol_idx}")
                
                # Set the flag to indicate a MOCS trial was run
                event_triggered.set() 

                # Reset the start time for the next pre-generated trial
                start_time = time.time()
                num_bumped_up_Sobol += 1
                self.keep_track_trials_finished_pregenSobol += 1
                
            time.sleep(0.1)  # Check every 10 ms
            
    def pause_experiment_for_breaks(self):
        # Break time
        self.communicator.append_message_to_file("Break")
        isResumeSet = False
        while not isResumeSet:  # Use `not` instead of `~`
            # Once participants end the break, the experiment can be resumed
            isResumeSet = self.communicator.check_last_word_in_file("Resume")
            time.sleep(0.5)
            
    def update_background(self, trial_sequence):
        #optional: change the background
        bg = getattr(trial_sequence, "background_rgb", None)
        room = getattr(trial_sequence, "cubicRoom_rgb", None)
        if bg is not None and room is not None:
            self.communicator.change_background(bg, room)
    
    def run_experiment_wMOCSinserted(self, client, trial_sequence, 
                                     max_wait_time=[2.9, 4.1]):
        """
        This method can be used to run or simulates color-discrimination responses 
            using AEPsych. 
        If this is a simulation, ground truth can be based on Wishart fits to pilot 
            data or CIELAB thresholds.

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
        
        #trial_counter is defined a little bit unsual: it's not the actual counter
        #of the current trial, but the counter of the original trial_sequence
        #e.g., 'trial_1_AEPsych_0', trial_counter is '1' in this case
        trial_counter = 0  
        
        #this flag determines whether we want to pause for bumping up the next 
        #available MOCS trials
        pause_for_bumpingMOCS = False

        while trial_counter < trial_sequence.nTrials_total:    
            #self.numConfig is 1 for a 4D experiment. This loop is not really
            #used for this experiment, but in case in the future we want to run
            #interleaved 3D tasks, the code is written to be more general
            for expt_idx in range(self.numConfig): 
                #check if this is a break trial
                if trial_counter in trial_sequence.break_trials:
                    self.pause_experiment_for_breaks()
                
                #optional: update the background
                if trial_counter == 0 and expt_idx == 0:
                    self.update_background(trial_sequence)
                    
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
                    
                    # write down the RGB values to a text file in a shared disk
                    binaryResp = self._send_stim_and_wait_for_resp(trial_counter,
                                                                   trial_identity,
                                                                   xref, x1)
                    
                    # Update the MOCS trial data with the simulated response
                    trial_sequence.update_data_MOCS(trial_idx, xref, x1, binaryResp)
                    # Mark the trial as completed within the time window
                    trial_sequence.set_trial_status(expt_idx, trial_counter, "Completed_in_time")
                    self.keep_track_trials_finished += 1
                    
                elif trial_type == 'AEPsych':
                    # Configure the trial for the AEPsych client
                    #self._configure_session_trial(client, trial_idx)
                    
                    if (self.keep_track_trials_finished - trial_counter) >= self.max_shifts_MOCS:
                        pause_for_bumpingMOCS = True
                    if (self.keep_track_trials_finished - trial_counter) == 0:
                        pause_for_bumpingMOCS = False
                        
                    # Flag to track if a MOCS trial was inserted
                    event_triggered = threading.Event()
                    # Start timing the AEPsych trial
                    start_time = time.time()  
                    # Create an event to stop the monitoring thread
                    stop_event = threading.Event()
                    
                    #if there are still MOCS trials and we do not pause for the fallback trial strategy
                    if (not pause_for_bumpingMOCS) and (not self.flag_noMore_MOCS):
                        # Start monitoring trial generation in a separate thread
                        monitor_thread = threading.Thread(target=self._monitor_time_insert_MOCS_trials,
                                                          args=(start_time, 
                                                                max_wait_time, 
                                                                trial_sequence,
                                                                expt_idx,
                                                                trial_counter,
                                                                stop_event,
                                                                event_triggered))  
                        monitor_thread.start()   
                    else:
                        # Create an event to stop the monitoring thread
                        stop_event = threading.Event()
                        # Start monitoring trial generation in a separate thread
                        monitor_thread = threading.Thread(target=self._monitor_time_insert_pregenSobol_trials,
                                                          args=(start_time, 
                                                                max_wait_time, 
                                                                trial_sequence,
                                                                expt_idx,
                                                                trial_counter,
                                                                stop_event,
                                                                event_triggered))    
                        monitor_thread.start()  
                        
                    # Request a new trial configuration from AEPsych
                    trial_AEPsych = client.ask()
                    
                    # Once AEPsych finishes, stop the monitoring thread
                    stop_event.set()
                    monitor_thread.join()  # Ensure the thread finishes before continuing
                    
                    #record the time AEPsych hands back an answer
                    end_time = time.time()
                    
                    # Extract stimulus dimensions for the trial
                    trial_val = [trial_AEPsych["config"][s][0] for s in self.parnames]
                    # Derive xref and x1 based on par1, par2, par3 and par4 
                    xref, x1, trial_val_report = self._derive_xref_x1(trial_idx, 
                                                                      trial_val,
                                                                      config_index = expt_idx
                                                                      )
                    # Get a response (can either be a simulated resp or from a real participant)
                    binaryResp = self._send_stim_and_wait_for_resp(trial_counter,
                                                                   trial_identity,
                                                                   xref, x1
                                                                   )
                
                    # Report the result back to AEPsych
                    client.tell(config=dict(zip(self.parnames, trial_val_report)),
                                outcome=binaryResp)
                                
                    # Update trial-related data
                    self._update_trial_lists(xref, x1, binaryResp)
                    self.keep_track_trials_finished += 1
                    
                    # Record elapsed time for the trial
                    trial_duration = end_time - start_time
                    time_elapsed.append(trial_duration)
                    
                    # Mark trial as completed with appropriate status
                    if event_triggered.is_set():
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
    
class LoadExptInfo:
    @staticmethod
    def load_pregenerated_MOCS(file_path, session_today, nTrials_MOCS_perSession):
        """
        Load and extract a subset of pre-generated MOCS trials based on the session number.
        
        Parameters:
        - file_path: str, path to the pre-generated MOCS trials file
        - session_today: int, current session number
        - nTrials_MOCS_perSession: int, number of trials to extract per session
        
        Returns:
        - MOCS_subset_trials: dict, containing subsets of 'xref' and 'x1' trials for the session
        - idxTrial_MOCS: list, range of indices for the extracted trials
        """
        with open(file_path, 'rb') as f:
            vars_dict = pickled.load(f)
        
        MOCS_xref_shuffled = vars_dict['MOCS_xref_shuffled']  # Shuffled reference colors
        MOCS_x1_shuffled = vars_dict['MOCS_x1_shuffled']      # Shuffled comparison color
        
        # Determine the start and end indices for the current session's trials
        idxTrial_MOCS = [(session_today-1) * nTrials_MOCS_perSession, 
                         session_today * nTrials_MOCS_perSession]
        
        # Extract the corresponding subset of trials for the current session
        MOCS_subset_trials = {
            'xref': MOCS_xref_shuffled[idxTrial_MOCS[0]:idxTrial_MOCS[1]],  # Subset of reference colors
            'x1': MOCS_x1_shuffled[idxTrial_MOCS[0]:idxTrial_MOCS[1]]       # The corresponding comparison stimuli
        }
        
        return MOCS_subset_trials, idxTrial_MOCS
    
    @staticmethod
    def load_pregenerated_Sobol(file_path, session_today):
        with open(file_path, 'rb') as f:
            vars_dict = pickled.load(f)
        Sobol_xref = vars_dict['Sobol_xref'][int(session_today-1)]
        Sobol_x1 = vars_dict['Sobol_x1'][int(session_today-1)]
            
        Sobol_subset_trials = {
            'xref': Sobol_xref,
            'x1': Sobol_x1,
            'binaryResp': np.full((Sobol_xref.shape[0],), np.nan)}
        return Sobol_subset_trials

    @staticmethod
    def load_pregenerated_val_scaler(file_path, session_today, nTrials_AEPsych_perSession):
        """
        Load the Sobol scalers generated in the first session
        
        Parameters:
        - file_path: str, path where session data is stored.
        - session_today: int, which session
        - nTrials_AEPsych_perSession: int, number of AEPsych trials per session.
        
        Returns:
        - customized_sobol_scaler: list, subset of shuffled Sobol scalers for the current session.
        - idxTrial_AEPsych: list, index range of the selected trials.
        - customized_sobol_scaler_all: list, full list of Sobol scalers from the first session.
        """
        
        # Load the stored Sobol scalers from the first session.
        with open(file_path, 'rb') as f:
            vars_dict2 = pickled.load(f)
        customized_sobol_scaler_all = vars_dict2['customized_sobol_scaler_all']  
        
        # Determine the index range for the current session's subset of Sobol scalers.
        idxTrial_AEPsych = [(session_today-1)*nTrials_AEPsych_perSession, 
                            session_today*nTrials_AEPsych_perSession]
        
        # Extract the subset of Sobol scalers for the current session.
        customized_sobol_scaler = customized_sobol_scaler_all[idxTrial_AEPsych[0]:idxTrial_AEPsych[1]]
        
        return customized_sobol_scaler, idxTrial_AEPsych, customized_sobol_scaler_all

    
    
    