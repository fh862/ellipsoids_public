# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 20:55:45 2025

@author: Fangfang
"""

import jax
import time
from datetime import datetime
import dill as pickled
import socket
import numpy as np
import re
import tkinter as tk
import random
import sys
import os
script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from core import oddity_task
try:
    from core import geodesics
except ImportError:
    geodesics = None  # so attribute access doesn't crash

#%% pop up windows to get subject information and comments after the experiment
def get_experiment_info_custom():
    """
    Create a customized popup window to collect subject_id, subject_init, and session_today.
    
    Returns:
        tuple: subject_id (int), subject_init (str), session_today (int)
    """
    # Function to submit the data
    def submit():
        try:
            # Validate and retrieve data
            subject_id = int(subject_id_entry.get())
            subject_init = subject_init_entry.get().strip()
            session_today = int(session_today_entry.get())
            
            # Ensure no fields are empty
            if not subject_init:
                raise ValueError("Subject initials cannot be empty.")
            
            # Update result and close the window
            result["subject_id"] = subject_id
            result["subject_init"] = subject_init
            result["session_today"] = session_today
            popup.destroy()
        except ValueError as e:
            error_label.config(text=f"Error: {e}", fg="red")

    # Create the main popup window
    popup = tk.Tk()
    popup.title("Enter Experiment Info")
    popup.geometry("400x300")  # Set window size

    # Configure font size
    label_font = ("Arial", 14)
    entry_font = ("Arial", 12)
    button_font = ("Arial", 12)
    
    # Create labels and entry fields
    tk.Label(popup, text="Enter Subject ID:", font=label_font).pack(pady=5)
    subject_id_entry = tk.Entry(popup, font=entry_font)
    subject_id_entry.pack(pady=5)
    
    tk.Label(popup, text="Enter Subject Initials:", font=label_font).pack(pady=5)
    subject_init_entry = tk.Entry(popup, font=entry_font)
    subject_init_entry.pack(pady=5)
    
    tk.Label(popup, text="Enter Today's Session Number:", font=label_font).pack(pady=5)
    session_today_entry = tk.Entry(popup, font=entry_font)
    session_today_entry.pack(pady=5)
    
    # Error message label
    error_label = tk.Label(popup, text="", font=("Arial", 10))
    error_label.pack(pady=5)
    
    # Submit button
    tk.Button(popup, text="Submit", command=submit, font=button_font).pack(pady=10)
    
    # Center window
    popup.eval('tk::PlaceWindow . center')
    
    # Initialize result dictionary
    result = {}
    
    # Run the Tkinter event loop
    popup.mainloop()
    
    # Return the collected data
    if result:
        return result["subject_id"], result["subject_init"], result["session_today"]
    else:
        return None, None, None  # Return None values if the window was closed

def get_comment_after_session():
    """
    Create a customized popup window to collect subject_id, subject_init, and session_today.
    
    Returns:
        tuple: subject_id (int), subject_init (str), session_today (int)
    """
    def submit():
        try:
            # Validate and retrieve data
            comment = str(comment_text.get("1.0", tk.END)).strip()
            result["comment"] = comment
            popup.destroy()
        except ValueError as e:
            error_label.config(text=f"Error: {e}", fg="red")

    # Create the main popup window
    popup = tk.Tk()
    popup.title("Comment")
    popup.geometry("600x400")  # Set larger window size

    # Configure font sizes
    label_font = ("Arial", 14)
    entry_font = ("Arial", 12)
    button_font = ("Arial", 12)

    # Create labels and entry fields
    tk.Label(popup, text="How did the experiment go?", font=label_font).pack(pady=5)
    comment_text = tk.Text(popup, font=entry_font, wrap="word", height=10, width=50)  # Use Text widget for multiline input
    comment_text.pack(pady=10)

    # Error message label
    error_label = tk.Label(popup, text="", font=("Arial", 10))
    error_label.pack(pady=5)

    # Submit button
    tk.Button(popup, text="Submit", command=submit, font=button_font).pack(pady=10)

    # Center the window
    popup.eval('tk::PlaceWindow . center')

    # Initialize result dictionary
    result = {}

    # Run the Tkinter event loop
    popup.mainloop()

    # Return the collected data
    if result:
        return result["comment"]
    return None
    
#%% This class is used for creating directory and files for session data
class ExperimentFileManager:
    def __init__(self, subject_id, subject_init, networkDisk_path, is_practice = True):
        """
        Initialize the file manager for a specific subject.
        
        Args:
            subject_id (str): Unique identifier for the subject.
            networkDisk_path (str): Base directory where files are stored.
        """
        self.subject_id = subject_id
        self.subject_init = subject_init
        self.networkDisk_path = networkDisk_path
        self.is_practice = is_practice
        self._check_networkDisk_path()
        
        #create a path just for that subject
        #create subject directory if not exists
        if self.is_practice: 
            self.str_ext = '_practice'
            self.path_sub = os.path.join(self.networkDisk_path, f'sub{subject_id}', 'practice')
        else:
            self.str_ext = ''
            self.path_sub = os.path.join(self.networkDisk_path, f'sub{subject_id}')
        os.makedirs(self.path_sub, exist_ok = True)
        
        self.session_data = {}  # Dictionary to store session metadata
        self.pickle_file = os.path.join(self.path_sub, 
                                        f"sub{subject_id}_{subject_init}_expt_record{self.str_ext}.pkl")
    
    def _check_networkDisk_path(self):
        # Check if the path exists
        if os.path.exists(self.networkDisk_path):
            print(f"The path exists: {self.networkDisk_path}")
        else:
            raise ValueError(f"The path does not exist: {self.networkDisk_path}")
            
    def _validate_subject_init(self):
        """
        Checks whether a text file exists in self.path_sub that starts with 
        either:
            - 'sub{subject_id}_{subject_init}_practice_session1'
            - 'sub{subject_id}_{subject_init}_session1'
        followed by anything and ending with '.txt'.
    
        If found, do nothing. If not, raise an error.
        """
        pattern_practice = re.compile(fr"sub{self.subject_id}_{self.subject_init}_practice_session1.*\.txt$")
        pattern = re.compile(fr"sub{self.subject_id}_{self.subject_init}_session1.*\.txt$")
    
        # Walk through all files in self.path_sub and its subdirectories
        for root, _, files in os.walk(self.path_sub):
            for filename in files:
                print(f"Checking file: {filename}")  # Debugging line
                if pattern_practice.match(filename) or pattern.match(filename):
                    print(f"Match found: {filename}")  # Debugging line
                    return  # File found, do nothing
    
        # If no matching file is found, raise an error
        raise FileNotFoundError(
            f"No matching file found in {self.path_sub} for subject ID '{self.subject_id}' "
            f"and init '{self.subject_init}'."
        )
    
    def _validate_session_num(self, session_num):
        # Retrieve past session numbers
        # (the keys might include strings, e.g., 5, '5_aborted_2025-01-22 16:00:32')
        past_session_keys = list(self.session_data.keys())
        
        # Select only integers and exclude strings
        # if a key is a string, it means it's an aborted session
        past_session_num = [num for num in past_session_keys if isinstance(num, int)]
        
        # Validate session number
        if session_num < 1:
            raise ValueError("Session number must be larger than 1.")
        elif session_num > 1:
            #check whether the input initial matches subject ID 
            self._validate_subject_init()
        
        if not past_session_num:  # No previous session numbers
            if session_num != 1:
                raise ValueError("The first session must be 1.")
        else:  # There are previous sessions
            if session_num in past_session_num:
                # Check the status of the session
                if self.session_data[session_num]['status'] == 'Done':
                    raise ValueError("This session was already completed in the past.")
                else:
                    pressed_button = input(
                        "There is an existing file for this session,\n"
                        "but the status shows that the session was not completed.\n"
                        "Please confirm that this is true and press Y/N to proceed/stop: "
                    )
                    if pressed_button.lower() != "y":
                        print("Operation cancelled.")
                        return None, None
                    else:
                        #changed the key in self.session_data
                        # Get the current timestamp
                        timestamp = datetime.now()
                        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        key_aborted_session = f"{session_num}_aborted_{timestamp_str}"
                        
                        print(key_aborted_session)
                        self.session_data[key_aborted_session] = \
                            self.session_data[session_num]
            elif session_num != (max(past_session_num) + 1):
                raise ValueError(
                    f"Previous session numbers are: {past_session_num}. "
                    f"The next one should be {max(past_session_num) + 1}."
                )
                    
    def create_session_file(self, session_num):           
        # Validate session number
        self._validate_session_num(session_num)
            
        #Generate the file name and path:
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f"sub{self.subject_id}_{self.subject_init}{self.str_ext}_"+\
                    f"session{session_num}_{date_time}.txt"
        file_path = os.path.join(self.path_sub, file_name)
        
        # Create the file with metadata
        with open(file_path, 'w') as file:
            file.write(f"Subject ID: {self.subject_id}\n")
            file.write(f"Subject initial: {self.subject_init}\n")
            file.write(f"Session: {session_num}\n")
            file.write(f"Date and Time: {date_time}\n")
            
        # Update session data
        self.session_data[session_num] = {
            "sub_id": self.subject_id,
            "sub_initial": self.subject_init,
            "file_name": file_name,
            "date_time": date_time,
            "session_number": session_num,
            "sender_path_sub": self.path_sub,
            "is_practice": self.is_practice,
            "status": 'Created'
        }
        
        # Save the updated state
        self.save_state()
        
        print(f"File created and state saved: {file_path}")
        return file_path, file_name
    
    def status_updates(self, status, session_num = None):
        """
        Update the status of the latest session file from the sender's perspective.
        
        Args:
            recipient_status (str): The status to update. Possible values are 
                                    'Confirmed', 'Communicating', or 'Done'.
            session_num (int): default is the latest session
        
        Raises:
            ValueError: If the recipient_status is invalid or if there are no sessions.
        """
        valid_statuses = {'Confirmed', 'Communicating', 'Done'}
        
        # Validate the status
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Valid options are {valid_statuses}.")
        
        # Check if there is at least one session
        if not self.session_data:
            raise ValueError("No session data available to update.")
        
        # Determine which session to update
        if session_num is None:
            # (the keys might include strings, e.g., 5, '5_aborted_2025-01-22 16:00:32')
            past_session_keys = list(self.session_data.keys())
            
            # Select only integers and exclude strings
            # if a key is a string, it means it's an aborted session
            past_session_num = [num for num in past_session_keys if isinstance(num, int)]
            session_num = max(past_session_num)  # Get the latest session number

        #update the status
        self.session_data[session_num]["status"] = status
        
        # Save the updated state
        self.save_state()
        
        print(f"Updated session {session_num} status to: {status}")

    def add_comments(self, comment, session_num = None):
        """
        Add a comment to a specific session in the session data.
        
        Parameters:
        - comment (str): The comment text to add to the session.
        - session_num (int or None, optional): The session number to update. 
          If None, the method updates the latest session based on the highest session number.
        
        Raises:
        - ValueError: If there is no session data available to update.
        """
        
        # Check if there is at least one session
        if not self.session_data:
            raise ValueError("No session data available to add any comment.")
            
        # Determine which session to update
        if session_num is None:
            # (the keys might include strings, e.g., 5, '5_aborted_2025-01-22 16:00:32')
            past_session_keys = list(self.session_data.keys())
            
            # Select only integers and exclude strings
            # if a key is a string, it means it's an aborted session
            past_session_num = [num for num in past_session_keys if isinstance(num, int)]
            
            session_num = max(past_session_num)  # Get the latest session number
            
        #update the status
        self.session_data[session_num]["comment"] = comment
        
        # Save the updated state
        self.save_state()
    
    def save_state(self):
        """
        Save the current state of the class as a pickle file.
        """
        with open(self.pickle_file, 'wb') as pkl_file:
            pickled.dump(self, pkl_file)
        print(f"State saved to pickle: {self.pickle_file}")
        
    def list_files(self):
        """
        List all files created for the subject and print them.
        
        Returns:
            list: List of file names.
        """
        file_names = [data["file_name"] for data in self.session_data.values()]
        
        # Print the file names
        print("Files created for the subject:")
        for file_name in file_names:
            print(f"- {file_name}")
        
        return file_names
    
    @staticmethod
    def load_state(pickle_file):
        """
        Load a saved instance of ExperimentFileManager from a pickle file.
        
        Args:
            pickle_file (str): Path to the pickle file.
    
        Returns:
            ExperimentFileManager: Loaded instance.
        """
        with open(pickle_file, 'rb') as pkl_file:
            instance = pickled.load(pkl_file)
        print(f"State loaded from pickle: {pickle_file}")
        return instance
        
    def export_expt_log_txt(self, file_name, sim_interleaved_trial_sequence,
                            column_widths = [24, 30, 30, 30, 20, 20], 
                            shuffled_order = None):
        """
        Export a human-readable experiment log as a plain-text table.
    
        Each row corresponds to the status of one trial for one interleaved
        experiment (condition), optionally reordered by `shuffled_order`.
    
        Parameters
        ----------
        file_name : str
            Name of the output text file (no path).
        sim_interleaved_trial_sequence : object
            Object that stores trial metadata, including:
                - nTrials_total : int
                - trial_status  : list of lists; trial_status[cond][trial]
                                  is a list of status strings.
        column_widths : list of int, optional
            Fixed width (in characters) for each column when printing.
            Extra columns beyond this list are printed without padding.
        shuffled_order : np.ndarray, optional
            If provided, must be an integer array of shape
            (num_interleaved_expts, nTrials_total), where
            shuffled_order[m, n] gives the condition index (mm) whose
            n-th trial should be printed in slot (m, n). If None, the
            condition index is taken to be m directly.
        """
        nTrials_total = sim_interleaved_trial_sequence.nTrials_total

        trial_status = sim_interleaved_trial_sequence.trial_status
        n_conds = len(trial_status)

        if shuffled_order is not None:
            assert shuffled_order.shape == (n_conds, nTrials_total), (
                "shuffled_order has incorrect shape: "
                f"expected ({n_conds}, {nTrials_total}), got {shuffled_order.shape}."
            )
    
        # Open the log file for writing
        out_path = os.path.join(self.path_sub, file_name)
        with open(out_path, "w", encoding="utf-8") as file:
            # Loop over trials in chronological order within each block
            for n in range(nTrials_total):
                # Loop over each interleaved experiment (condition slot)
                for m in range(n_conds):
    
                    # Decide which condition index to read from
                    if shuffled_order is not None:
                        mm = shuffled_order[m, n]
                    else:
                        mm = m
    
                    # Retrieve the status list for this condition and trial
                    list_status = sim_interleaved_trial_sequence.trial_status[mm][n]
    
                    # Left-justify each string to its column width
                    aligned_row = " | ".join(
                        (item.ljust(column_widths[i]) if i < len(column_widths) else item)
                        for i, item in enumerate(list_status)
                    )
    
                    # Echo to console (optional debugging)
                    print(aligned_row)
    
                    # Write the formatted row to the file
                    file.write(aligned_row + "\n")   

#%% This class is used for communications between two computers via a network disk
class CommunicateViaTextFile:
    def __init__(self, network_disk_path, retry_delay = 1/60, timeout = 1200):
        """
        Initialize the CommunicateViaTextFile class with the file path.
        
        Args:
            network_disk_path (str): 
                Path to the shared network disk where messages will be appended and read.
            retry_delay (float, optional): 
                Time (in seconds) to wait before retrying file operations.
                Default is `1/60` seconds (~16.67ms), matching the duration of 1 frame 
                on a 60 Hz monitor, ensuring efficient polling without excessive CPU load.
            timeout (float, optional): 
                Maximum time (in seconds) to wait before raising a timeout error.
                Default is `1200` seconds (20 minutes), allowing long-duration operations 
                before termination.

        """
        self.network_disk_path = network_disk_path
        self.computer_name = socket.gethostname()
        self.retry_delay = retry_delay #default is the duration of 1 frame (assuming the monitor is 60 Hz)
        self.timeout = timeout #default is 1200s (20 mins)
        self.terminate = False
        
    def check_and_handle_file(self, file_name):
        """
        Checks if the directory exists, and handles file creation or renaming.

        Args:
            file_name (str): The name of the file to check or create.

        Returns:
            str: The full path of the newly created or handled file.
        """
        # Ensure the directory exists
        if not os.path.exists(self.network_disk_path):
            os.makedirs(self.network_disk_path)

        # Full path to the file
        file_path = os.path.join(self.network_disk_path, file_name)

        # Check if the file exists
        if os.path.exists(file_path):
            print('Found the file.')
        else:
            # Create a new file with the original name
            with open(file_path, 'w'):
                pass  # File is created and immediately closed

        self.network_disk_fullfile = file_path
        
    def append_message_to_file(self, message):
        """
        Appends a message with a timestamp to the specified file.

        Args:
            message (str): The message to append to the file.

        Raises:
            TimeoutError: If the file cannot be opened for writing within the timeout period.
            IOError: If there is an issue writing to the file.
        """
        start_time = time.time()

        while True:
            try:
                # Open the file in append mode
                with open(self.network_disk_fullfile, 'a') as file:
                    # Get the current timestamp
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Append the message with a timestamp to the file
                    file.write(f"{timestamp} - {self.computer_name}: {message}\n")
                    return  # Exit the function after successful write
            except IOError:
                # Check if timeout has been exceeded
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Timeout: Unable to write to file within {self.timeout} seconds.")
                
                # Pause before retrying
                time.sleep(self.retry_delay)
        
    def extract_last_line(self):
        """
        Extracts the last line from the file.

        Returns:
            str: The last line of the file.

        Raises:
            TimeoutError: If the file cannot be read within the timeout period.
            IOError: If there is an issue reading the file.
        """
        start_time = time.time()

        while True:
            try:
                # Open the file for reading
                with open(self.network_disk_fullfile, 'r') as file:
                    last_line = ''
                    # Read the file line by line to get the last line
                    for line in file:
                        last_line = line.strip()  # Keep trimming whitespace
                return last_line
            except IOError:
                # Check if timeout has been exceeded
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Timeout: Unable to read file within {self.timeout} seconds.")
                
                # Pause before retrying
                time.sleep(self.retry_delay)
    
    def extract_last_word_in_file(self, last_line = None):
        """
        extract the last word.
    
        Args:
            word (str): The word to check.
    
        Returns:
            tuple: A tuple containing:
                - bool: True if the word is the last word of the last line, otherwise False.
                - str: The last line of the file.
        
        Raises:
            IOError: If the file cannot be opened for reading.
        """
        try:
            if last_line is None:
                # Read the file line by line to get the last line
                last_line = self.extract_last_line()
    
            # Split the last line into words
            if last_line:
                words = last_line.split()
                return words[-1]
    
            # If the file is empty, return False and an empty string
            return False, ""
        except IOError:
            raise IOError("Failed to open file for reading.")
        
    def check_last_word_in_file(self, word, last_word = None):
        """
        Checks if the specified word is the last word of the last line in the file.
    
        Args:
            word (str): The word to check.
    
        Returns:
            tuple: A tuple containing:
                - bool: True if the word is the last word of the last line, otherwise False.
                - str: The last line of the file.
        
        Raises:
            IOError: If the file cannot be opened for reading.
        """
        if last_word is None:
            last_word = self.extract_last_word_in_file()
        return last_word == word
            
    def extract_rgb_values(self, input_string):
        """
        Extracts the RGB values from the input string.

        Args:
            input_string (str): The input string containing RGB values.

        Returns:
            np.ndarray: A NumPy array of the RGB values.
        """
        try:            
            # Find the RGB part in the message
            rgb_part = input_string.split(' ')[-2]  # Extract the part before "Image_Display"
            
            # Extract individual R, G, B values
            r_value = float(rgb_part.split('_')[0][1:])  # Remove 'R' and convert to float
            g_value = float(rgb_part.split('_')[1][1:])  # Remove 'G' and convert to float
            b_value = float(rgb_part.split('_')[2][1:])  # Remove 'B' and convert to float
            
            # Return as a NumPy array
            return np.array([r_value, g_value, b_value])
        except Exception as e:
            raise ValueError(f"Failed to extract RGB values from input: {input_string}. Error: {e}")
            
    def initialize_communication(self):
        """
        Writes an initial message to the file and waits for a response from Unity 
        indicating that initialization is complete.
        
        Raises:
            IOError: If the file cannot be opened for writing or reading.
        """
        # Append the message to the file
        self.append_message_to_file("Set_Up_to_Communicate")

        start_time = time.time()
        # Wait for Unity to send back "Ready_To_Communicate"
        while True:
            is_ready_to_communicate = self.check_last_word_in_file("Ready_To_Communicate")
            is_trial_requested = self.check_last_word_in_file("Trial_requested")
            if is_ready_to_communicate or is_trial_requested:
                break
            
            # Check if the timeout duration has been exceeded
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Timeout: Did not receive 'Ready_To_Communicate' within {self.timeout} seconds.")
            # Pause for a short period to prevent CPU overload
            time.sleep(self.retry_delay)
            
    def change_background(self, background_RGB, cubicRoom_RGB):
        # Create the message to indicate the current stimulus to display
        message_setup = f"Background_R{background_RGB[0]:.8f}_G{background_RGB[1]:.8f}_B{background_RGB[2]:.8f} "+\
                        f"CubicRoom_R{cubicRoom_RGB[0]:.8f}_G{cubicRoom_RGB[1]:.8f}_B{cubicRoom_RGB[2]:.8f} " +\
                        "Change_Background"

        # Append the message to the file
        self.append_message_to_file(message_setup)

        start_time = time.time()

        # Wait for Unity to send back a message indicating the image has been displayed
        while True:
            is_image_confirmed = self.check_last_word_in_file("Change_Background_Confirmed")
            if is_image_confirmed:
                break

            # Check if the timeout duration has been exceeded
            if time.time() - start_time > self.timeout:
                raise TimeoutError("Timeout: the recipient did not confirm the RGB values in time.")

            # Pause for a short period to prevent CPU overload
            time.sleep(self.retry_delay)
            
    def send_RGBvals(self, trial_info, ref_RGB, comp_RGB, comp2_RGB = None, background_RGB = None):
        """
        Sends the current RGB values for display to a shared file and waits for confirmation 
        from the recipient (e.g., Unity) that the image has been displayed.

        Args:
            trial_info (str): Metadata or identifier for the current trial.
            ref_RGB (list or tuple): RGB values [R, G, B] for the reference stimulus.
            comp_RGB (list or tuple): RGB values [R, G, B] for the first comparison stimulus.
            comp2_RGB (list or tuple, optional): RGB values [R, G, B] for a second comparison stimulus.
                                                 If None, only one comparison stimulus is used.
            background_RGB (list|tuple, optional): [R, G, B] for the room/background.
                                                   If None, the background is unchanged.

        Raises:
            TimeoutError: If no confirmation is received from the recipient within the timeout period.
        """

        # Supported paradigms:
        # (1) Oddity: three items in a triangle; two are the reference, one is the comparison.
        # (2) Suprathreshold: reference at top; two distinct comparisons at bottom.
        # (3) Oddity + varying background: as (1), but with a modified background.

        # If a second comparison RGB value is provided, we assume the suprathreshold task.
        # Otherwise, we default to the oddity task.
        if comp2_RGB is not None: 
            str_comp2 = f"Comp2_R{comp2_RGB[0]:.8f}_G{comp2_RGB[1]:.8f}_B{comp2_RGB[2]:.8f} "
        else:
            str_comp2 = ""
            
        if background_RGB is not None:
            str_background = f"Background_R{background_RGB[0]:.8f}_G{background_RGB[1]:.8f}_B{background_RGB[2]:.8f} "
        else:
            str_background = ""
        
        # Create the message to indicate the current stimulus to display
        message_image_for_display = f"{trial_info} "+\
                                    f"Ref_R{ref_RGB[0]:.8f}_G{ref_RGB[1]:.8f}_B{ref_RGB[2]:.8f} "+\
                                    f"Comp_R{comp_RGB[0]:.8f}_G{comp_RGB[1]:.8f}_B{comp_RGB[2]:.8f} "+\
                                    f"{str_comp2}{str_background}Image_Display"

        # Append the message to the file
        self.append_message_to_file(message_image_for_display)

        start_time = time.time()

        # Wait for Unity to send back a message indicating the image has been displayed
        while True:
            is_image_confirmed = self.check_last_word_in_file("Image_Confirmed")
            if is_image_confirmed:
                break

            # Check if the timeout duration has been exceeded
            if time.time() - start_time > self.timeout:
                raise TimeoutError("Timeout: the recipient did not confirm the RGB values in time.")

            # Pause for a short period to prevent CPU overload
            time.sleep(self.retry_delay)
                
    def finalize(self):
        """
        Appends a message to the file indicating that the sequence is done.
        """
        self.append_message_to_file("Done")
        self.terminate = True
            
    @staticmethod
    def extract_resp(input_string, str_idx=-2):
        """
        Extracts the binary response from the input string following the 'Resp' keyword.
    
        Args:
            input_string (str): The input string containing the response data.
            str_idx (int): The index of the response substring in the split input string.
                           Defaults to -2 (second to last substring).
    
        Returns:
            int: The binary response (0 or 1).
    
        Raises:
            ValueError: If the response substring is not found, is not binary, or an error 
            occurs during extraction.
        """
        try:
            # Split the string and access the specified index
            resp_str = input_string.split(' ')[str_idx]
    
            # Check if the substring starts with 'Resp'
            if not resp_str.startswith('Resp'):
                raise ValueError(
                    "The substring at the specified index does not contain the 'Resp' keyword. Please verify the input string and index."
                )
    
            # Extract the last character of the 'Resp' substring and convert it to an integer
            resp = int(resp_str[-1])
    
            # Validate that the response is binary (0 or 1)
            if resp not in [0, 1, 2]:
                raise ValueError(
                    "The extracted response is not binary. Expected values are 0, 1 or 2."
                )
    
            return resp
        except Exception as e:
            raise ValueError(
                f"Failed to extract the response from input: {input_string}. Error: {e}"
            )     
            
#%% The following methods are used by the recipient system, which is written in C#. 
# These Python methods are solely for testing purposes to verify proper functionality 
# of the network disk connection between the host computer (BELUGA) and the recipient 
# computer (Stonefish).
    def confirm_communication(self):
        start_time = time.time()
        # Wait for command
        while True:
            is_set_up_to_communicate = self.check_last_word_in_file("Set_Up_to_Communicate")
            if is_set_up_to_communicate:
                # Append the message to the file
                self.append_message_to_file("Ready_To_Communicate")
                break
                
            # Check if the timeout duration has been exceeded
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Timeout: Did not receive 'Set_Up_to_Communicate' within {self.timeout} seconds.")
            # Pause for a short period to prevent CPU overload
            time.sleep(self.retry_delay)    
        
    @staticmethod
    def predict_probability_correct_Wishart(gt_Wishart, xref, x1, resp_time=0.2):
        """
        Simulates a trial where a participant identifies the odd stimulus based on
        a given Wishart model. This function is primarily used to test communication
        between two computers; the actual response will be generated and stored in Unity.
    
        The function computes the probability of correctly identifying the odd stimulus 
        by evaluating the model's basis functions at the reference and comparison stimuli.
    
        Args:
            gt_Wishart (object): An instance of the Wishart model containing parameters 
                                 and methods for computing the weighted sums.
            xref (array-like): Normalized feature vector of the reference stimulus 
                               (values in the range [-1, 1]).
            x1 (array-like): Normalized feature vector of the comparison stimulus 
                             (values in the range [-1, 1]).
            resp_time (float, optional): Simulated response time in seconds (default: 0.2s).
    
        Returns:
            int: A binary response (0 or 1), where 1 indicates a correct response.
        """
    
        # Compute the model's weighted sum of basis functions at the reference stimulus
        Uref = gt_Wishart.model.compute_U(gt_Wishart.W_est, xref)
        
        # Compute the model's weighted sum of basis functions at the comparison stimulus
        U1 = gt_Wishart.model.compute_U(gt_Wishart.W_est, x1)
        
        # Simulate the decision-making process for identifying the odd stimulus
        signed_diff = oddity_task.simulate_oddity_one_trial(
            (xref, x1, Uref, U1),
            gt_Wishart.opt_key,
            gt_Wishart.opt_params['mc_samples'],
            gt_Wishart.model.diag_term
        )
        
        # Compute the probability of correctly identifying the odd stimulus
        pX1 = oddity_task.approx_cdf_one_trial(
            0.0, signed_diff, gt_Wishart.opt_params['bandwidth']
        )
        
        # Generate a random response based on the predicted probability
        binaryResp = int(np.random.rand() < pX1)
        
        # Simulate response time delay
        time.sleep(resp_time)
        
        return binaryResp
        
    @staticmethod
    def predict_probability_correct_random():
        # Randomly generate a binary response (0 or 1)
        return random.randint(0, 1)
    
    @staticmethod
    def predict_comp_random():
        # Randomly generate a binary response (comp1 or comp2 is more similar compared to the reference)
        return random.randint(1, 2)

    @staticmethod
    def predict_comp_Wishart(gt_Wishart, xref, x1, x2):
        """
        Make predictions of the suprathreshold judgments

        Parameters
        ----------
        gt_Wishart (object): An instance of the Wishart model containing parameters 
                             and methods for computing the weighted sums.
        xref (array-like): Normalized feature vector of the reference stimulus 
                           (values in the range [-1, 1]).
        x1 (array-like): Normalized feature vector of the comparison stimulus #1
                         (values in the range [-1, 1]).
        x2 (array-like): Normalized feature vector of the comparison stimulus #2
                         (values in the range [-1, 1]).

        Returns
        -------
            int: A binary response (1 or 2), where 1 indicates comp#1.

        """
        
        # Compute the model's weighted sum of basis functions at the reference stimulus
        Uref = gt_Wishart.model.compute_U(gt_Wishart.W_est, xref)
        
        # Compute the model's weighted sum of basis functions at the comparison stimuli
        U1 = gt_Wishart.model.compute_U(gt_Wishart.W_est, x1)
        U2 = gt_Wishart.model.compute_U(gt_Wishart.W_est, x2)
        
        # Simulate the decision-making process for identifying the odd stimulus.
        signed_diff = oddity_task.simulate_oddity_suprathres_one_trial(
            (xref, x1, x2, Uref, U1, U2),
            gt_Wishart.opt_key,
            gt_Wishart.opt_params['mc_samples'],
            gt_Wishart.model.diag_term
        )
        
        # Compute the probability of correctly identifying the odd stimulus using the signed difference.
        pX2 = oddity_task.approx_cdf_one_trial(0.0, signed_diff,
                                               gt_Wishart.opt_params['bandwidth'])
        
        # Generate a random response based on the predicted probability
        # response = 2: comp2 is judged to be more different relative to ref
        # response = 1: comp1 is judged to be more different relative to ref
        binaryResp = int(np.random.rand() < pX2) + 1
        
        return binaryResp
    
    @staticmethod
    def predict_comp_geodesics_M1(gt_Wishart, xref, x1, x2, popsize = 64, numgen = 30):
        # Compute the model's weighted sum of basis functions at the reference stimulus
        Uref = gt_Wishart.model.compute_U(gt_Wishart.W_est, xref)
        
        # Compute the model's weighted sum of basis functions at the comparison stimuli
        U1 = gt_Wishart.model.compute_U(gt_Wishart.W_est, x1)
        U2 = gt_Wishart.model.compute_U(gt_Wishart.W_est, x2)

        # Number of stimulus dimensions
        ndims_cov, ndims_extra = Uref.shape

        # Generate random draws from isotropic, standard gaussians
        keys = jax.random.split(gt_Wishart.opt_key, num=5)
        nn0 = jax.random.normal(keys[0], shape=(1, ndims_extra))
        nn1 = jax.random.normal(keys[1], shape=(1, ndims_extra))
        nn2 = jax.random.normal(keys[2], shape=(1, ndims_extra))

        # Re-scale and translate the noisy samples to have the correct mean and
        # covariance.
        #     z0 ~ Normal(mref, Uref @ Uref.T).
        #     z1 ~ Normal(mref, Uref @ Uref.T).
        #     z2 ~ Normal(mprobe, Uprobe @ Uprobe.T).
        z0 = nn0 @ Uref.T + xref[None, :] 
        z1 = nn1 @ U1.T + x1[None, :] 
        z2 = nn2 @ U2.T + x2[None, :] 
        
        #% load the weight
        geodesics.load_modelW(gt_Wishart.model, gt_Wishart.W_est)    

        # compute the geodesic path from z0 and z1
        keys_z01 = jax.random.split(keys[3], z0.shape[0])
        best_v0s_z0z1 = geodesics.batch_estimate_v0(z0, z1, keys_z01,
                                                    popsize= popsize,
                                                    num_generations= numgen
                                                    )
        paths_z0z1, dists_z0z1 = geodesics.batch_paths_and_dists(z0, best_v0s_z0z1)
        
        # compute the geodesic path from z0 and z2
        keys_z02 = jax.random.split(keys[4], z0.shape[0])
        best_v0s_z0z2 = geodesics.batch_estimate_v0(z0, z2, keys_z02, 
                                                    popsize= popsize, 
                                                    num_generations= numgen
                                                    )
        paths_z0z2, dists_z0z2 = geodesics.batch_paths_and_dists(z0, best_v0s_z0z2)
        
        if dists_z0z2 - dists_z0z1 > 0:
            return 2
        else:
            return 1
        
    def confirm_RGBvals(self, gt_Wishart=None, color_thres=None, response_delay=3):
        """
        Performs a handshake check between the host (BELUGA) and recipient (Stonefish)
        via the network disk to confirm that RGB values were successfully sent and
        acknowledged. This function simulates a response and appends a confirmation
        message to the shared file.

        This method is primarily used for communication testing purposes. In real 
        experiments, the response will come from the human subject and be handled 
        by C# in Unity.

        Args:
            gt_Wishart (optional): A WishartModel instance used to simulate a probability-based response.
            color_thres (optional): An object containing the RGB-to-model-space transformation matrix.
            response_delay (int): How long to wait (in seconds) before sending back the simulated response.
        """
        start_time = time.time()

        # Continuously monitor the shared file for incoming trial commands
        while True:
            last_line = self.extract_last_line()
            last_word = self.extract_last_word_in_file(last_line=last_line)

            is_image_display = self.check_last_word_in_file("Image_Display", last_word=last_word)
            is_done = self.check_last_word_in_file("Done", last_word=last_word)
            is_break = self.check_last_word_in_file("Break", last_word=last_word)

            # If experiment signals it's done, terminate the session
            if is_done:
                self.terminate = True
                break

            # If a "Break" command is received, briefly pause then acknowledge with "Resume"
            if is_break:
                time.sleep(1)
                self.append_message_to_file("Resume")
                break

            # If an image display command is received, extract and process RGB values
            if is_image_display:
                packed_trial_info = self.extract_ref_comp_rgb_values(last_line)

                # Oddity task (3 stimuli total, but only one unique comparison RGB)
                if len(packed_trial_info) == 3:
                    trial_type, ref_rgb, comp_rgb = packed_trial_info

                    if (gt_Wishart is not None) and (color_thres is not None):
                        # Project RGBs into 2D model space and simulate response using ground truth Wishart model
                        xref = color_thres.M_RGBTo2DW @ ref_rgb
                        x1 = color_thres.M_RGBTo2DW @ comp_rgb
                        response = self.predict_probability_correct_Wishart(\
                                        gt_Wishart, xref[:2], x1[:2])
                    else:
                        response = self.predict_probability_correct_random()

                    str_comp2 = ""

                # Suprathreshold task (with 2 unique comparison RGBs)
                elif len(packed_trial_info) == 4:
                    trial_type, ref_rgb, comp_rgb, comp2_rgb = packed_trial_info
                    if gt_Wishart.model.num_dims == 2:
                        xref = color_thres.M_RGBTo2DW @ ref_rgb
                        x1 = color_thres.M_RGBTo2DW @ comp_rgb
                        x2 = color_thres.M_RGBTo2DW @ comp2_rgb
                        response = self.predict_comp_geodesics_M1(\
                                        gt_Wishart, xref[:2], x1[:2], x2[:2])
                    else:
                        xref = color_thres.N_unit_to_W_unit(ref_rgb)
                        x1 = color_thres.N_unit_to_W_unit(comp_rgb)
                        x2 = color_thres.N_unit_to_W_unit(comp2_rgb)
                        response = self.predict_comp_geodesics_M1(\
                                        gt_Wishart, xref, x1, x2)
                    str_comp2 = f"Comp2_R{comp2_rgb[0]:.8f}_G{comp2_rgb[1]:.8f}_B{comp2_rgb[2]:.8f} "

                # Wait before sending response (simulate processing time)
                time.sleep(response_delay)

                # Construct confirmation message including response
                message_image_for_display = f"{trial_type} " + \
                    f"Ref_R{ref_rgb[0]:.8f}_G{ref_rgb[1]:.8f}_B{ref_rgb[2]:.8f} " + \
                    f"Comp_R{comp_rgb[0]:.8f}_G{comp_rgb[1]:.8f}_B{comp_rgb[2]:.8f} " + \
                    f"{str_comp2}Resp{response} Image_Confirmed"

                # Send response to shared file
                self.append_message_to_file(message_image_for_display)
                break

            # If no valid command received within timeout, raise an error
            if time.time() - start_time > self.timeout:
                raise TimeoutError("Timeout: the sender did not send out the RGB values in time.")

            # Sleep briefly to avoid busy-waiting
            time.sleep(self.retry_delay)
        
    @staticmethod
    def parse_rgb_match(match):
        """
        Convert a regex match object containing R, G, B groups into a NumPy array.

        Args:
            match (re.Match): A regex match object with 3 capture groups (R, G, B).

        Returns:
            np.ndarray: A NumPy array of shape (3,) containing the RGB values as floats.
        """
        return np.array([float(match.group(i)) for i in range(1, 4)])
        
    @staticmethod
    def extract_ref_comp_rgb_values(input_string):
        """
        Extracts the trial type and RGB values for reference and comparison stimuli
        from a formatted input string.

        Args:
            input_string (str): A string containing encoded RGB values for the stimuli.

        Returns:
            tuple: 
                - trial_type (str): The 5th word in the input string, typically describing the trial type.
                - ref_rgb (np.ndarray): RGB values for the reference stimulus.
                - comp_rgb (np.ndarray): RGB values for the first comparison stimulus.
                - comp2_rgb (np.ndarray, optional): RGB values for the second comparison stimulus, 
                  if present in the input string.
        
        Raises:
            ValueError: If required RGB fields ('Ref' or 'Comp') are missing or the format is invalid.
        """
        try:
            # Extract the trial type (assumed to be the 5th word in the string)
            trial_type = input_string.split(' ')[4]

            # Regex pattern for RGB triplets (e.g., R0.5_G0.5_B0.5)
            re_pattern = r"R([0-9\.\-]+)_G([0-9\.\-]+)_B([0-9\.\-]+)"

            # Search for matches of each RGB tag in the string
            ref_match = re.search(fr"Ref_{re_pattern}", input_string)
            comp_match = re.search(fr"Comp_{re_pattern}", input_string)
            comp2_match = re.search(fr"Comp2_{re_pattern}", input_string)

            # Ensure required fields are present
            if not ref_match or not comp_match:
                raise ValueError("Failed to find required 'Ref' or 'Comp' RGB values in the input string.")

            # Parse RGB values into NumPy arrays
            ref_rgb = CommunicateViaTextFile.parse_rgb_match(ref_match)
            comp_rgb = CommunicateViaTextFile.parse_rgb_match(comp_match)

            # Optionally extract comp2 if present
            if comp2_match:
                comp2_rgb = CommunicateViaTextFile.parse_rgb_match(comp2_match)
                return trial_type, ref_rgb, comp_rgb, comp2_rgb
            else:
                return trial_type, ref_rgb, comp_rgb

        except Exception as e:
            raise ValueError(f"Failed to extract data from input: {input_string}. Error: {e}")
            
        
        
        
        
        
        
        
        
        
        
            
            
            