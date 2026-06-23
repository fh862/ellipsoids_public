#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 21:58:01 2024

@author: fangfang

The goal of this script is to simulate AEPsych trials using a WPPM fit as the
ground truth. The ground truth WPPM was obtained by fitting data collected
without adaptive sampling. Instead, trials were placed near discrimination
thresholds with added Gaussian noise. The thresholds were derived from
CIE1994 predictions, corresponding to contours with ΔE = 2.5.

This script supports simulations in both:
    - a 2D/4D isoluminant plane (mapped to a square bounded between −1 and 1), and
    - a 3D/6D color cube.

The simulated trial data are saved to a pickle file and can be fitted later
using separate analysis scripts.

NOTE:
AEPsych 0.7.3 is not compatible with JAX 0.5.3. When running these simulation
scripts, it is recommended to create a separate Python environment that has
only AEPsych 0.7.3 installed (without JAX). A different environment can then
be used for running the JAX-based analysis code.

"""
import importlib.metadata
import matplotlib.pyplot as plt
import dill as pickled
import os
from analysis.utils_load import get_path
import numpy as np
from dataclasses import replace
from analysis.color_thres import color_thresholds
from analysis.config_generator import ConfigGenerator
from analysis.sim_trials import SimulateTrialGivenWishart
from analysis.cross_validation import expt_data
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization,\
    Plot2DSamplingSettings
from aepsych.server import AEPsychServer
from aepsych_client import AEPsychClient
# Get the installed version of AEPsych
aepsych_version = importlib.metadata.version("aepsych")

#define output directory for output files and figures
stim_dims = 2
psyfield_dims = 4
baseDir = get_path("dropbox_root_mac")

# Top-level analysis folder
root = os.path.join(baseDir, "META_analysis")

# Subfolder corresponding to the current task dimensionality
cie = os.path.join(f"{psyfield_dims}dTask", "CIE")

# Output directories for different types of results
output_figDir_sims = os.path.join(root, "Simulation_FigFiles", cie)
output_DBfileDir   = os.path.join(root, "Simulation_DataFiles", cie)
output_fileDir     = os.path.join(root, "ModelFitting_DataFiles", cie)

# Create directories if they do not already exist
directories = [output_figDir_sims, output_DBfileDir, output_fileDir]
# Ensure directories exist (create if they don't)
for directory in directories:
    os.makedirs(directory, exist_ok=True)

#%% 
#------------------------------------------------------------------------------ 
# SECTION 1: Load data files and define the ground-truth model
#------------------------------------------------------------------------------ 
# Additional keyword arguments needed for 2D stimulus configurations
kwargs = {"plane_2D": "Isoluminant plane"} if stim_dims == 2 else {}

# Initialize the color-threshold dataset object
color_thres_data = color_thresholds(stim_dims, baseDir, **kwargs)

# Load the W → display-color transformation used for the 2D task
if stim_dims == 2:
    color_thres_data.load_transformation_matrix(file_date="02242025")
    
# Load CIE-based predictions and the corresponding Wishart model fits
coloralg = 'CIE1994'
color_thres_data.load_CIE_data(CIE_version = coloralg)
color_thres_data.load_model_fits(CIE_version = coloralg)  

# Retrieve the fitted Wishart-model predictions from the loaded dataset
gt_Wishart  = color_thres_data.get_data('model_pred_Wishart', dataset = 'Wishart_data')

#%% 
#------------------------------------------------------------------------------ 
# SECTION 2: Set up configuration files
#------------------------------------------------------------------------------ 
# simulation subject
SUBJ  = 1

# Directory containing customized AEPsych configuration files
path_str_config = get_path("aepsych_config_dir")

# Name of the template configuration file
file_config = f"single_{psyfield_dims}d_colorDiscrimination_EAVC_4strats_new.ini"

# Load the template configuration
config_gen = ConfigGenerator(psyfield_dims,
                             base_path = path_str_config,
                             load_file_name = file_config, 
                             version_new=True,
                             load_default_config=False
                             )

# Retrieve parameter names and strategy names from the config
par_names = config_gen._string_to_list('common','parnames')
strat_names = config_gen._string_to_list('common','strategy_names')

# Retrieve acquisition function and parameter bounds
acqf_str = config_gen.config_parser.get('opt_strat','acqf')

# boundary for each parameter
# 2D: [ref_dim1, ref_dim2, delta_dim1, delta_dim2]
# 3D: [ref_dim1, ref_dim2, ref_dim3, delta_dim1, delta_dim2, delta_dim3]
lb_all = [float(config_gen.config_parser.get(p,'lower_bound')) for p in par_names]
ub_all = [float(config_gen.config_parser.get(p,'upper_bound')) for p in par_names]

print(lb_all)
print(ub_all)

# Number of trials assigned to each strategy
# Example:
#   strategy 1: Sobol sampling in a smaller range
#   strategy 2: Sobol sampling in a larger range
#   strategy 3+: model-based adaptive strategies
NTRIALS_STRAT = [300, 300, 300, 1100]  #[1500, 1500, 1500, 25500]

# Update the minimum number of trials for each strategy
for strat_n, nT in zip(strat_names, NTRIALS_STRAT):
    config_gen.modify_configurations(strat_n, 'min_asks', str(nT))   

# Map strategy names to their corresponding number of trials
strat_dict = dict(zip(strat_names, NTRIALS_STRAT))

# Total number of trials across all strategies
NTRIALS = np.sum(np.array(NTRIALS_STRAT))

# On the Mac, force CPU usage rather than GPU
config_gen.modify_configurations('GPClassificationModel', 'use_gpu', 'False')
config_gen.modify_configurations('OptimizeAcqfGenerator', 'use_gpu', 'False')

# Export the modified configuration as a string
config = config_gen.get_config_as_string()
config_all = [config]

#%% 
#------------------------------------------------------------------------------ 
# SECTION 3: Use AEPsych to for trial placement
#------------------------------------------------------------------------------ 
# define the server!
db_file_name = f"Sim{psyfield_dims}dTask_colorDiscrimination_"+\
                f"{acqf_str}_{NTRIALS}Trials_{NTRIALS_STRAT[0]}_{NTRIALS_STRAT[1]}_"+\
                f"{NTRIALS_STRAT[2]}_{NTRIALS_STRAT[3]}_sub{SUBJ}_gt{coloralg}.db"
server = AEPsychServer(database_path = os.path.join(output_DBfileDir, db_file_name))
client = AEPsychClient(server=server)

# Send a config message to the server, passing in a configuration filename
client.configure(config_str=config)

# Define a Sobol scaler list to be used in simulations.
# if the length is smaller than the number of strategies, 1's will be automatically padded.
# if the length is greater than the number of strategies, the extra part will be truncated.
SOBOL_SCALER = [1/4, 2/4, 3/4, 1]

# Create an instance of a simulation class. This class simulates color discrimination
# responses. The responses are determined by the WPPM fit. Each time AEPsych hands back 
# a pair of reference and comparison stimuli, the routine uses the WPPM fit to make 
# predictions about the probability of correct responses, which is then used to simulate
# a binary response.
AEPsych_trial_given_Wishart_gt = SimulateTrialGivenWishart(psyfield_dims,
                                                           config_all, 
                                                           gt_Wishart,
                                                           val_scaler= SOBOL_SCALER
                                                           )
# start the simulation
AEPsych_trial_given_Wishart_gt.run_simulation(client)

#%% 
#------------------------------------------------------------------------------ 
# SECTION 4: Visualize trial placement
#------------------------------------------------------------------------------ 
# Base plotting settings for all sampling visualizations
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_sims, fontsize = 10)
pltSampSettings = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)

# Visualization object for plotting reference–comparison stimulus pairs
sampling_vis = SamplingRefCompPairVisualization(stim_dims,
                                                color_thres_data,
                                                settings = pltSampSettings,
                                                save_fig = False
                                                )

data_vis_AEPsych = expt_data(server._strats[-1].x[:,0:stim_dims], 
                             server._strats[-1].x[:,0:stim_dims] +\
                             server._strats[-1].x[:,stim_dims:],
                             server._strats[-1].y, None)
    
# This array defines the opacity of markers in the plots, decreasing with more trials.
marker_alpha = [0.5, 0.5, 0.5, 0.2]

# Determine the index ranges of trials belonging to each strategy
# (e.g., [0:300], [300:600], [600:900], [900:6000])
slc_datapoints_to_show_lb = np.concatenate(([0], np.cumsum(NTRIALS_STRAT)[:-1]))
slc_datapoints_to_show_ub = np.cumsum(NTRIALS_STRAT)

# Loop over the selected data points to generate and visualize each corresponding figure.
for i, (lb_i, ub_i) in enumerate(zip(slc_datapoints_to_show_lb, slc_datapoints_to_show_ub)):
    
    # Construct a filename indicating the total number of trials shown
    # and, if applicable, the starting index of the block
    str_trial_idx = f'{ub_i:05}total' if lb_i == 0 else f'{ub_i:05}total_from{lb_i:05}'
    fig_name_i = f"{db_file_name[:-3]}_{str_trial_idx}.pdf"
    
    # Update plotting settings for this block
    pltSampSettings = replace(pltSampSettings,
                              fig_size = (5,5.5),
                              linealpha = marker_alpha[i],  # Line transparency for this subset of data
                              bounds = ub_all[0] *np.array([-1,1]),
                              flag_rescale_axes_label = False,
                              ticks = np.linspace(-0.7,0.7,5),
                              comp_markeralpha = marker_alpha[i], # Marker transparency for this subset of data
                              fig_name = fig_name_i)
    
    # Create a 2D or 3D axis depending on stimulus dimensionality
    if stim_dims == 2:
        fig, ax = plt.subplots(1, 1, figsize = pltSampSettings.fig_size, dpi= pltSampSettings.dpi)
    else:
        fig = plt.figure(figsize = pltSampSettings.fig_size, dpi= pltSampSettings.dpi)
        ax = fig.add_subplot(111, projection='3d')

    # Plot reference–comparison pairs for the selected trial range
    sampling_vis.plot_sampling(data_vis_AEPsych.xref_all[lb_i:ub_i],  # AEPsych_trial_given_Wishart_gt.xref_all[lb_i:ub_i]
                               data_vis_AEPsych.x1_all[lb_i:ub_i],    # AEPsych_trial_given_Wishart_gt.x1_all[lb_i:ub_i]
                               ax = ax, 
                               settings = pltSampSettings
                               )  
    # Save the figure as a PDF
    plt.show()
    #fig.savefig(os.path.join(sims_path, f"{fig_name_i}.pdf"))    
            
#%%
# ---------------------------------------------------
# SECTION 5: save data
# ----------------------------------------------------
output_file = f"{db_file_name[:-3]}.pkl"
full_path_file = os.path.join(output_fileDir, output_file)

variable_names = ['aepsych_version','stim_dims','psyfield_dims','coloralg',
                  'color_thres_data', 'gt_Wishart','SUBJ','file_config', 
                  'par_names', 'strat_dict', 'lb_all', 'ub_all', 'NTRIALS',
                  'config_all', 'db_file_name', 'SOBOL_SCALER', 
                  'AEPsych_trial_given_Wishart_gt']
vars_dict = {}

for var_name in variable_names:
    try:
        # Check if the variable exists in the global scope
        vars_dict[var_name] = eval(var_name)
    except NameError:
        # If the variable does not exist, assign None and print a message
        vars_dict[var_name] = None
        print(f"Variable '{var_name}' does not exist. Assigned as None.")

# Write the list of dictionaries to a file using pickle
with open(full_path_file, 'wb') as f:
    pickled.dump(vars_dict, f)
