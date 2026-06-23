#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:39:44 2024

@author: fangfang

The goal of this script is to simulate AEPsych trials using a WPPM fit as the
ground truth. The ground truth WPPM was obtained by fitting data collected
without adaptive sampling. Instead, trials were placed near discrimination
thresholds with added Gaussian noise. The thresholds were derived from
CIE1994 predictions, corresponding to contours with ΔE = 2.5.

This script supports simulations in both:
    - a 2D/4D isoluminant plane (mapped to a square bounded between −1 and 1), and
    - a 3D/6D color cube.

Note that an important difference between this script and sim_adaptive_4d6d.py 
is that both the reference and delta values are varying in that script, while 
here the reference is fixed at multiple locations and only the delta values are 
varying.

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
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization,\
    Plot2DSamplingSettings
from plotting.wishart_plotting import PlotSettingsBase 
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
cie = os.path.join(f"{stim_dims}dTask", "CIE")

output_figDir_sims = os.path.join(baseDir,'META_analysis','Simulation_FigFiles', cie)
output_DBfileDir = os.path.join(baseDir, 'META_analysis','Simulation_DataFiles', cie)
output_fileDir = os.path.join(baseDir, 'META_analysis','ModelFitting_DataFiles', cie)

# List of directories to ensure existence
directories = [output_figDir_sims,  output_DBfileDir, output_fileDir]
# Ensure directories exist (create if they don't)
for directory in directories:
    os.makedirs(directory, exist_ok=True)

#%% -------------------------------------------------- 
# SECTION 1: load files and define the ground truth
#-----------------------------------------------------
# Additional keyword arguments needed for 2D stimulus configurations
kwargs = {"plane_2D": "Isoluminant plane"} if stim_dims == 2 else {}

# Initialize the color-threshold dataset object
color_thres_data = color_thresholds(stim_dims, baseDir, **kwargs)

# Load the W → display-color transformation used for the 2D task
if stim_dims == 2:
    color_thres_data.load_transformation_matrix(file_date="02242025")
    
# Load Wishart model fits
coloralg = 'CIE1994'
num_grid_pts = 5 #5 or 7
color_thres_data.load_CIE_data(CIE_version = coloralg, num_grid_pts=num_grid_pts)
color_thres_data.load_model_fits(CIE_version = coloralg)  

# Retrieve specific data from Wishart_data
gt_Wishart  = color_thres_data.get_data(f'model_pred_Wishart_grid{num_grid_pts}',
                                        dataset = 'Wishart_data')

grid = color_thres_data.get_data(f'grid{num_grid_pts}',
                                 dataset = 'Wishart_data')

#%% --------------------------------------------------
# SECTION 2: Set up configuration files
# ----------------------------------------------------
# simulation subject
SUBJ = 1

# Directory containing customized AEPsych configuration files
path_str_config = get_path("aepsych_config_dir")

# Name of the template configuration file
file_config = f"single_{stim_dims}d_colorDiscrimination_EAVC_4strats_new.ini"

# Load the template configuration
config_gen = ConfigGenerator(stim_dims,
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
# 2D: [delta_dim1, delta_dim2]
# 3D: [delta_dim1, delta_dim2, delta_dim3]
lb_all = [float(config_gen.config_parser.get(p,'lower_bound')) for p in par_names]
ub_all = [float(config_gen.config_parser.get(p,'upper_bound')) for p in par_names]

print(lb_all)
print(ub_all)

# define strategy and corresponding trials
# strategy 1: sobol generator within a small range
# strategy 2: sobol generator within a medium range
# strategy 3: sobol generator within a large range
# strategy 4: model-based strategy
NTRIALS_STRAT = [100, 100, 100, 50]  

# Update the minimum number of trials for each strategy
for strat_n, nT in zip(strat_names, NTRIALS_STRAT):
    config_gen.modify_configurations(strat_n, 'min_asks', str(nT))   

# Creating a dictionary from two lists using zip
strat_dict = dict(zip(strat_names, NTRIALS_STRAT))

# Total number of trials across all strategies
NTRIALS = np.sum(np.array(NTRIALS_STRAT))

# Calculate the total number of configurations
nRefs = num_grid_pts ** stim_dims

# Reshape each dimension's grid matrix into a column vector and concatenate them
ref = np.reshape(grid, (nRefs, -1))

# On the Mac, force CPU usage rather than GPU
config_gen.modify_configurations('GPClassificationModel', 'use_gpu', 'False')
config_gen.modify_configurations('OptimizeAcqfGenerator', 'use_gpu', 'False')

# Export the modified configuration as a string
config = config_gen.get_config_as_string()
config_all = [config] * nRefs
    
#%% ------------------------------------------------
# SECTION 3: Use AEPsych to for trial placement
# --------------------------------------------------
# define the server!
db_file_name = f"Sim{stim_dims}dTask_interleaved{nRefs}_colorDiscrimination_"+\
                f"{acqf_str}_{NTRIALS}Trials_{NTRIALS_STRAT[0]}_{NTRIALS_STRAT[1]}_"+\
                f"{NTRIALS_STRAT[2]}_{NTRIALS_STRAT[3]}_sub{SUBJ}_gt{coloralg}.db"
                
server = AEPsychServer(database_path = os.path.join(output_DBfileDir,  db_file_name))
client = AEPsychClient(server=server)

# Define a Sobol scaler list to be used in simulations.
# if the length is smaller than the number of strategies, 1's will be automatically padded.
# if the length is greater than the number of strategies, the extra part will be truncated.
SOBOL_SCALER = [1/4, 2/4, 3/4, 1]

# Create an instance of a simulation class. This class simulates color discrimination
# responses. The responses are determined by the Wishart process model. Each time AEPsych
# hands back a pair of reference and comparison stimuli, the routine uses the Wishart model to
# make predictions about the probability of correct responses, which is then used to simulate
# a binary response.
AEPsych_trial_given_Wishart_gt = SimulateTrialGivenWishart(stim_dims,
                                                           config_all, 
                                                           gt_Wishart,     
                                                           ref = ref,  #this is only needed if we interleave multiple expts
                                                           pseudo_randomize= True, 
                                                           pseudo_randomize_seed=SUBJ,
                                                           val_scaler= SOBOL_SCALER
                                                           )
# start the simulation
AEPsych_trial_given_Wishart_gt.run_simulation(client)

#%% ------------------------------------------------
# SECTION 4: Visualize trial placement
# --------------------------------------------------
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_sims, fontsize = 8)
pltSampSettings = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)

sampling_vis = SamplingRefCompPairVisualization(stim_dims,
                                                color_thres_data,
                                                settings = pltSampSettings, 
                                                save_fig = False
                                                )

# This array defines the opacity of markers in the plots, decreasing with more trials.
marker_alpha = [0.2, 0.2, 0.2, 0.5]

# Determine the index ranges of trials belonging to each strategy
slc_datapoints_to_show_lb = np.concatenate(([0], np.cumsum(NTRIALS_STRAT)[:-1])) * nRefs
slc_datapoints_to_show_ub = np.cumsum(NTRIALS_STRAT) * nRefs

# Loop over the selected data points to generate and visualize each corresponding figure.
for i, (lb_i, ub_i) in enumerate(zip(slc_datapoints_to_show_lb, slc_datapoints_to_show_ub)):
    # Construct a filename for each figure based on the plane and number of experiments.
    str_trial_idx = f'{ub_i:05}total' if lb_i == 0 else f'{ub_i:05}total_from{lb_i:05}'
    fig_name_i = f"{db_file_name[:-3]}_{str_trial_idx}.pdf"
            
    pltSampSettings = replace(pltSampSettings,
                            linealpha = marker_alpha[i],        
                            bounds = [np.min(grid), np.max(grid)],
                            comp_markeralpha = marker_alpha[i], 
                            ticks = np.unique(grid),
                            fig_name = fig_name_i
                            )
    
    if stim_dims == 2:
        fig, ax = plt.subplots(1, 1, figsize = pltSampSettings.fig_size, dpi= pltSampSettings.dpi)
    else:
        fig = plt.figure(figsize = pltSampSettings.fig_size, dpi= pltSampSettings.dpi)
        ax = fig.add_subplot(111, projection='3d') 
    # Visualize the trials up to the nth data point with specified marker transparency.
    sampling_vis.plot_sampling(AEPsych_trial_given_Wishart_gt.xref_all[lb_i:ub_i],  # Reference stimuli up to the nth data point
                               AEPsych_trial_given_Wishart_gt.x1_all[lb_i:ub_i],    # Comparison stimuli up to the nth data point
                               ax = ax, 
                               settings = pltSampSettings
                               )                
    # Save the figure as a PDF
    plt.show()
    #fig.savefig(os.path.join(output_figDir_sims, f"{fig_name_i}.pdf"))
    
#%%
# ---------------------------------------------------
# SECTION 5: save data
# ----------------------------------------------------
output_file = f"{db_file_name[:-3]}.pkl"
full_path2 = os.path.join(output_fileDir,output_file)

variable_names = ['aepsych_version','stim_dims','psyfield_dims','coloralg',
                  'num_grid_pts','color_thres_data','gt_Wishart','grid','SUBJ',
                  'file_config','par_names','lb_all', 'ub_all', 'NTRIALS_STRAT',
                  'NTRIALS', 'nRefs','ref','config_all','db_file_name',
                  'SOBOL_SCALER', 'AEPsych_trial_given_Wishart_gt']
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
with open(full_path2, 'wb') as f:
    pickled.dump(vars_dict, f)
    

