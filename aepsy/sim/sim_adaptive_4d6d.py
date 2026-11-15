#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 21:58:01 2024

@author: fangfang

Simulate the AEPsych trials of the color-discrimination experiment, using a
WPPM fit as the ground-truth observer.

Ground truth: a WPPM fit (fit_6d_simobserver.py) to 60,000 nonadaptive
simulated trials (sim_nonadaptive_CIELab.py). Those trials were Sobol-sampled,
placed near the CIEDE2000 threshold (ΔE = 1.68 at 66.7% correct) with Gaussian
jitter, and their responses drawn from a Weibull psychometric function.

AEPsych runs the experiment's configuration and trial counts (run_6d_expt.py):
three Sobol strategies (1,500 trials each) followed by 20,500 adaptive (EAVC)
trials. Every AEPsych proposal gets a simulated response from the ground-truth
WPPM, which is then told back to AEPsych. Only AEPsych trials are simulated here:
the experiment's fallback Sobol trials and MOCS trials are never told to
AEPsych, so they are simulated separately.

The stimulus dimensions can be set to 2 (4D, isoluminant plane, a square
bounded between −1 and 1 in W) or 3 (6D, RGB cube), but the ground-truth fit
loaded in Section 1 is the 6D one; a 4D run needs a 2D fit.

Outputs (in ELPS_analysis/Simulation_DataFiles/<N>D/AEPsych/): the AEPsych
database (.db) and a pickle (.pkl) with the simulated trials, to be fitted with
fit_6d_simobserver.py.

NOTE: the environment needs both AEPsych and JAX (for the ground-truth
predictions), as the experiment's environment does.

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
from analysis.trial_sequence import shuffle_sobol_trials
from analysis.cross_validation import expt_data
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization,\
    Plot2DSamplingSettings
from aepsych.server import AEPsychServer
from aepsych_client import AEPsychClient
# Get the installed version of AEPsych
aepsych_version = importlib.metadata.version("aepsych")

# Stimulus dimensions (2: isoluminant plane, 4D task; 3: RGB cube, 6D task)
# and output directories for data files and figures
stim_dims = 3
psyfield_dims = stim_dims * 2
baseDir = get_path("dropbox_root_mac")

# Top-level analysis folder
root = os.path.join(baseDir, "ELPS_analysis")

# Subfolder for AEPsych simulations of this task dimensionality, next to the
# nonadaptive simulations (e.g. 6D/CIE2000, 6D/gt_Wishart). File names record
# the ground truth (_gt<coloralg>).
sim_subdir = os.path.join(f"{psyfield_dims}D", "AEPsych")

# Output directories: the AEPsych .db and the .pkl of simulated trials share a folder
output_figDir  = os.path.join(root, "Simulation_FigFiles", sim_subdir)
output_fileDir = os.path.join(root, "Simulation_DataFiles", sim_subdir)

# Create directories if they do not already exist
directories = [output_figDir, output_fileDir]
# Ensure directories exist (create if they don't)
for directory in directories:
    os.makedirs(directory, exist_ok=True)

#%% 
#------------------------------------------------------------------------------ 
# SECTION 1: Load data files and define the ground-truth model
#------------------------------------------------------------------------------ 
# Additional keyword arguments needed for 2D stimulus configurations
kwargs = {"plane_2D": "Isoluminant plane"} if stim_dims == 2 else {}

# Color-space helper (W <-> RGB conversions), used for plotting
color_thres_data = color_thresholds(stim_dims, baseDir, **kwargs)

# Load the W → display-color transformation used for the 2D task
if stim_dims == 2:
    color_thres_data.load_transformation_matrix(file_date="02242025")
    
# Load the WPPM fit to the CIE-based nonadaptive simulated trials
# (fitted with fit_6d_simobserver.py; model_pred_Wishart is saved in the file)
coloralg = 'CIE2000'
gt_fit_dir = os.path.join(baseDir, 'ELPS_analysis', 'ModelFitting_DataFiles',
                          f'{psyfield_dims}D', coloralg)
gt_fit_file = "Fitted_SimTrialData_nonadaptive_6DExpt_CIE2000_dE1.68_beta1.505_sobol_"+\
    "60000total_jitter0.3_seed0_ref0.1_0.1_0.1to0.9_0.9_0.9_decayRate0.4_nBasisDeg5.pkl"
gt_fit_path = os.path.join(gt_fit_dir, gt_fit_file)
with open(gt_fit_path, 'rb') as f:
    gt_fit = pickled.load(f)

# Fitted Wishart-model prediction object used as the simulated observer
gt_Wishart  = gt_fit['model_pred_Wishart']
# Use a tiny bandwidth to make smoothing of the simulated choice probability negligible.
gt_Wishart.opt_params['bandwidth'] = 1e-8

#%% 
#------------------------------------------------------------------------------ 
# SECTION 2: Set up configuration files
#------------------------------------------------------------------------------ 
# simulation subject
SUBJ  = 1
# Seed for the simulated response draws (NumPy's global stream). AEPsych's own
# Sobol strategies are not seeded, so trial placement still varies across runs.
rnd_seed_resp = SUBJ
np.random.seed(rnd_seed_resp)

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

# Number of trials assigned to each strategy:
#   strategies 1-3: Sobol sampling over the same parameter bounds; their
#                   comparison offsets are shrunk by the Sobol scalers below
#                   (1/4, 2/4, 3/4), hence the names small/medium/large
#   strategy 4:     model-based adaptive sampling (EAVC)
NTRIALS_STRAT = [1500, 1500, 1500, 1500]  # same as the 6D experiment (run_6d_expt.py) #20500

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
# SECTION 3: Use AEPsych for trial placement
#------------------------------------------------------------------------------ 
# AEPsych server and client. The database name records the trial counts, subject,
# response seed, and ground truth; the .pkl in Section 5 shares its stem.
db_file_name = f"Sim{psyfield_dims}dTask_colorDiscrimination_"+\
                f"{acqf_str}_{NTRIALS}Trials_{NTRIALS_STRAT[0]}_{NTRIALS_STRAT[1]}_"+\
                f"{NTRIALS_STRAT[2]}_{NTRIALS_STRAT[3]}_sub{SUBJ}_seed{rnd_seed_resp}_gt{coloralg}.db"
server = AEPsychServer(database_path = os.path.join(output_fileDir, db_file_name))
client = AEPsychClient(server=server)

# Send a config message to the server, passing in a configuration filename
client.configure(config_str=config)

# Scalers applied to the comparison offsets, one per strategy (1 = unscaled)
SOBOL_SCALER = [1/4, 2/4, 3/4, 1]

# As in the 6D experiment (run_6d_expt.py), shuffle the scalers of the first three
# (Sobol) strategies across their trials, giving one scaler per AEPsych trial.
# The adaptive strategy keeps its scaler of 1.
customized_sobol_scaler = shuffle_sobol_trials(SOBOL_SCALER, NTRIALS_STRAT, 3,
                                               seed=rnd_seed_resp)

# Simulated observer. Each time AEPsych proposes a trial, the scaled comparison
# offset is added to the reference, the ground-truth WPPM predicts p(correct) for
# the pair, and a binary response is drawn and told back to AEPsych.
AEPsych_trial_given_Wishart_gt = SimulateTrialGivenWishart(psyfield_dims,
                                                           config_all, 
                                                           gt_Wishart,
                                                           customized_val_scaler=customized_sobol_scaler
                                                           )
# Run all AEPsych trials (no fallback trials inserted; slow: many hours for 25,000 trials)
AEPsych_trial_given_Wishart_gt.run_simulation(client)

#%% 
#------------------------------------------------------------------------------ 
# SECTION 4: Visualize trial placement
#------------------------------------------------------------------------------ 
# Base plotting settings for all sampling visualizations
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 10)
pltSampSettings = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)

# Visualization object for plotting reference–comparison stimulus pairs
sampling_vis = SamplingRefCompPairVisualization(stim_dims,
                                                color_thres_data,
                                                settings = pltSampSettings,
                                                save_fig = False
                                                )

# Trials as told to AEPsych: reference and reference + scaled offset (W units)
data_vis_AEPsych = expt_data(server._strats[-1].x[:,0:stim_dims], 
                             server._strats[-1].x[:,0:stim_dims] +\
                             server._strats[-1].x[:,stim_dims:],
                             server._strats[-1].y, None)
    
# This array defines the opacity of markers in the plots, decreasing with more trials.
marker_alpha = [0.5, 0.5, 0.5, 0.2]

# Determine the index ranges of trials belonging to each strategy
# (here [0:1500], [1500:3000], [3000:4500], [4500:25000])
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
    # Figures are shown, not saved; to save, uncomment the line below
    plt.show()
    #fig.savefig(os.path.join(output_figDir, fig_name_i))
            
#%%
# ---------------------------------------------------
# SECTION 5: save data
# ----------------------------------------------------
output_file = f"{db_file_name[:-3]}.pkl"
full_path_file = os.path.join(output_fileDir, output_file)

variable_names = ['aepsych_version','stim_dims','psyfield_dims','coloralg', 
                  'gt_fit_path', 'rnd_seed_resp', 'color_thres_data', 
                  'gt_Wishart','SUBJ','file_config', 
                  'par_names', 'strat_dict', 'lb_all', 'ub_all', 'NTRIALS',
                  'config_all', 'db_file_name', 'SOBOL_SCALER', 'customized_sobol_scaler',
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

# Write the dictionary to a pickle file. The simulated trials (xref, x1, binaryResp,
# pX1, ...) are stored on AEPsych_trial_given_Wishart_gt.
with open(full_path_file, 'wb') as f:
    pickled.dump(vars_dict, f)
