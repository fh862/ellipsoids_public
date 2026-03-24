#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Dec 16 13:27:53 2024

@author: fangfang

The goal of this script is to simulate psychophysical trials using a non-adaptive
trial-placement strategy. Instead of selecting stimuli adaptively, comparison
stimuli are placed near precomputed discrimination thresholds.

Discrimination thresholds are obtained from Wishart-process model fits to human
psychophysical data. Different from `sim_nonadaptive_fixedRef_W.py`, here the reference
will not be fixed on a grid, but sobol sampled instead.

For each sampled reference and chromatic direction, we have two ways to determine
where to place the comparison stimulus.

(1) flag_approx = False  (exact threshold via grid search)
    - We treat the sampled direction as fixed and march outward from the reference
      along that direction using a dense grid of step sizes (vector lengths).
    - For each candidate comparison stimulus, we evaluate the Wishart observer model
      to get p(correct) for the oddity task.
    - We then pick the candidate whose p(correct) is closest to the model’s target
      threshold performance (e.g., 66.7% correct), and use that point as the
      “threshold comparison” for this (ref, direction) pair.
    - Finally, we add Gaussian jitter around this threshold point (scaled by the
      ref→threshold distance) to generate the actual trial comparison stimulus.

(2) flag_approx = True  (fast threshold approximation via covariance scaling)
    - Instead of searching over step sizes, we approximate the threshold contour
      using the model’s predicted sensory-noise covariance at the reference.
    - For each reference, we compute the covariance (Σ) implied by the Wishart model,
      convert Σ into ellipse / ellipsoid parameters (axis length and rotation angles),
      and then enlarge the axes by a scalar factor.
    - This scalar (“Sigma_scaler”) is chosen so that a displacement to the enlarged
      ellipse / ellipsoid boundary corresponds approximately to the target threshold 
      performance (it is computed once by simulating the oddity observer with unit 
      covariance and inding the scale that yields the target p(correct)).
    - Given the sampled direction, we compute the distance from the ellipse center
      to the boundary along that direction, yielding an approximate threshold point.
    - As in the exact method, we then add Gaussian jitter around that threshold point
      to generate the final comparison stimulus.

In both approaches, after placing the comparison stimulus we simulate the response
by evaluating the model-predicted p(correct) for the (ref, comp) pair and sampling
a Bernoulli response (1 = correct, 0 = incorrect).

"""

import sys
import numpy as np
import dill as pickled
import os
from dataclasses import replace
import matplotlib.pyplot as plt
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.trial_placement import TrialPlacement_sobolRef_W, StimConfig_W_sobolref
from analysis.utils_load import select_file_and_get_path, extract_sub_number
from plotting.wishart_plotting import PlotSettingsBase
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization, \
    Plot2DSamplingSettings
from plotting.adaptive_sampling_plotting import Plot3DSamplingHTMLSettings,\
    SamplingRefCompPairVisualization_html

#%% 
# -----------------------------------------------------------
# Set up stimulus configuration and run simulations
# -----------------------------------------------------------
rnd_seed = 800       # RNG seed for reproducible NumPy-side randomness
nSims = 75000        # number of simulated trials (one Sobol sample per trial)
jitter = 0.3       # relative jitter level (sensible range: 0.1 - 0.5)

# Select a pre-fit WPPM/Wishart model file:
#   - 2D isoluminant-plane fit (4D experiment): 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl'
#   - 3D RGB-cube fit (6D experiment):          'Fitted_ColorDiscrimination_6dExpt_RGBcube_sub1_decayRate0.4_nBasisDeg5.pkl'
file_dir, file_name = select_file_and_get_path()
subN = extract_sub_number(file_name)
gt_file_path = os.path.join(file_dir, file_name)

# Load ground-truth model object + metadata
with open(gt_file_path, "rb") as f:
    gt_Wishart = pickled.load(f)
color_thres_data = gt_Wishart["color_thres_data"]
ndims = color_thres_data.color_dimension         

# Choose Sobol bounds based on dimensionality:
#   - 2D: sample [dim1, dim2, theta] where theta sets direction in the plane
#   - 3D: sample [dim1, dim2, dim3, theta, phi] where (theta, phi) set a 3D direction
#
# NOTE: bounds here are in the same coordinate system as the model space used by the fit.
if ndims == 2:
    fixed_plane = "lum"                    # isoluminant plane convention in config
    ref_sobol_lb = [-0.8, -0.8, 0]         # lb: [dim1, dim2, theta_deg] 
    ref_sobol_ub = [0.8, 0.8, 360]         # ub 
else:
    fixed_plane = ""                       # no slicing: full 3D cube
    ref_sobol_lb = [-0.8, -0.8, -0.8, 0, 0]      # [dim1, dim2, dim3, theta_deg, phi_deg]
    ref_sobol_ub = [0.8, 0.8, 0.8, 360, 180]     # ub

# Pack simulation settings into a config object (used by the trial-placement class)
stim_config = StimConfig_W_sobolref(fixed_plane= fixed_plane,
                                    random_seed=rnd_seed,
                                    nSims=nSims,
                                    random_jitter=jitter,
                                    sobol_lb= ref_sobol_lb,
                                    sobol_ub= ref_sobol_ub
                                    )

# Initialize trial-placement simulator (generates refs/dirs, finds threshold comps, 
# jitters, simulates responses)
sim_trial = TrialPlacement_sobolRef_W(gt_Wishart, stim_config)

# Run the simulation.
#   - flag_approx=True: fast threshold approximation via covariance scaling 
#       (memory-friendly; large nSims is OK).
#   - flag_approx=False: exact threshold search via dense grid evaluation along 
#       each direction (can be memory-intensive; large nSims may exhaust RAM).
flag_approx_thres = True
sim_trial.run_sim(flag_approx= flag_approx_thres)

# varying dimensions (2D: [0, 1]; 3D: [0, 1, 2])
vd = stim_config.varying_color_dim

# Reformat outputs into "trial list" arrays:
#   - xref: (nSims, ndims)
#   - x1:   (nSims, ndims)
#   - y:    (nSims,)
#
# NOTE: `sim["ref"]` and `sim["comp"]` are stored with dims-first convention (nd_total, nSims).
xref = np.moveaxis(sim_trial.sim["ref"], 0, -1)[..., vd]
x1 = np.moveaxis(sim_trial.sim["comp"], 0, 1)[..., vd]
y = sim_trial.sim["resp_binary"]

#%% 
# -----------------------------------------------------------
# Visualize trial placement
# -----------------------------------------------------------
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'
output_figDir = os.path.join(base_dir, 'Simulation_FigFiles',f'{ndims*2}D', 'gt_Wishart')
output_fileDir = os.path.join(base_dir, 'Simulation_DataFiles',f'{ndims*2}D', 'gt_Wishart')
os.makedirs(output_figDir, exist_ok=True)
os.makedirs(output_fileDir, exist_ok=True)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 8)

#figure name
str_optional = (color_thres_data.plane_2D.replace(" ", "") + '_') if ndims == 2 else ''
str2_optional = 'true_thres_' if not flag_approx_thres else ''
figname = f"SimTrialData_nonadaptive_{ndims*2}DExpt_{str_optional}{str2_optional}"+\
    f"{nSims}total_jitter{jitter}_seed{rnd_seed}"
flag_plot = False
if flag_plot:
    if ndims == 2:    
        pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
        pltSettings_tp = replace(pltSettings_tp,
                                 bounds = [ref_sobol_lb[0], ref_sobol_ub[0]],
                                 ref_markersize = 20,
                                 ref_markeralpha = 0.5,
                                 comp_markeralpha = 0.5,
                                 linealpha = 0.2,
                                 ticks = np.linspace(-0.7, 0.7, 5),
                                 flag_rescale_axes_label = False,
                                 fig_name = f"{figname}.pdf"
                                 )
        
        sampling_vis = SamplingRefCompPairVisualization(ndims,
                                                        color_thres_data,
                                                        settings = pltSettings_tp,
                                                        save_fig = True
                                                        )
        sampling_vis.plot_sampling(xref, x1, settings = pltSettings_tp)     
        plt.show()
    else:
        plt3Dhtml_settings = Plot3DSamplingHTMLSettings()
        plt3Dhtml_settings = replace(plt3Dhtml_settings, font_size = 12)
        vis_sample_html = SamplingRefCompPairVisualization_html(settings=plt3Dhtml_settings)
        fig = vis_sample_html.plot_sampling(xref, x1)
        out_html = os.path.join(output_figDir, f"{figname}.html")
        fig.write_html(out_html, include_plotlyjs=True)

#%% 
# -----------------------------------------------------------
# Save data
# -----------------------------------------------------------
# If we want to slot these simulated Sobol trials into the real experiment as
# "fallback" trials, we need to split them into session-sized chunks.
flag_use_in_expt = True
if flag_use_in_expt:
    # Number of sessions to split the simulated trials into.
    # (We often generate more sessions than we end up using.)
    nSessions = 50

    # Total number of simulated trials must be divisible by nSessions so that each
    # session gets the same number of fallback trials.
    if nSims % nSessions != 0:
        raise ValueError(
            f"nSims ({nSims}) must be divisible by nSessions ({nSessions}) "
            "to split trials evenly across sessions."
        )

    # Trials per session (integer)
    nTrials_per_session = nSims // nSessions

    # Reshape to (nSessions, nTrials_per_session, ndims)
    Sobol_xref = xref.reshape(nSessions, nTrials_per_session, ndims)
    Sobol_x1   = x1.reshape(nSessions, nTrials_per_session, ndims)

output_path = os.path.join(output_fileDir,f'{figname}.pkl')
variable_names = ['gt_file_path','sim_trial','color_thres_data','xref', 'x1', 'y',
                  'Sobol_xref', 'Sobol_x1']
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
with open(output_path, 'wb') as f:
    pickled.dump(vars_dict, f)


        