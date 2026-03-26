#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:01:11 2024

@author: fangfang

This script simulates psychophysical trials using a non-adaptive trial-placement
strategy. Rather than selecting stimuli adaptively, comparison stimuli are placed
near precomputed color-discrimination thresholds.

The discrimination thresholds are obtained from the scripts
`Isothreshold_ellipses_3slices_CIELab.py` and
`Isothreshold_ellipses_isoluminant_CIELab.py`. In those scripts, thresholds are
derived under CIELab ground truth (CIE1976, CIE1994, and CIE2000) on fixed grids
of reference stimuli.

In the present script, reference stimuli and chromatic directions are sampled
using Sobol sequences. Specifically:
  • In the 2D case, Sobol samples are drawn over [dim1, dim2, θ].
  • In the 3D case, Sobol samples are drawn over [dim1, dim2, dim3, θ, φ].

The sampled angle(s) (θ, or θ and φ) define the chromatic direction of the
comparison stimulus. Along this direction, the corresponding discrimination
threshold is computed, and a trial is placed near that threshold by adding
Gaussian jitter. The jitter magnitude is proportional to the distance between
the reference and threshold comparison stimulus.

Binary responses are simulated using a psychometric function defined as a
function of ΔE, such that a 1-JND difference (ΔE = 2.5) corresponds to 66.7%
correct performance. For each simulated trial, the ΔE between the reference
and comparison stimuli is computed, the corresponding percent-correct value
is evaluated from the psychometric function, and a binary response is generated
by sampling from a Bernoulli distribution (1 = correct, 0 = incorrect).

This script is intended for simulating data without repeated reference locations,
primarily for 2D planar slices of color space. For simulations that use fixed grids
of reference stimuli, see `sim_nonadaptive_2d_interleaved.py`.

Debugging note:
When debugging, set the jitter to a very small value (e.g., 0.01) and restrict
the Sobol bounds to a narrow region around a single reference (e.g.,
sobol_lb = [0.85, 0.15, 0], sobol_ub = [0.85, 0.15, 360]).
If the sampling procedure is implemented correctly, the simulated trials should
lie very close to the ground-truth threshold contour.

"""

import os
import numpy as np
import dill as pickled
import matplotlib.pyplot as plt
from dataclasses import replace
from analysis.simulations_CIELab import SimThresCIELab
from analysis.trial_placement import StimConfig_RGBslices_sobolref, \
    StimConfig_isoluminant_sobolref, TrialPlacement_sobolRef
from analysis.color_thres import color_thresholds
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization, \
    Plot2DSamplingSettings
from plotting.wishart_plotting import PlotSettingsBase
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization,\
        PlotWeibullPMFSettings
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'

#%% 
# -----------------------------------------------------------
# Set up color-threshold object + stimulus configuration
# -----------------------------------------------------------
stim_dims = 2                  #even though the stimulus lives in 2D space, 
psyfield_dims = 4              #this is actually a 4d psychometric field.
rnd_seed = 0                   #for reproducibility
colordiff_alg = "CIE1994"      #algorithm for color differences
plane_2D = "Isoluminant plane" #"GB plane", "RB plane", "RG plane", "Isoluminant plane"
jitter = 0.3                   # Sampling noise level (fraction of ref->threshold distance); keep tiny for debugging
nSims = 6000                   # Number of simulated trials 

# Color-threshold helper
color_thres_data = color_thresholds(stim_dims,
                                    base_dir,
                                    plane_2D=plane_2D,
                                    )

# Common config fields shared by both RGB slices and the isoluminant plane
common_kwargs = dict(gt=colordiff_alg,
                     random_seed=rnd_seed,
                     nSims=nSims,
                     random_jitter=jitter,
                     )

# Build the appropriate stimulus config for this plane
if plane_2D == "Isoluminant plane":    
    # Isoluminant: work in 2D W-space (bounded [-1, 1]) but keep transforms for RGB <-> W
    color_thres_data.load_transformation_matrix(file_date="02242025")

    # Sobol samples are [dim1, dim2, theta] where theta sets the direction in the plane
    # Bounds below span the usual model-space region (adjust as needed).
    stim_config = StimConfig_isoluminant_sobolref(
        **common_kwargs,
        M_RGBTo2DW=color_thres_data.M_RGBTo2DW,
        M_2DWToRGB=color_thres_data.M_2DWToRGB,
        sobol_lb = [-0.7, -0.7, 0],    #data will be in model space (bounded between -1 and 1) for the isoluminant plane
        sobol_ub = [0.7, 0.7, 360]
    )
else:    
    # RGB slice: choose which RGB channel is held fixed based on the selected 2D plane
    plane_to_fixed = {"GB plane": "R", "RB plane": "G", "RG plane": "B"}
    fixed_val = 0.5  # typical mid-gray slice; change if you want a different slice
    
    # Sobol samples are [dim1, dim2, theta] where dim1/dim2 are the *varying* RGB channels.
    stim_config = StimConfig_RGBslices_sobolref(
        **common_kwargs,
        fixed_val = fixed_val,
        fixed_plane=plane_to_fixed[plane_2D],
        sobol_lb = [0.15, 0.15, 0],   #data will be in normalized RGB space (bounded between 0 and 1) for other planes
        sobol_ub = [0.85, 0.85, 360]
    )
    
    color_thres_data.fixed_value = fixed_val

#%%
# -----------------------------------------------------------
# Load GT CIE dataset + initialize trial placement object
# -----------------------------------------------------------
# Set up the simulation object
sim_trial = TrialPlacement_sobolRef(config = stim_config)

#% Define the Weibull psychometric function with specified parameters
# Calculate the probability of correct response given alpha and beta.
deltaE_1JND = 2.5
sim_trial.setup_WeibullFunc(alpha = 3.189, 
                            beta = 1.505, 
                            guessing_rate = 1/3, 
                            deltaE_1JND= deltaE_1JND
                            ) 
# Print the target probability based on the Weibull function for the given delta E
print(f"target probability: {sim_trial.sim['pC_given_alpha_beta']}")

# Initialize the SimThresCIELab object with a path to necessary files and background RGB value
background_RGB = np.array([0.5, 0.5, 0.5])
sim_CIELab = SimThresCIELab(background_RGB)
sim_trial.run_sim(sim_CIELab)    
    
#%%
# Define output direcotries
output_figDir = os.path.join(base_dir, 'Simulation_FigFiles',f'{psyfield_dims}D', f'{colordiff_alg}')
output_fileDir = os.path.join(base_dir, 'Simulation_DataFiles',f'{psyfield_dims}D', f'{colordiff_alg}')
os.makedirs(output_figDir, exist_ok=True)
os.makedirs(output_fileDir, exist_ok=True)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 8)

#figure name
str_optional = (plane_2D.replace(" ", "") + '_') if stim_dims == 2 else ''
figname = f"SimTrialData_nonadaptive_{psyfield_dims}DExpt_{str_optional}{colordiff_alg}_"+\
    f"{nSims}total_jitter{jitter}_seed{rnd_seed}"

#first visualize the Weibull psychometric functions
sim_vis = TrialPlacementVisualization(sim_trial, 
                                      settings = pltSettings_base,
                                      save_fig = False)
pltSettings_PMF = replace(PlotWeibullPMFSettings(), **pltSettings_base.__dict__)
pltSettings_PMF = replace(pltSettings_PMF, xticks = np.linspace(0, 9, 4))
x_PMF = np.linspace(0,9,100)
sim_vis.plot_WeibullPMF(x_PMF, settings = pltSettings_PMF)

# Create settings instance with custom fig_dir
bounds = np.array([stim_config.sobol_lb[0], stim_config.sobol_ub[0]])
if plane_2D != 'Isoluminant plane':
    bounds = color_thres_data.N_unit_to_W_unit(bounds)
pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
pltSettings_tp = replace(pltSettings_tp,
                         linealpha = 0.3,        # Line transparency for this subset of data
                         bounds = bounds,
                         ref_markersize = 10,
                         ticks = np.linspace(-0.7, 0.7, 5),
                         flag_rescale_axes_label = False if plane_2D == 'Isoluminant plane' else True,
                         ref_markeralpha = 0.6,
                         comp_markeralpha = 0.3,
                         )
sampling_vis = SamplingRefCompPairVisualization(stim_dims,
                                                color_thres_data,
                                                settings = pltSettings_tp,
                                                save_fig = False
                                                )

# These two sets of data are selected for no particular reason
vd = sim_trial.config.varying_color_dim
xref = sim_trial.sim['ref'].T
x1 = sim_trial.sim['comp'].T
if plane_2D != 'Isoluminant plane':
    xref = color_thres_data.N_unit_to_W_unit(xref) #the last row is a filler row (all 1's)
    x1 = color_thres_data.N_unit_to_W_unit(x1) #so we can just get rid of that row

fig, ax = plt.subplots(1, 1, figsize = pltSettings_tp.fig_size, dpi= pltSettings_tp.dpi)
sampling_vis.plot_sampling(xref[:,vd], x1[:,vd], ax = ax,
                           settings = pltSettings_tp)     
ax.set_title(f'{plane_2D} ({colordiff_alg})')
# Save the figure as a PDF
fig.savefig(os.path.join(output_figDir, f'{figname}.pdf'), bbox_inches='tight')    
plt.show()


#%% save to pkl
full_path = os.path.join(output_fileDir, f'{figname}.pkl')

variable_names = ['sim_trial','color_thres_data','background_RGB', 'sim_CIELab']
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
with open(full_path, 'wb') as f:
    pickled.dump(vars_dict, f)


