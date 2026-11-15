#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:27:53 2024

@author: fangfang

Simulate oddity-task trials with non-adaptive trial placement, using a
Wishart-process model (WPPM) fit as the ground truth.

Discrimination thresholds come from a WPPM fit to human psychophysical data.
Reference stimuli are either Sobol-sampled or fixed on a regular grid (`fixed_ref`);
chromatic directions are always Sobol-sampled:
  • Sobol references (fixed_ref = False): one Sobol point per trial,
    [dim1, dim2, θ] on the 2D isoluminant plane (4D psychometric field) or
    [dim1, dim2, dim3, θ, φ] in the 3D cube (6D psychometric field).
  • Fixed-grid references (fixed_ref = True): `num_grid_pts` points per dimension,
    spanning the same reference bounds. Each grid point gets an equal block of the
    `nSims` trials, and only the angles (θ in 2D; θ and φ in 3D) are Sobol-sampled.

For each reference and chromatic direction, there are two ways to place the
comparison stimulus.

(1) flag_approx_thres = False  (exact threshold via grid search)
    - Treat the sampled direction as fixed and march outward from the reference
      along that direction using a dense grid of step sizes (vector lengths).
    - For each candidate comparison stimulus, evaluate the WPPM observer to get
      p(correct) for the oddity task.
    - Pick the candidate whose p(correct) is closest to the model's target
      threshold performance (e.g., 66.7% correct), and use that point as the
      "threshold comparison" for this (ref, direction) pair.
    - Add Gaussian jitter around this threshold point (scaled by the
      ref-to-threshold distance) to generate the trial's comparison stimulus.

(2) flag_approx_thres = True  (fast threshold approximation via covariance scaling)
    - Instead of searching over step sizes, approximate the threshold contour
      using the model's predicted sensory-noise covariance at the reference.
    - For each reference, compute the covariance (Σ) implied by the WPPM, convert
      Σ into ellipse / ellipsoid parameters (axis lengths and rotation angles),
      and enlarge the axes by a scalar factor.
    - This scalar ("Sigma_scaler") is chosen so that a displacement to the enlarged
      ellipse / ellipsoid boundary corresponds approximately to the target threshold
      performance. It is computed once by simulating the oddity observer with unit
      covariance and finding the scale that yields the target p(correct).
    - Given the sampled direction, compute the distance from the ellipse center
      to the boundary along that direction, yielding an approximate threshold point.
    - As in the exact method, add Gaussian jitter around that threshold point to
      generate the final comparison stimulus.

In both approaches, if `flag_sim_resp` is True, the response is then simulated by
evaluating the model-predicted p(correct) for each (ref, comp) pair and sampling a
Bernoulli response (1 = correct, 0 = incorrect). Otherwise the responses stay NaN:
the experiment's fallback trials only need the stimuli. The stimuli are the same
either way.

"""

import os
import numpy as np
import dill as pickled
import matplotlib.pyplot as plt
from dataclasses import replace
from analysis.utils_load import get_path, select_file_and_get_path, extract_sub_number
from analysis.trial_placement import TrialPlacement_sobolRef_W, StimConfig_W_sobolref
from plotting.wishart_plotting import PlotSettingsBase
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization, \
    Plot2DSamplingSettings, Plot3DSamplingHTMLSettings, SamplingRefCompPairVisualization_html

# Root of the ELPS Dropbox folder; simulated data and figures are saved under it
base_dir = get_path("dropbox_root_mac_elps")

#%%
# -----------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------
rnd_seed = 800             # RNG seed for reproducible NumPy-side randomness
fixed_ref = False          # True: references on a fixed grid; False: Sobol-sampled references
num_grid_pts = 7           # fixed_ref only: grid points per dimension
nSims = 25000              # total number of simulated trials;
                           # with fixed_ref, it must be divisible by num_grid_pts**ndims
jitter = 0.3               # relative jitter level (sensible range: 0.1 - 0.5)
# Threshold placement:
#   - True: fast threshold approximation via covariance scaling
#       (memory-friendly; large nSims is OK).
#   - False: exact threshold search via dense grid evaluation along each direction
#       (can be memory-intensive; large nSims may exhaust RAM).
flag_approx_thres = True
# Simulate a binary response per trial from the ground-truth model's p(correct).
# Not needed for the experiment's fallback trials; slow for large nSims.
flag_sim_resp = False
flag_plot = True           # plot the trial placement

# To slot these simulated Sobol trials into the real experiment as "fallback"
# trials, they are split into session-sized chunks. This is meant for Sobol
# references: with fixed_ref, each session would get only a contiguous subset of
# the grid references.
flag_use_in_expt = True
nSessions = 50             # number of sessions to split the trials into (often more than are used)
# Every session gets the same number of fallback trials, so nSims must divide evenly.
# Checked here so a mismatch fails before the simulation runs.
if flag_use_in_expt and nSims % nSessions != 0:
    raise ValueError(
        f"nSims ({nSims}) must be divisible by nSessions ({nSessions}) "
        "to split trials evenly across sessions."
    )

# Select a pre-fit WPPM/Wishart model file:
#   - 2D isoluminant-plane fit (4D experiment): 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.4_varScaler0.0003_nBasisDeg5.pkl'
#   - 3D RGB-cube fit (6D experiment):          'Fitted_ColorDiscrimination_6dExpt_RGBcube_sub1_decayRate0.4_nBasisDeg5.pkl'
file_dir, file_name = select_file_and_get_path()
subN = extract_sub_number(file_name)
gt_file_path = os.path.join(file_dir, file_name)

#%%
# -----------------------------------------------------------
# 2. Ground-truth model and trial-placement config
# -----------------------------------------------------------
with open(gt_file_path, "rb") as f:
    gt_Wishart = pickled.load(f)
# Use a tiny bandwidth to make smoothing of the simulated choice probability negligible.
# The fit file is a dict; the bandwidth lives on the model-prediction object.
gt_Wishart['model_pred_Wishart'].opt_params['bandwidth'] = 1e-8

color_thres_data = gt_Wishart["color_thres_data"]
ndims = color_thres_data.color_dimension   # 2 for isoluminant; 3 for RGB cube

# Sobol bounds, in the model (W) space used by the fit:
#   - 2D: [dim1, dim2, theta_deg], where theta sets the direction in the plane
#   - 3D: [dim1, dim2, dim3, theta_deg, phi_deg], where (theta, phi) set a 3D direction
# With fixed_ref, the reference bounds set the grid's range (±0.7 matches the
# grid the fits are evaluated on) and only the angles are sampled.
if ndims == 2:
    fixed_plane = "lum"                    # isoluminant plane convention in config
    ref_sobol_lb = [-0.8, -0.8, 0]
    ref_sobol_ub = [0.8, 0.8, 360]
else:
    fixed_plane = ""                       # no slicing: full 3D cube
    ref_sobol_lb = [-0.8, -0.8, -0.8, 0, 0]
    ref_sobol_ub = [0.8, 0.8, 0.8, 360, 180]

stim_config = StimConfig_W_sobolref(fixed_plane=fixed_plane,
                                    random_seed=rnd_seed,
                                    nSims=nSims,
                                    fixed_ref=fixed_ref,
                                    num_grid_pts=num_grid_pts,
                                    random_jitter=jitter,
                                    sobol_lb=ref_sobol_lb,
                                    sobol_ub=ref_sobol_ub
                                    )

#%%
# -----------------------------------------------------------
# 3. Output files
# -----------------------------------------------------------
# Fixed-grid runs are labeled by the stimulus dimensions (2D/3D, the convention for
# fixed reference grids); Sobol runs by the psychometric-field dimensions (4D/6D),
# the folder and name the experiment scripts expect
expt_dims = ndims if fixed_ref else 2 * ndims
output_figDir = os.path.join(base_dir, 'Simulation_FigFiles', f'{expt_dims}D', 'gt_Wishart')
output_fileDir = os.path.join(base_dir, 'Simulation_DataFiles', f'{expt_dims}D', 'gt_Wishart')
os.makedirs(output_figDir, exist_ok=True)
os.makedirs(output_fileDir, exist_ok=True)

# Shared stem of the figure and data files. The experiment scripts (e.g.
# run_6d_expt.py) load the Sobol fallback trials by this name, so the Sobol-mode
# format stays as it is. Fixed-grid files add the grid size and reference bounds.
plane_tag = (color_thres_data.plane_2D.replace(" ", "") + '_') if ndims == 2 else ''
thres_tag = 'true_thres_' if not flag_approx_thres else ''
grid_tag = (f"grid{num_grid_pts}_ref{'_'.join(map(str, ref_sobol_lb[:ndims]))}"
            f"to{'_'.join(map(str, ref_sobol_ub[:ndims]))}_") if fixed_ref else ''
# Files with simulated responses are marked; files without keep the experiments' name
resp_tag = 'simResp_' if flag_sim_resp else ''
file_stem = f"SimTrialData_nonadaptive_{expt_dims}DExpt_{plane_tag}{thres_tag}{grid_tag}"+\
    f"{resp_tag}{nSims}total_jitter{jitter}_seed{rnd_seed}"
print(f"Output file stem: {file_stem}")

#%%
# -----------------------------------------------------------
# 4. Simulate trials
# -----------------------------------------------------------
# Generates refs/dirs, finds threshold comps, jitters, and (if flag_sim_resp)
# simulates responses
sim_trial = TrialPlacement_sobolRef_W(gt_Wishart, stim_config)
sim_trial.run_sim(flag_approx=flag_approx_thres, flag_sim_resp=flag_sim_resp)

# Trial list, one row per trial (W units): xref, x1 (nSims, ndims); y (nSims,).
# sim["ref"] and sim["comp"] are stored dims-first: (nd_total, nSims).
vd = stim_config.varying_color_dim
xref = sim_trial.sim["ref"].T[..., vd]
x1 = sim_trial.sim["comp"].T[..., vd]
y = sim_trial.sim["resp_binary"]

#%%
# -----------------------------------------------------------
# 5. Visualize
# -----------------------------------------------------------
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize=8)

if flag_plot:
    if ndims == 2:
        ref_bounds = [ref_sobol_lb[0], ref_sobol_ub[0]]
        pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
        pltSettings_tp = replace(pltSettings_tp,
                                 bounds=ref_bounds,
                                 ref_markersize=20,
                                 ref_markeralpha=0.5,
                                 comp_markeralpha=0.5,
                                 linealpha=0.2,
                                 ticks=np.linspace(*ref_bounds, 5),
                                 flag_rescale_axes_label=False,
                                 fig_name=f"{file_stem}.pdf"
                                 )
        sampling_vis = SamplingRefCompPairVisualization(ndims,
                                                        color_thres_data,
                                                        settings=pltSettings_tp,
                                                        save_fig=True
                                                        )
        sampling_vis.plot_sampling(xref, x1, settings=pltSettings_tp)
        plt.show()
    else:
        plt3Dhtml_settings = replace(Plot3DSamplingHTMLSettings(), font_size=12)
        vis_sample_html = SamplingRefCompPairVisualization_html(settings=plt3Dhtml_settings)
        vis_sample_html.write_interactive_html(xref,
                                               x1,
                                               os.path.join(output_figDir, f"{file_stem}.html"),
                                               page_title=file_stem,
                                               )

#%%
# -----------------------------------------------------------
# 6. Save
# -----------------------------------------------------------
if flag_use_in_expt:
    # Split the trials into sessions: (nSessions, nTrials_per_session, ndims)
    nTrials_per_session = nSims // nSessions
    Sobol_xref = xref.reshape(nSessions, nTrials_per_session, ndims)
    Sobol_x1 = x1.reshape(nSessions, nTrials_per_session, ndims)
else:
    Sobol_xref, Sobol_x1 = None, None

vars_dict = {'gt_file_path': gt_file_path,
             'sim_trial': sim_trial,
             'color_thres_data': color_thres_data,
             'xref': xref,
             'x1': x1,
             'y': y,
             'Sobol_xref': Sobol_xref,
             'Sobol_x1': Sobol_x1,
             }
with open(os.path.join(output_fileDir, f"{file_stem}.pkl"), 'wb') as f:
    pickled.dump(vars_dict, f)
