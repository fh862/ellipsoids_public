#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:27:53 2024

@author: fangfang

Simulate oddity-task trials with non-adaptive placement, using a WPPM fit (to
human or simulated data) as the ground-truth observer. Used, e.g., for the 6D
experiment's fallback trials and for simulated datasets to fit.

References and directions (`num_grid_pts`):
  • None: one Sobol point per trial, [dim1, dim2, θ] on the isoluminant plane
    (4D field) or [dim1, dim2, dim3, θ, φ] in the cube (6D field).
  • An integer n: references on an n-per-dimension grid, each with an equal block
    of trials; only the directions are Sobol-sampled.

Comparison: the threshold point along the direction (p(correct) = the fit's
target, e.g. 66.7%) plus Gaussian jitter scaled by the ref-to-threshold distance.
The threshold point is found
  • exactly (flag_approx_thres = False): the step along the direction whose
    p(correct) is closest to the target, from a dense grid search (memory-heavy
    for large nSims); or
  • approximately (flag_approx_thres = True): the boundary of the WPPM covariance
    ellipse/ellipsoid at the reference, enlarged by one scalar that gives the
    target p(correct) for a unit-covariance observer.

Responses and use (`flag_sim_resp`):
  • True: a Bernoulli response per trial from the WPPM's p(correct), for fitting.
    Output: xref, x1, y.
  • False: y stays NaN; for the experiment's fallback trials, so the trials are
    also split into per-session pools (Sobol_xref, Sobol_x1; see Section 6).
The stimuli are the same either way.

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
rnd_seed = 1             # RNG seed for reproducible NumPy-side randomness
num_grid_pts = None        # references: None = Sobol-sampled; an integer = fixed grid with that many points per dimension
fixed_ref = num_grid_pts is not None
nSims = 21892              # total number of simulated trials;
                           # with a grid, it must be divisible by num_grid_pts**ndims
jitter = 0.3               # relative jitter level (sensible range: 0.1 - 0.5)
# Threshold placement:
#   - True: fast threshold approximation via covariance scaling
#       (memory-friendly; large nSims is OK).
#   - False: exact threshold search via dense grid evaluation along each direction
#       (can be memory-intensive; large nSims may exhaust RAM).
flag_approx_thres = True
# Responses, which also set what the file is for:
#   - True: simulate a binary response per trial from the ground-truth model's
#     p(correct), to fit the model to these trials.
#   - False: no responses (y stays NaN); the trials are the experiment's fallback
#     trials, so they are also split into per-session pools (Section 6).
flag_sim_resp = False

# Experiment fallback trials only (flag_sim_resp = False): number of per-session
# pools (often more than are used). Meant for Sobol references: with a grid, each
# session would get only a contiguous subset of the grid references.
nSessions = 1
# Every session gets the same number of fallback trials, so nSims must divide evenly.
# Checked here so a mismatch fails before the simulation runs.
if not flag_sim_resp and nSims % nSessions != 0:
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

# The name doesn't record the ground truth, so a run can match an existing file,
# including a participant's experiment file. Asked here, before the simulation runs.
output_file = os.path.join(output_fileDir, f"{file_stem}.pkl")
if os.path.exists(output_file):
    answer = input(f"{output_file} already exists. Overwrite it? [y/N] ")
    if answer.strip().lower() != 'y':
        raise FileExistsError(f"Kept the existing file: {output_file}. "
                              "Change rnd_seed or nSims to save under a new name.")

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
flag_plot = True           # plot the trial placement
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
# xref, x1, y: all trials, in order; what a fit to these simulated trials uses.
# Sobol_xref, Sobol_x1 (experiment files only, flag_sim_resp = False): the same
# trials split into per-session pools of fallback trials, (nSessions,
# nTrials_per_session, ndims). The experiment reads only these: each session loads
# its own row (LoadExptInfo.load_pregenerated_Sobol) and uses trials from the start
# of it as needed; the rest of the pool goes unused. Experiment files therefore
# store the trials twice (a few MB extra).
if not flag_sim_resp:
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
with open(output_file, 'wb') as f:
    pickled.dump(vars_dict, f)
