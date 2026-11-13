#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:01:11 2024

@author: fangfang

Simulate oddity-task trials with non-adaptive trial placement and a CIELab ΔE
ground truth (CIE1976, CIE1994, or CIE2000), on a 2D planar slice (4D psychometric
field) or in the 3D RGB cube (6D psychometric field).

Reference stimuli are either Sobol-sampled or fixed on a regular grid (`fixed_ref`);
chromatic directions are always Sobol-sampled:
  • Sobol references (fixed_ref = False): one Sobol point per trial, [dim1, dim2, θ]
    in 2D or [R, G, B, θ, φ] in 3D, so no reference location is repeated.
  • Fixed-grid references (fixed_ref = True): `num_grid_pts` points per dimension,
    spanning the same reference bounds. Each grid point gets an equal block of the
    `nSims` trials, and only the angles (θ in 2D; θ and φ in 3D) are Sobol-sampled.
The angle(s) set the chromatic direction of the comparison stimulus.

Each trial is simulated in three steps:
  1. Along the sampled direction, solve for the comparison stimulus whose ΔE from
     the reference equals `deltaE_1JND`. The threshold is solved trial by trial
     with `SimThresCIELab`; no precomputed thresholds are loaded.
  2. Add Gaussian jitter to that threshold point. In each channel, the jitter SD
     is `jitter` times the reference-to-threshold distance.
  3. Compute ΔE between the reference and the jittered comparison, map it to
     p(correct) with a Weibull psychometric function, and draw a binary response
     from a Bernoulli distribution (1 = correct, 0 = incorrect).

The Weibull scale `alpha` is derived from `deltaE_1JND`, `beta`, and the guessing
rate, so that ΔE = deltaE_1JND gives p(correct) = `target_pC`.

Debugging note:
When debugging, set the jitter to a very small value (e.g., 0.01) and restrict
the Sobol bounds to a single reference (e.g., for a 2D RGB slice,
sobol_lb = [0.85, 0.15, 0], sobol_ub = [0.85, 0.15, 360]).
If the sampling procedure is implemented correctly, the simulated trials should
lie very close to the ΔE = deltaE_1JND contour around that reference.

"""

import os
import numpy as np
import dill as pickled
import matplotlib.pyplot as plt
from dataclasses import replace
from analysis.utils_load import get_path
from analysis.simulations_CIELab import SimThresCIELab
from analysis.trial_placement import StimConfig_CIELab_RGB, \
    StimConfig_CIELab_isoluminant, TrialPlacement_CIELab
from analysis.color_thres import color_thresholds
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization, \
    Plot2DSamplingSettings, Plot3DSamplingHTMLSettings, SamplingRefCompPairVisualization_html
from plotting.wishart_plotting import PlotSettingsBase
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization,\
        PlotWeibullPMFSettings

# Root of the ELPS Dropbox folder; simulated data and figures are saved under it
base_dir = get_path("dropbox_root_mac_elps")

#%%
# -----------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------
stim_dims = 3                  # 2: planar slice (4D psychometric field); 3: RGB cube (6D psychometric field)
if stim_dims == 2:
    plane_2D = "Isoluminant plane"   # "GB plane", "RB plane", "RG plane", "Isoluminant plane"
psyfield_dims = 2 * stim_dims  # reference + comparison dimensions
colordiff_alg = "CIE2000"      # ΔE formula: "CIE1976", "CIE1994", "CIE2000"
fixed_ref = False              # True: references on a fixed grid; False: Sobol-sampled references
num_grid_pts = 5               # fixed_ref only: grid points per dimension
nSims = 60000                   # total number of simulated trials (Sobol points are best balanced for a power of 2);
                               # with fixed_ref, it must be divisible by num_grid_pts**stim_dims
jitter = 0.3                   # jitter SD as a fraction of the ref-to-threshold distance; keep tiny for debugging
rnd_seed = 0                   # for reproducibility
background_RGB = np.array([0.5, 0.5, 0.5])  # display background; also the CIELab white point
# Monitor calibration (MMDDYYYY) for converting display RGB to XYZ (and, on the
# isoluminant plane, W to RGB): the formal 3D experiment's display for the RGB cube,
# the 4D (eLife) experiment's display for 2D planes
calib_date = "02012026" if stim_dims == 3 else "02242025"

# Weibull psychometric function: ΔE = deltaE_1JND gives p(correct) = target_pC
scaler_lookup_dict = {'CIE1976': 2.2, 'CIE1994': 1.68, 'CIE2000': 1.68}
deltaE_1JND = scaler_lookup_dict[colordiff_alg]        # ΔE of 1 JND; comparisons are placed around this ΔE
target_pC = 2/3                # p(correct) at deltaE_1JND
guessing_rate = 1/3            # chance performance in the 3-alternative oddity task
beta = 1.505                   # Weibull slope
# Solve 1 - (1 - guessing_rate) * exp(-(deltaE_1JND / alpha)**beta) = target_pC for alpha
alpha = deltaE_1JND / (-np.log((1 - target_pC) / (1 - guessing_rate)))**(1 / beta)

#%%
# -----------------------------------------------------------
# 2. Stimulus space and trial-placement config
# -----------------------------------------------------------
# Config fields shared by every stimulus space
common_kwargs = dict(gt=colordiff_alg,
                     random_seed=rnd_seed,
                     nSims=nSims,
                     fixed_ref=fixed_ref,
                     num_grid_pts=num_grid_pts,
                     random_jitter=jitter,
                     )

# The Sobol bounds below list the reference bounds, then the angle bounds. With
# fixed_ref, the reference bounds set the grid's range and only the angles are sampled.
if stim_dims == 3:
    # RGB cube: fixed_plane='' varies all three channels in normalized RGB (bounded between 0 and 1).
    # Sobol samples are [R, G, B, theta, phi]; theta is the azimuth in the R-G plane and
    # phi the inclination from +B, both in degrees.
    # Note: phi is sampled uniformly in degrees, so directions are denser near +/-B.
    color_thres_data = color_thresholds(stim_dims, base_dir)
    stim_config = StimConfig_CIELab_RGB(
        **common_kwargs,
        fixed_plane='',
        sobol_lb=[0.1, 0.1, 0.1, 0, 0],
        sobol_ub=[0.9, 0.9, 0.9, 360, 180],
    )
elif plane_2D == "Isoluminant plane":
    # Isoluminant plane: sample in 2D W (model) space, bounded between -1 and 1.
    # The RGB <-> W matrices convert stimuli to RGB before ΔE is computed.
    # Sobol samples are [dim1, dim2, theta], where theta sets the direction in the plane.
    color_thres_data = color_thresholds(stim_dims, base_dir, plane_2D=plane_2D)
    color_thres_data.load_transformation_matrix(file_date=calib_date)
    stim_config = StimConfig_CIELab_isoluminant(
        **common_kwargs,
        M_RGBTo2DW=color_thres_data.M_RGBTo2DW,
        M_2DWToRGB=color_thres_data.M_2DWToRGB,
        sobol_lb=[-0.7, -0.7, 0],
        sobol_ub=[0.7, 0.7, 360],
    )
else:
    # RGB slice: hold one channel fixed and vary the other two in normalized RGB
    # (bounded between 0 and 1). Sobol samples are [dim1, dim2, theta], where
    # dim1/dim2 are the varying channels.
    plane_to_fixed = {"GB plane": "R", "RB plane": "G", "RG plane": "B"}
    fixed_val = 0.5            # value of the fixed channel (mid-gray slice)
    color_thres_data = color_thresholds(stim_dims, base_dir, plane_2D=plane_2D,
                                        fixed_value=fixed_val)
    stim_config = StimConfig_CIELab_RGB(
        **common_kwargs,
        fixed_plane=plane_to_fixed[plane_2D],
        fixed_val=fixed_val,
        sobol_lb=[0.15, 0.15, 0],
        sobol_ub=[0.85, 0.85, 360],
    )

# Bounds of the reference location (the remaining Sobol entries are angles)
ref_lb = stim_config.sobol_lb[:stim_dims]
ref_ub = stim_config.sobol_ub[:stim_dims]

#%%
# -----------------------------------------------------------
# 3. Output files
# -----------------------------------------------------------
# Fixed-grid runs are labeled by the stimulus dimensions (2D/3D, the convention for
# fixed reference grids); Sobol runs by the psychometric-field dimensions (4D/6D)
expt_dims = stim_dims if fixed_ref else psyfield_dims
output_figDir = os.path.join(base_dir, 'Simulation_FigFiles', f'{expt_dims}D', f'{colordiff_alg}')
output_fileDir = os.path.join(base_dir, 'Simulation_DataFiles', f'{expt_dims}D', f'{colordiff_alg}')
os.makedirs(output_figDir, exist_ok=True)
os.makedirs(output_fileDir, exist_ok=True)

# Shared stem of the figure and data files. It records every setting that changes
# the simulated trials, so runs with different settings don't overwrite each other.
# The reference bounds are listed per dimension (e.g. ref0.1_0.1_0.1to0.9_0.9_0.9).
if stim_dims == 3:
    plane_tag = ''
elif plane_2D == 'Isoluminant plane':
    plane_tag = 'Isoluminantplane_'
else:
    plane_tag = f"{plane_2D.replace(' ', '')}{fixed_val:g}_"
sampling_tag = f"grid{num_grid_pts}" if fixed_ref else "sobol"
file_stem = f"SimTrialData_nonadaptive_{expt_dims}DExpt_{plane_tag}{colordiff_alg}_"+\
    f"dE{deltaE_1JND:g}_beta{beta:g}_{sampling_tag}_{nSims}total_jitter{jitter:g}_seed{rnd_seed}_"+\
    f"ref{'_'.join(map(str, ref_lb))}to{'_'.join(map(str, ref_ub))}"
print(f"Output file stem: {file_stem}")

#%%
# -----------------------------------------------------------
# 4. Simulate trials
# -----------------------------------------------------------
sim_trial = TrialPlacement_CIELab(config=stim_config)

# Weibull psychometric function: maps ΔE to p(correct)
sim_trial.setup_WeibullFunc(alpha=alpha,
                            beta=beta,
                            guessing_rate=guessing_rate,
                            deltaE_1JND=deltaE_1JND,
                            )
# Should equal target_pC
print(f"p(correct) at deltaE_1JND: {sim_trial.sim['pC_given_alpha_beta']:.4f}")

# Simulated observer: converts display RGB to CIELab (relative to the background)
# with the calib_date monitor primaries, and computes ΔE between two stimuli
sim_CIELab = SimThresCIELab(background_RGB, file_date=calib_date)

# Run the simulation (slow: one threshold search per trial)
sim_trial.run_sim(sim_CIELab)

#%%
# -----------------------------------------------------------
# 5. Visualize
# -----------------------------------------------------------
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize=8)

# Weibull psychometric function
sim_vis = TrialPlacementVisualization(sim_trial,
                                      settings=pltSettings_base,
                                      save_fig=False)
pltSettings_PMF = replace(PlotWeibullPMFSettings(), **pltSettings_base.__dict__)
pltSettings_PMF = replace(pltSettings_PMF, xticks=np.linspace(0, 9, 4))
x_PMF = np.linspace(0, 9, 100)
sim_vis.plot_WeibullPMF(x_PMF, settings=pltSettings_PMF)

# Reference and comparison stimuli, one row per trial: (nSims, 3)
xref = sim_trial.sim['ref'].T
x1 = sim_trial.sim['comp'].T

if stim_dims == 2:
    # Plot in W units. Isoluminant stimuli are already in W; RGB slices are
    # converted from normalized RGB (2x - 1). In both cases the fixed dimension
    # (a filler 1 for the isoluminant plane, 0 in W for an RGB slice) is dropped
    # by [:, vd] below.
    vd = sim_trial.config.varying_color_dim
    bounds = np.array([min(ref_lb), max(ref_ub)])
    xref_W, x1_W = xref, x1
    if plane_2D != 'Isoluminant plane':
        bounds = color_thres_data.N_unit_to_W_unit(bounds)
        xref_W = color_thres_data.N_unit_to_W_unit(xref)
        x1_W = color_thres_data.N_unit_to_W_unit(x1)

    pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
    pltSettings_tp = replace(pltSettings_tp,
                             linealpha=0.3,
                             bounds=bounds,
                             ref_markersize=10,
                             ticks=np.linspace(*bounds, 5),
                             flag_rescale_axes_label=plane_2D != 'Isoluminant plane',
                             ref_markeralpha=0.6,
                             comp_markeralpha=0.3,
                             )
    sampling_vis = SamplingRefCompPairVisualization(stim_dims,
                                                    color_thres_data,
                                                    settings=pltSettings_tp,
                                                    save_fig=True
                                                    )

    fig, ax = plt.subplots(1, 1, figsize=pltSettings_tp.fig_size, dpi=pltSettings_tp.dpi)
    sampling_vis.plot_sampling(xref_W[:, vd], x1_W[:, vd], ax=ax,
                               settings=pltSettings_tp)
    ax.set_title(f'{plane_2D} ({colordiff_alg})')
    fig.savefig(os.path.join(output_figDir, f'{file_stem}.pdf'), bbox_inches='tight')
    plt.show()
else:
    # 3D: interactive HTML of the ref-comp pairs in normalized RGB space
    pltSettings_html = replace(Plot3DSamplingHTMLSettings(),
                               ticks=np.linspace(min(ref_lb), max(ref_ub), 5),
                               lim=[0, 1], ell_alpha=0.4,
                               xlabel='R', ylabel='G', zlabel='B',
                               x1_opacity=0.8, line_opacity=0.8
                               )
    vis_sample_html = SamplingRefCompPairVisualization_html(settings=pltSettings_html)
    vis_sample_html.write_interactive_html(xref,
                                           x1,
                                           os.path.join(output_figDir, f"{file_stem}.html"),
                                           page_title=file_stem,
                                           )

#%%
# -----------------------------------------------------------
# 6. Save
# -----------------------------------------------------------
vars_dict = {'sim_trial': sim_trial,
             'color_thres_data': color_thres_data,
             'background_RGB': background_RGB,
             'sim_CIELab': sim_CIELab,
             }
with open(os.path.join(output_fileDir, f'{file_stem}.pkl'), 'wb') as f:
    pickled.dump(vars_dict, f)
