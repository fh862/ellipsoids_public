#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 15:09:10 2024

@author: fangfang
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Literal, Union
from copy import deepcopy
import os
from analysis.MOCS_thresholds import sim_MOCS_trials
from analysis.ellipses_tools import covMat_to_ellParamsQ, distance_to_ellipse_boundary, \
    compute_radii_scaler_to_reach_targetPC
from analysis.ellipsoids_tools import distance_to_ellipsoid_boundary

#%%
@dataclass
class StimConfig_CIELab_RGB:
    """
    Config for `TrialPlacement_CIELab` in normalized RGB space: an RGB slice
    (fixed_plane 'R', 'G' or 'B') or the RGB cube (fixed_plane '').

    sobol_lb / sobol_ub hold the reference bounds followed by the angle bounds:
    [dim1, dim2, theta] for a slice, [R, G, B, theta, phi] for the cube.
    With fixed_ref=True, references sit on a grid of num_grid_pts points per
    dimension spanning the reference bounds, and only the angles are Sobol-sampled.
    nSims is the total number of trials in both modes.
    """
    #Frozen
    flag_W_space: bool = field(init=False, default=False)
    # Changeable
    fixed_plane: Literal['R', 'G', 'B', ''] = 'R'
    gt: Literal ['CIE1976', 'CIE1994', 'CIE2000'] = 'CIE1994'
    fixed_val: float = 0.5
    nSims: int = 6000
    fixed_ref: bool = False      # True: references on a fixed grid; False: Sobol-sampled
    num_grid_pts: int = 5        # grid points per dimension (fixed_ref=True only)
    sobol_lb: list[float] = field(default_factory=lambda: [0.15, 0.15, 0.0])
    sobol_ub: list[float] = field(default_factory=lambda: [0.85, 0.85, 360.0])
    random_jitter: float = 0.3
    random_seed: Optional[int] = None

@dataclass
class StimConfig_CIELab_isoluminant:
    """
    Config for `TrialPlacement_CIELab` on the isoluminant plane, in 2D W space.

    sobol_lb / sobol_ub hold the reference bounds followed by the angle bound:
    [dim1, dim2, theta]. fixed_ref and num_grid_pts work as in StimConfig_CIELab_RGB.
    """
    # Frozen for isoluminant case (cannot be passed into __init__)
    fixed_plane: Literal['lum'] = field(init=False, default='lum')
    fixed_val: float = field(init=False, default=1.0)
    flag_W_space: bool = field(init=False, default=True)
    # Changeable
    gt: Literal['CIE1976', 'CIE1994', 'CIE2000'] = 'CIE1994'
    nSims: int = 6000
    fixed_ref: bool = False      # True: references on a fixed grid; False: Sobol-sampled
    num_grid_pts: int = 5        # grid points per dimension (fixed_ref=True only)
    random_jitter: float = 0.3
    sobol_lb: list[float] = field(default_factory=lambda: [-0.7, -0.7, 0.0])
    sobol_ub: list[float] = field(default_factory=lambda: [0.7, 0.7, 360.0])
    random_seed: Optional[int] = None
    M_RGBTo2DW: np.ndarray = field(default_factory=lambda: np.eye(3))
    M_2DWToRGB: np.ndarray = field(default_factory=lambda: np.eye(3))

StimConfig_CIELab = Union[StimConfig_CIELab_RGB, StimConfig_CIELab_isoluminant]

@dataclass
class StimConfig_W_sobolref:
    """
    Config for `TrialPlacement_sobolRef_W`, in W (model) space: the isoluminant
    plane (fixed_plane 'lum') or the 3D cube (fixed_plane '').

    sobol_lb / sobol_ub hold the reference bounds followed by the angle bounds:
    [dim1, dim2, theta] in 2D, [dim1, dim2, dim3, theta, phi] in 3D
    (e.g. [-0.7]*3 + [0, 0] to [0.7]*3 + [360, 180]). fixed_ref and num_grid_pts
    work as in StimConfig_CIELab_RGB, and nSims is the total number of trials.

    The class name is kept although it now also covers fixed-grid references:
    the experiments' pregenerated fallback-trial files (e.g. the ones
    run_6d_expt.py loads) store objects of this class, and unpickling them
    needs this exact name.
    """
    #Frozen
    flag_W_space: bool = field(init=False, default=True)
    # Changeable
    fixed_plane:Literal['lum', ''] = 'lum'
    fixed_val: Optional[float] = 1
    nSims: int = 6000
    fixed_ref: bool = False      # True: references on a fixed grid; False: Sobol-sampled
    num_grid_pts: int = 7        # grid points per dimension (fixed_ref=True only)
    random_jitter: float = 0.3
    sobol_lb: list[float] = field(default_factory=lambda: [-0.7, -0.7, 0.0])
    sobol_ub: list[float] = field(default_factory=lambda: [0.7, 0.7, 360.0])
    random_seed: Optional[int] = None

#%%
class NonAdaptiveTrialPlacement(ABC):
    """ 
    This class has common functionalities that can be shared by subclasses.
    The goal is to place trials near the thresholds given that we know what 
    the thresholds are. The ground truth thresholds can vary: 
        (1) CIELAB derived
        (2) Wishart model fits
    The references can be Sobol-sampled or fixed on a grid (config.fixed_ref).
    The stimulus space can vary:
        (i) GB, RG, RB planar slices or RGB cube
        (ii) Isoluminant plane

    Subclasses, by ground truth:
        TrialPlacement_CIELab      (CIELab ΔE)
        TrialPlacement_sobolRef_W  (Wishart model; the name predates fixed-grid
                                    references and is kept for saved pickles)
    """
    def __init__(self):
        #initialize the dictionary
        self.sim = {}
                 
    #%%
    def _unpack_config(self):
        """
        Parse the requested plane specification and populate `self.config` fields.
    
        Parameters
        ----------
        fixed_plane : str
            Plane selector:
            - 'R', 'G', 'B' : 2D slices of RGB cube where that channel is fixed
                * 'R' -> vary (G, B)  -> "GB plane"
                * 'G' -> vary (R, B)  -> "RB plane"
                * 'B' -> vary (R, G)  -> "RG plane"
            - 'lum' : isoluminant plane 
            - ''  : 3D case (no plane slicing)
    
        Side effects
        ------------
        Updates these fields on `self.config`:
            ndims, fixed_color_dim, varying_color_dim, stim_space

        """
        # One unified mapping (including 3D) keeps the control flow simple and prevents
        # partially-updated config states.
        space_mappings = {
            'R':  dict(ndims=2, fixed_color_dim=0, varying_color_dim=[1, 2],
                       stim_space='GB plane'),
            'G':  dict(ndims=2, fixed_color_dim=1, varying_color_dim=[0, 2],
                       stim_space='RB plane'),
            'B':  dict(ndims=2, fixed_color_dim=2, varying_color_dim=[0, 1],
                       stim_space='RG plane'),
            'lum': dict(ndims=2, fixed_color_dim=2, varying_color_dim=[0, 1],
                        stim_space='Isoluminant plane'),
            '': dict(ndims=3, fixed_color_dim=None, varying_color_dim=[0,1,2],
                       stim_space='cube'),
        }
    
        if self.config.fixed_plane not in space_mappings:
            allowed = ", ".join(space_mappings.keys())
            raise ValueError(f"Unknown fixed_plane={self.config.fixed_plane!r}. Allowed: {allowed}")
    
        spec = space_mappings[self.config.fixed_plane]
        for k, v in spec.items():
            setattr(self.config, k, v)
            
        # --- sanity checks ---
        # If using a fixed reference grid, the grid resolution must be specified.
        if self.config.fixed_ref and self.config.num_grid_pts is None:
            raise ValueError(
                "Config error: `fixed_ref=True` requires `num_grid_pts` to be set "
                "(number of grid points per axis for the reference grid)."
            )
            
        # If we are slicing the RGB cube (R/G/B fixed), we must know the fixed channel value.
        # (For example: fixed_plane='R' requires fixed_val to set the R coordinate.)
        if self.config.fixed_plane in {"R", "G", "B"} and self.config.fixed_val is None:
            raise ValueError(
                "Config error: `fixed_val` must be provided when `fixed_plane` is 'R', 'G', or 'B'."
            )
            
    def _validate_stim(self, stim):
        """
        Post-process a sampled stimulus so it matches the current plane definition
        and stays within valid bounds.
    
        What this does
        --------------
        1) If we are simulating a 2D slice, ensure the returned stimulus contains the
           fixed coordinate at `fixed_color_dim`:
             - If `stim` already has 3 rows (full RGB / full 3D vector), overwrite the
               fixed coordinate in-place.
             - If `stim` has only the varying coordinates (shape (2, ...) ), insert the
               fixed coordinate to form a full 3-vector.
    
           Note: for the isoluminant case, `fixed_val=1` is a filler dimension that makes
           later W->RGB conversion convenient.
    
        2) Clip values to the valid domain:
             - W/model space: [-1, 1]
             - RGB space:     [0, 1]
    
        Parameters
        ----------
        stim : array-like
            Either a full stimulus (shape (3, ...) ) or varying dimensions only
            (shape (2, ...) ) depending on caller.
    
        Returns
        -------
        stim_out : ndarray
            Validated stimulus with the expected dimensionality and bounds.
        """
        stim = np.asarray(stim)
    
        # Ensure the fixed coordinate is present/consistent for 2D-slice simulations.
        if self.config.ndims == 2 and self.config.fixed_color_dim is not None:
            fd= self.config.fixed_color_dim
            fv = self.config.fixed_val
    
            if stim.shape[0] == 3:
                # Full stimulus already: overwrite the fixed coordinate.
                stim[fd, ...] = fv
            elif stim.shape[0] == 2:
                # Only varying dims provided: insert the fixed coordinate.
                stim = np.insert(stim, fd, fv, axis=0)
            else:
                raise ValueError(
                    f"Expected stim with first dim 2 or 3 for ndims==2, got shape {stim.shape}"
                )
    
        # Clip to the appropriate bounds.
        lo, hi = (-1, 1) if self.config.flag_W_space else (0, 1)
        return np.clip(stim, lo, hi)
            
    #%% mandatory changes
    @abstractmethod
    def _initialize(self):
        """
        Initializes necessary simulation fields within `self.sim`.
    
        This method should:
        - Pre-allocate arrays for storing simulated comparison stimuli, 
          probability values, and response data.
        - Set up any additional parameters required for the simulation.
    
        Notes:
        ------
        - This method must be implemented in subclasses.
        """
        pass
            
    @abstractmethod
    def run_sim_1ref(self):
        """
        Simulates comparison stimuli for a single reference stimulus.
    
        This method should:
        - Generate a set of test stimuli based on a given reference.
        - Compute color differences and probabilities of correct responses.
        - Store the simulation results for the reference stimulus.
    
        Notes:
        ------
        - This method must be implemented in subclasses.
        """
        pass
    
    @abstractmethod
    def run_sim(self):
        """
        Runs the full simulation by generating comparison stimuli for multiple reference points.
    
        This method should:
        - Iterate over all reference stimuli.
        - Call `run_sim_1ref()` for each reference to generate comparison trials.
        - Aggregate the results across all references.
    
        Notes:
        ------
        - This method must be implemented in subclasses.
        """
        pass
    
#%%
class TrialPlacement_CIELab(NonAdaptiveTrialPlacement):
    def __init__(self, config: StimConfig_CIELab):
        """
        Non-adaptive trial placement with threshold distances defined by a CIE-based
        ΔE metric. References are either Sobol-sampled or fixed on a regular grid
        (`config.fixed_ref`); directions are always Sobol-sampled.

        High-level idea
        --------------
        Each simulated trial is generated as:

          1) Choose a reference location and a chromatic direction.
             - Sobol references (fixed_ref=False): sample (x, y, θ) for a 2D slice,
               (x, y, z, θ, φ) for the 3D volume, one Sobol point per trial.
             - Fixed-grid references (fixed_ref=True): the reference is a grid point
               (num_grid_pts per dimension; each gets an equal block of trials), and
               only the angles θ (2D) or θ, φ (3D) are Sobol-sampled.

          2) For the sampled (ref, direction), find the threshold comparison stimulus
             such that ΔE(ref, comp_thres) = deltaE_1JND (e.g., targeting 66.7% correct
             under the fitted psychometric function / task design).

          3) Add Gaussian jitter around the threshold comparison stimulus.
             The jitter magnitude scales with the step length ||comp_thres - ref||:
                 jitter ~ N(0, (||Δ|| * random_jitter))

          4) Compute ΔE(ref, comp) and generate a binary response using a Weibull
             psychometric function.

        Notes
        -----
        - This is “non-adaptive” because trial placement does not depend on prior responses.
        - Stimuli can be represented either in RGB space ([0, 1]) or in a model/W space
          (bounded in [-1, 1]) if the stimulus lives on an isoluminant plane, depending on
          `config.flag_W_space` and `config.stim_space`.
        - The mapping to RGB is handled by `_convert_stim_toRGB()` before ΔE is computed.
        - Randomness: the single `config.random_seed` seeds three separate generators:
            * scipy's scrambled Sobol sampler (references and direction angles);
            * NumPy's global stream (`np.random.seed` at the start of `run_sim`), which is
              shared, in trial order, by the threshold search's random starting lengths
              (`SimThresCIELab.find_vecLen`), the jitter, and the Bernoulli responses;
            * in `TrialPlacement_sobolRef_W`, a local `default_rng` for the covariance
              scaler (`compute_radii_scaler_to_reach_targetPC`).
          The W model's p(correct) uses the fit's own JAX key (`opt_key`), not this seed.
          Because the global stream is shared, adding or removing a draw in any of these
          steps changes all later draws: one step's randomness can't be varied while
          holding the others fixed. Separate generators per step (e.g. via
          `np.random.SeedSequence(random_seed).spawn(k)`) would decouple them and may be
          preferable, but would change existing outputs and was not worth the rework.

        Parameters
        ----------
        config : StimConfig_CIELab
            Stimulus configuration describing:
              - ndims (2 or 3), stim_space (e.g., 'Isoluminant plane' or RGB slices or RGB cube)
              - Sobol bounds (sobol_lb, sobol_ub), random_seed, nSims (total trials)
              - reference sampling (fixed_ref, num_grid_pts)
              - plane/slice bookkeeping (varying_color_dim, fixed_color_dim)
              - jitter (random_jitter)
              - ΔE metric choice (gt) and any required transforms (e.g., M_2DWToRGB)

        """
        super().__init__()
        self.config = config
        self._unpack_config()
        self._initialize()
        
    def _initialize(self):
        """
        Updates `self.sim`, a dictionary of arrays storing per-trial quantities.
        Arrays are initialized with NaNs for easy debugging and vectorized writing.

        Note
        ----
        The Weibull parameters (alpha, beta, guessing_rate) and deltaE_1JND are updated
        separately via `setup_WeibullFunc()`.
        
        """
        
        # Determine the base shape based on 
        nSims = self.config.nSims
                
        # Common arrays for all cases
        self.sim.update({
            'ref': np.full((3, nSims), np.nan),       
            'comp': np.full((3, nSims), np.nan),
            'vecDir_unit': np.full((3, nSims), np.nan),
            'probC': np.full((nSims,), np.nan),
            'resp_binary': np.full((nSims,), np.nan),
            'deltaE': np.full((nSims,), np.nan)
        })
                
    def _convert_stim_toRGB(self, stim):
        """
        `stim` is a single stimulus vector (shape: (3,) or (3, N)).
        It may already be in RGB space, or it may be in a "Wishart/model" space
        (depending on `self.config.flag_W_space` and dimensionality).
        
        """
        if not self.config.flag_W_space:
            return stim                                   # already normalized RGB in [0, 1]
        if self.config.ndims == 2 and self.config.stim_space == 'Isoluminant plane':
            return self.config.M_2DWToRGB @ stim          # 2D W (+ filler 1) -> RGB
        if self.config.ndims == 3:
            return (stim + 1) / 2                         # 3D W in [-1, 1] -> RGB in [0, 1]
        raise ValueError(
            "flag_W_space=True but unsupported config: "
            f"ndims={self.config.ndims}, stim_space={self.config.stim_space!r}"
        )
        
    def _sobol_generate_ref(self):
        """
        Generate each trial's reference location and chromatic direction.

        Sobol references (config.fixed_ref = False): `sample_sobol` returns one
        quasi-random point per trial, shape (nSims, D):
          - 2D: [x, y, theta]          -> D = 3
          - 3D: [x, y, z, theta, phi]  -> D = 5

        Fixed-grid references (config.fixed_ref = True): the references are the
        points of a regular grid with `num_grid_pts` values per dimension, spanning
        the reference part of the Sobol bounds. Each grid point gets an equal,
        contiguous block of trials, so nSims must be divisible by the number of grid
        points. Only the angles are Sobol-sampled ([theta] in 2D, [theta, phi] in 3D),
        as one sequence over all trials; each reference's directions are a
        contiguous run of it.

        We then:
          1) convert the angles into unit direction vectors `vecDir_unit` (cos/sin, so
             already unit length; no separate normalization step),
          2) validate/clip reference coordinates to the valid domain (W-space or RGB-space).
        """
        nd = self.config.ndims       # number of reference coordinates (2 or 3)
        nSims = self.config.nSims
        lb, ub = self.config.sobol_lb, self.config.sobol_ub

        if self.config.fixed_ref:
            # Checked here, before the slow per-trial loop in run_sim
            n_refs = self.config.num_grid_pts ** nd
            if nSims % n_refs != 0:
                raise ValueError(
                    f"nSims ({nSims}) must be divisible by the number of grid "
                    f"references ({n_refs} = {self.config.num_grid_pts}^{nd}) so that "
                    "every reference gets the same number of trials."
                )
            # Grid points (n_refs, nd), each repeated for a contiguous block of trials: (nd, nSims)
            axes = [np.linspace(lb[d], ub[d], self.config.num_grid_pts) for d in range(nd)]
            grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, nd)
            ref = np.repeat(grid, nSims // n_refs, axis=0).T
            lb, ub = lb[nd:], ub[nd:]                     # Sobol over the angles only

        samples = sim_MOCS_trials.sample_sobol(nSims, lb, ub, force_center=False,
                                               seed=self.config.random_seed)
        if not self.config.fixed_ref:
            # Split into reference coordinates (nd, nSims) and angles (nSims, nd - 1)
            ref, samples = samples[:, :nd].T, samples[:, nd:]
        theta = np.radians(samples[:, 0])

        vd = self.config.varying_color_dim
        if nd == 2:
            # Unit directions in the plane; no motion along the fixed dimension
            self.sim['vecDir_unit'][vd] = np.stack((np.cos(theta), np.sin(theta)))
            self.sim['vecDir_unit'][self.config.fixed_color_dim] = 0
        else:
            # theta: azimuth in the x-y plane; phi: inclination from +z
            phi = np.radians(samples[:, 1])
            self.sim['vecDir_unit'] = np.stack((np.cos(theta) * np.sin(phi),
                                                np.sin(theta) * np.sin(phi),
                                                np.cos(phi)))

        # Store validated references (clipped to the appropriate bounds depending on space).
        self.sim['ref'] = self._validate_stim(ref)
        
    def _jitter(self, ref, comp):
        """
        Compute additive jitter for (ref, comp).
        
        Jitter model
        ------------
        For each trial, let Δ = comp - ref and step = ||Δ||.
        We add i.i.d. Gaussian noise per coordinate:
        
            jitter ~ N(0, (step * random_jitter))
        
        This makes jitter scale with the threshold distance (larger steps get more jitter).

    
        Accepts either:
          - single pair: ref, comp shape (3,)
          - batched pairs: ref, comp shape (3, N)

        """
        ref, comp = np.asarray(ref), np.asarray(comp)
        if ref.ndim not in (1, 2) or ref.shape[0] != 3:
            raise ValueError(f"`ref` and `comp` must be shape (3,) or (3, N); got {ref.shape}.")

        # SD = random_jitter × ref-to-comp step length (one trial, or one per column)
        if ref.ndim == 1:
            jitter = np.random.randn(3) * np.linalg.norm(comp - ref) * self.config.random_jitter
        else:
            step = np.linalg.norm(comp - ref, axis=0)                       # (N,)
            jitter = np.random.randn(*ref.shape) * (step * self.config.random_jitter)

        if self.config.fixed_color_dim is not None:
            jitter[self.config.fixed_color_dim] = 0
        return jitter
    
    def run_sim_1ref(self, sim_CIELab, ref, comp):  
        """
        Simulate a single binary response for one (ref, comp) trial.
    
        Steps:
          1) Convert stimuli to RGB if they are represented in W/model space.
          2) Compute perceptual distance ΔE(ref, comp) using the selected CIE method.
          3) Map ΔE -> p(correct) via the Weibull psychometric function.
          4) Draw a Bernoulli sample to produce a binary response.
    
        Parameters
        ----------
        sim_CIELab : SimThresCIELab
            Helper used to compute ΔE in CIELab / CIE94 / CIEDE2000.
        ref : array-like, shape (3,)
            Reference stimulus (RGB or W/model space, depending on config).
        comp : array-like, shape (3,)
            Comparison stimulus (RGB or W/model space, depending on config).
    
        Returns
        -------
        deltaE : float
            Perceptual distance between ref and comp.
        probC : float
            Predicted probability of a correct response.
        resp_binary : int
            Simulated response (1 = correct, 0 = incorrect).
        """
        
        # Convert the reference RGB values to Lab color space.
        rgb_ref = self._convert_stim_toRGB(ref)
        rgb_comp = self._convert_stim_toRGB(comp)
            
        # Compute perceptual distance using the configured CIE ΔE variant
        deltaE = sim_CIELab.compute_deltaE(rgb_ref, None,None,
                                           comp_RGB=rgb_comp, 
                                           method=self.config.gt
                                           )
        
        # Psychometric function: ΔE -> p(correct)
        probC = self.WeibullFunc(deltaE,
                                 self.sim['alpha'], 
                                 self.sim['beta'], 
                                 self.sim['guessing_rate']
                                 )
        
        # Bernoulli draw for a binary response
        randNum = np.random.rand() 
        resp_binary = (randNum < probC).astype(int)
            
        return deltaE, probC, resp_binary

    def run_sim(self, sim_CIELab):
        """
        Simulate trials with Sobol-sampled or fixed-grid references (config.fixed_ref).

        For each simulated trial:
          1) Choose a reference and a direction (see `_sobol_generate_ref`).
          2) Define a threshold comparison stimulus in the chosen direction such that
             ΔE(ref, comp_thres) = deltaE_1JND (using the configured CIE metric).
          3) Add Gaussian jitter around that threshold point.
          4) Convert to RGB if needed, compute ΔE, map through psychometric function,
             and sample a binary response.
    
        Parameters
        ----------
        sim_CIELab : object
            Helper that computes ΔE and (for isoluminant plane) can solve for the
            threshold point along a direction.

        """
        # Seed NumPy's global stream (None = fresh entropy). Threshold-search starts, jitter,
        # and responses all draw from it in trial order (see the class docstring).
        np.random.seed(self.config.random_seed)
        
        # References and unit directions for every trial
        self._sobol_generate_ref()
        vd = self.config.varying_color_dim
        deltaE_1JND = self.sim['deltaE_1JND']
    
        for i in range(self.config.nSims):
            ref_i, dir_i = self.sim['ref'][:, i], self.sim['vecDir_unit'][:, i]
            
            # 1) Threshold comparison along the direction (no jitter yet)
            if self.config.stim_space == 'Isoluminant plane':
                # Solve in 2D W space (varying dims only); the last output is the
                # threshold point in W
                *_, stim_thres = sim_CIELab.find_threshold_point_on_isoluminant_plane(
                    ref_i[vd], dir_i[vd], self.config.M_RGBTo2DW, self.config.M_2DWToRGB,
                    deltaE_1JND, coloralg=self.config.gt)
            else:
                # RGB: solve for the step length that reaches deltaE_1JND
                opt_vecLen = sim_CIELab.find_vecLen(ref_i, dir_i, deltaE_1JND,
                                                    coloralg=self.config.gt)
                stim_thres = opt_vecLen * dir_i + ref_i
                
            # 2) Full 3-vector (fixed dim inserted), clipped to bounds
            stim_thres = self._validate_stim(stim_thres)
            
            # 3) Jitter around the threshold point
            self.sim['comp'][:, i] = self._validate_stim(stim_thres + self._jitter(ref_i, stim_thres))
            
            # 4) ΔE, p(correct), and a binary response
            self.sim['deltaE'][i], self.sim['probC'][i], self.sim['resp_binary'][i] = \
                self.run_sim_1ref(sim_CIELab, ref_i, self.sim['comp'][:, i])
                
    def setup_WeibullFunc(self, alpha, beta, guessing_rate, deltaE_1JND):
        """
        Sets up the parameters for the Weibull psychometric function and calculates
        the probability of correct response for a given deltaE value.

        Parameters:
        - alpha (float): Scale parameter of the Weibull function, controlling the threshold.
        - beta (float): Shape parameter, controlling the slope.
        - guessing_rate (float): The probability of a correct guess by chance.
        """
        # Validate input parameters
        if alpha <= 0:
            raise ValueError("Alpha must be positive.")
        if beta <= 0:
            raise ValueError("Beta must be positive.")
        if not (0 <= guessing_rate <= 1):
            raise ValueError("Guessing rate must be between 0 and 1.")
    
        # Define parameters for the psychometric function used in the simulation.
        self.sim.update({
            "alpha": alpha,
            "beta": beta,
            "guessing_rate": guessing_rate,
            "deltaE_1JND": deltaE_1JND,
        })
    
        # Calculate the probability of correct response given alpha and beta.
        self.sim['pC_given_alpha_beta'] = self.WeibullFunc(deltaE_1JND,
                                                           alpha, 
                                                           beta, 
                                                           guessing_rate
                                                           )   
                
    @staticmethod
    def WeibullFunc(x, alpha, beta, guessing_rate):
        """
        Computes the Weibull psychometric function, giving the probability of a 
        correct response.
        
        Parameters:
        - x (float or array-like): The stimulus intensity (e.g., deltaE).
        - alpha (float): Scale parameter of the Weibull function.
        - beta (float): Shape parameter of the Weibull function.
        - guessing_rate (float): The probability of a correct guess by chance.
        
        Returns:
        - pCorrect (float or array-like): The probability of a correct response.
        
        """
        pCorrect = (1 - (1-guessing_rate)*np.exp(- (x/alpha)** beta))
        return pCorrect
                   
#%% 
class TrialPlacement_sobolRef_W(TrialPlacement_CIELab):
    def __init__(self, gt_Wishart, config: StimConfig_W_sobolref):
        """
        Non-adaptive trial placement using *Wishart-model* predicted thresholds.

        This subclass shares the reference and direction sampling of
        `TrialPlacement_CIELab` (Sobol-sampled or fixed-grid references, set by
        `config.fixed_ref`), but uses a fitted Wishart model instead of a ΔE metric:
        the model sets the threshold distance along each direction (see `run_sim`)
        and predicts p(correct) for the simulated responses.

        The class name predates fixed-grid references. It is kept because the
        experiments' pregenerated fallback-trial files store objects of this class.

        Parameters
        ----------
        gt_Wishart : dict
            Container holding Wishart-model outputs (a fit file's saved variables).
            Expected keys include:
              - 'model_pred_Wishart' : fitted/predictive model object

        config : StimConfig_W_sobolref
            Simulation configuration (ndims, nSims, fixed_ref, num_grid_pts, jitter,
            bounds, etc.).

        """
        super().__init__(config=config)
        self._unpack_gt(gt_Wishart)
    
    def _unpack_gt(self, gt_Wishart):
        """
        Load the Wishart-model “ground truth” prediction object from a serialized dict.
        
        The `gt_Wishart` container is expected to store a fitted/predictive Wishart model
        (typically produced by an offline fitting script). 
        
        Parameters
        ----------
        gt_Wishart : dict
            Dictionary containing a saved Wishart prediction object under the key
            'model_pred_Wishart'.
          
        """
        #deep-copied prediction object (contains model + fitted weights)
        self.gt_model_pred = deepcopy(gt_Wishart['model_pred_Wishart'])
        #underlying WishartProcessModel instance
        self.model = self.gt_model_pred.model
        #fitted weight tensor used to generate predictions
        self.W_est = self.gt_model_pred.W_est
            
    def _initialize(self):    
        """
        Preallocate simulation buffers (NaN-filled) for `nSims` trials.
        
        This class operates in the model/W space but stores stimuli and directions as
        3-vectors for a consistent interface with RGB conversion. In 2D-slice settings,
        we use a homogeneous-like convention:
          - ref/comp: last row is set to 1 so a 3×3 transform (e.g., W→RGB) can be applied.
          - vecDir_unit: last row is fixed at 0 so directions lie in the 2D plane.
        
        Shapes
        ------
          - ref, comp, vecDir_unit : (3, nSims)
          - probC, resp_binary     : (nSims,)

        Same arrays as `TrialPlacement_CIELab`, minus `deltaE`: the Wishart model
        predicts p(correct) directly, so there is no ΔE to store.
        """
        super()._initialize()
        self.sim.pop('deltaE')

    def _find_comp_thres(self):
        """
        Find comparison stimuli that land closest to the model's target threshold.
        
        For each simulated reference (nSims, 3), we:
          1) sweep a grid of candidate step lengths `vecLength_grid` along a unit direction
             `vecDir_unit`,
          2) evaluate the model-predicted percent-correct for each candidate,
          3) choose the candidate whose p(correct) is closest to `self.gt_model_pred.target_pC`.
        
        Returns
        -------
        comp_thres : ndarray, shape (3, nSims)
        
        Note: We broadcast the reference and comparison stimuli across nRep so that 
        all candidate (ref, comp) pairs can be evaluated in a single call to
        gt_model_pred._compute_pChoosingX1. This vectorized evaluation takes advantage 
        of JAX’s strengths (batching and JIT compilation) and is substantially faster
        than looping over references or step lengths in Python.
        
        """
        vd = self.config.varying_color_dim
        ref, dirs = self.sim["ref"], self.sim["vecDir_unit"]       # (3, nSims) each

        # Candidate step lengths along each direction: (nRep,)
        vecLength_grid = np.asarray(self.gt_model_pred._set_up_grid_search(), dtype=float)

        # Candidate comparisons in the varying dims, every step length × every trial:
        # (nRep, 1, 1) * (nSims, |vd|) broadcasts to (nRep, nSims, |vd|)
        comp_cand = ref[vd].T + vecLength_grid[:, None, None] * dirs[vd].T
        ref_cand = np.broadcast_to(ref[vd].T, comp_cand.shape)

        # Model-predicted p(correct) for every candidate pair, back to (nRep, nSims)
        probC = self.gt_model_pred._compute_pChoosingX1(ref_cand.reshape(-1, len(vd)),
                                                         comp_cand.reshape(-1, len(vd)))
        probC = np.reshape(probC, (vecLength_grid.size, -1))

        # Per trial, the step length whose p(correct) is closest to target_pC
        idx = np.abs(probC - self.gt_model_pred.target_pC).argmin(axis=0)   # (nSims,)

        # Threshold comparisons in the full stored representation: (3, nSims).
        # The fixed (filler) dim keeps its value because its direction component is 0.
        return ref + dirs * vecLength_grid[idx]
    
    def _find_comp_thres_approx(self):
        """
        Approximate threshold-level comparison stimuli via the local noise ellipse.
        
        This is a fast surrogate for the `_find_comp_thres` approach (i.e., the method 
        that explicitly searches along each chromatic direction until the model reaches 
        the target probability). Instead, we approximate the isothreshold contour by:
          - computing the model-predicted local covariance Σ(ref),
          - converting Σ into ellipse / ellipsoid parameters 
          - scaling the ellipse / ellipsoid axis length by a precomputed factor 
              (`self.Sigmas_scaler`) that roughly maps “1 unit Gaussian separation” to 
              the desired performance level,
          - intersecting the (scaled) ellipse with the sampled unit direction vector.
        
        Intuition
        ---------
        The exact threshold contour is not guaranteed to be a perfect scaled covariance
        contour, but in practice it can be a close approximation and is much faster.
        
        Returns
        -------
        comp_thres : ndarray, shape (ndims, nSims)
        
        """
        vd = np.asarray(self.config.varying_color_dim, dtype=int)
        nSims = int(self.config.nSims)
        nd = self.config.ndims
        
        # References in the varying subspace, shape: (len(vd), nSims)
        ref_v = np.asarray(self.sim["ref"][vd], dtype=float)
        
        # 1) Compute local noise covariances Σ(ref) from the fitted model.
        #    Many APIs take points as (nSims, D), hence the transpose.
        U = self.model.compute_U(self.W_est, ref_v.T)    # expected: (nSims, ndims, ndims + 1)
        Sigmas = self.model.compute_Sigmas(U)            # expected: (nSims, ndims, ndims)
        
        # Distance to the ellipse (2D) or ellipsoid (3D) boundary along a direction.
        # covMat_to_ellParamsQ returns the rotation angles these expect: theta in 2D,
        # theta and phi in 3D.
        distance_to_boundary = distance_to_ellipse_boundary if nd == 2 \
            else distance_to_ellipsoid_boundary

        # Step vectors (ref -> threshold) in the varying subspace
        vecLength_thres_approx = np.full((nd, nSims), np.nan)

        # 2-4) For each trial:
        #   2) Σ -> axis lengths and rotation angle(s), with the axes enlarged by
        #      `Sigmas_scaler` to approximate the threshold contour
        #   3) distance r from the reference to that boundary along the trial's direction
        #   4) step = r * direction
        for n in range(nSims):
            _, _, axisl, *angles = covMat_to_ellParamsQ(Sigmas[n])
            dir_n = self.sim["vecDir_unit"][vd, n]
            r = distance_to_boundary(*(axisl * self.Sigmas_scaler), *angles, *dir_n)
            vecLength_thres_approx[:, n] = r * dir_n
        
        # Store intermediates for debugging / inspection
        self.sim.update({
            "U": U,
            "Sigmas": Sigmas,
            "vecLength_thres_approx": vecLength_thres_approx,
        })
        
        # Approximate threshold comparisons in varying subspace
        comp_thres = ref_v + vecLength_thres_approx
        return comp_thres
    
    def run_sim_1ref(self, ref, comp):
        """
        Simulates a single trial of an oddity task using the Wishart model to predict 
        the probability of correctly identifying the odd stimulus. 
        """
        
        # Compute the probability of correctly identifying the odd stimulus 
        # using the signed difference.
        probC = self.gt_model_pred._compute_pChoosingX1(ref, comp)
        
        # Generate a random response based on the predicted probability 
        randNum = np.random.rand(probC.shape[0]) 
        resp_binary = (randNum < probC).astype(int)
        
        return probC, resp_binary    

    def run_sim(self, flag_approx = True):
        """
        Simulate non-adaptive trials with Sobol-sampled or fixed-grid references
        (config.fixed_ref) and Sobol-sampled directions.

        Pipeline
        --------
        1) Choose reference locations and direction angles (see `_sobol_generate_ref`),
           then convert angles into unit direction vectors (`vecDir_unit`).
        2) (No separate normalization: the vectors are unit length by construction.)
        3) For each (ref, direction), find the threshold comparison stimulus:
             - Approx mode (`flag_approx=True`): approximate the threshold contour using the
               local noise covariance (Σ) from the Wishart fit, scaled by `Sigmas_scaler`.
             - Exact mode (`flag_approx=False`): grid-search step length along the direction
               and pick the point whose model-predicted p(correct) is closest to target.
        4) Validate/standardize the threshold stimulus representation (e.g., insert fixed/filler
           coordinates, clip to bounds).
        5) Add Gaussian jitter around the threshold point (scaled by the ref→threshold distance).
        6) Store the final comparison stimuli in `self.sim['comp']`.
        
        """
        # Seed NumPy's global stream (shared by jitter and responses; see the
        # TrialPlacement_CIELab docstring for how random_seed is used)
        np.random.seed(self.config.random_seed)
        
        # 1) Sobol sample reference locations + direction angles, then build `self.sim['vecDir_unit']`.
        self._sobol_generate_ref()
        
        # 3) Find the threshold comparison stimulus along each sampled direction.
        if flag_approx:
            # Approximate thresholding:
            #   - Compute a scalar that maps a *unit-noise* Gaussian ellipse to the target
            #     performance (e.g., 66.7% correct for 3AFC oddity).
            #   - Then enlarge the local covariance ellipse by this scalar and intersect it
            #     with each direction to get an approximate threshold point.
            #
            # Internally, `compute_radii_scaler_to_reach_targetPC` typically:
            #   * draws samples from unit Gaussians for (xref, x0, x1),
            #   * increases the separation between xref and x1,
            #   * evaluates the observer model’s p(correct),
            #   * returns the separation needed to hit `target_pC`.
            # Seeded so that the same random_seed reproduces the same trials
            self.Sigmas_scaler, *_ = compute_radii_scaler_to_reach_targetPC(
                self.gt_model_pred.target_pC, ndims = self.config.ndims,
                seed = self.config.random_seed
            )
            stim_thres = self._find_comp_thres_approx()
        else:
            # Exact thresholding:
            # grid-search step lengths along each direction and pick the candidate whose
            # model-predicted p(correct) is closest to `target_pC`.
            stim_thres = self._find_comp_thres()
        
        # 4) Standardize threshold stimuli into the full stimulus representation (e.g., insert
        # fixed/filler coordinates for 2D slices) and clip to valid bounds.
        stim_thres_v = self._validate_stim(stim_thres)
        
        # 5) Add Gaussian jitter around each threshold point (in the same space as ref/comp).
        # `_jitter` supports vectorized inputs: ref/comp can be (3,) or (3, nSims).
        jitter = self._jitter(self.sim["ref"], stim_thres_v)
        
        # 6) Final comparisons for each simulated trial.
        self.sim["comp"] = self._validate_stim(stim_thres_v + jitter)
