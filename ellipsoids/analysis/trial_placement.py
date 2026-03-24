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
import sys
import os
script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from analysis.MOCS_thresholds import sim_MOCS_trials
from analysis.ellipses_tools import stretch_unit_circle, rotate_relocate_stretched_ellipse,\
    covMat_to_ellParamsQ, distance_to_ellipse_boundary, compute_radii_scaler_to_reach_targetPC
from analysis.ellipsoids_tools import distance_to_ellipsoid_boundary

#%%
@dataclass
class StimConfig_RGBslices:
    # Frozen
    fixed_ref: bool = field(init=False, default=True)
    flag_W_space: bool = field(init=False, default=False)
    # Changeable
    fixed_plane: Literal['R', 'G', 'B', 'lum'] = 'R'
    gt: Literal ['CIE1976', 'CIE1994', 'CIE2000'] = 'CIE1994'
    fixed_val: float = 0.5
    num_grid_pts: int = 5
    nSims: int = 240
    random_jitter: float = 0.3
    random_seed: Optional[int] = None
    
@dataclass
class StimConfig_isoluminant:
    # Frozen for isoluminant case (cannot be passed into __init__)
    fixed_plane: Literal['lum'] = field(init=False, default='lum')
    fixed_val: float = field(init=False, default=1.0)
    flag_W_space: bool = field(init=False, default=True)
    fixed_ref: bool = field(init=False, default=True)
    # Changeable
    gt: Literal['CIE1976', 'CIE1994', 'CIE2000'] = 'CIE1994'
    fixed_ref: bool = True
    num_grid_pts: int = 5
    nSims: int = 240
    random_jitter: float = 0.3
    random_seed: Optional[int] = None
    M_RGBTo2DW: np.ndarray = field(default_factory=lambda: np.eye(3))
    M_2DWToRGB: np.ndarray = field(default_factory=lambda: np.eye(3))
    
StimConfig = Union[StimConfig_RGBslices, StimConfig_isoluminant]

@dataclass
class StimConfig_RGBslices_sobolref:
    #Frozen
    fixed_ref: bool = field(init=False, default=False)
    flag_W_space: bool = field(init=False, default=False)
    # Changeable
    fixed_plane: Literal['R', 'G', 'B', 'lum'] = 'R'
    gt: Literal ['CIE1976', 'CIE1994', 'CIE2000'] = 'CIE1994'
    fixed_val: float = 0.5
    nSims: int = 6000
    sobol_lb: list[float] = field(default_factory=lambda: [0.15, 0.15, 0.0]) 
    sobol_ub: list[float] = field(default_factory=lambda: [0.85, 0.85, 360.0])
    random_jitter: float = 0.3
    random_seed: Optional[int] = None

@dataclass
class StimConfig_isoluminant_sobolref:
    # Frozen for isoluminant case (cannot be passed into __init__)
    fixed_plane: Literal['lum'] = field(init=False, default='lum')
    fixed_val: float = field(init=False, default=1.0)
    flag_W_space: bool = field(init=False, default=True)
    fixed_ref: bool = field(init=False, default=False)
    # Changeable
    gt: Literal['CIE1976', 'CIE1994', 'CIE2000'] = 'CIE1994'
    nSims: int = 6000
    random_jitter: float = 0.3
    sobol_lb: list[float] = field(default_factory=lambda: [-0.7, -0.7, 0.0])
    sobol_ub: list[float] = field(default_factory=lambda: [0.7, 0.7, 360.0])
    #for 3D it would be
    #[-0.7, -0.7, -0.7, 0.0, 0.0]
    #[0.7, 0.7, 0.7, 360.0, 180.0]
    random_seed: Optional[int] = None
    M_RGBTo2DW: np.ndarray = field(default_factory=lambda: np.eye(3))
    M_2DWToRGB: np.ndarray = field(default_factory=lambda: np.eye(3))

StimConfig_sobolref = Union[StimConfig_RGBslices_sobolref, StimConfig_isoluminant_sobolref]

@dataclass
class StimConfig_W:
    #Frozen
    flag_W_space: bool = field(init=False, default=True)
    fixed_ref: bool = field(init=False, default=True)
    
    # Changeable
    fixed_plane:Literal['lum', ''] = 'lum'
    fixed_val: Optional[float] = 1
    num_grid_pts: int = 7
    nSims: int = 240
    random_jitter: float = 0.3
    random_seed: Optional[int] = None
    
@dataclass
class StimConfig_W_sobolref:
    #Frozen
    flag_W_space: bool = field(init=False, default=True)
    fixed_ref: bool = field(init=False, default=False)    
    # Changeable
    fixed_plane:Literal['lum', ''] = 'lum'
    fixed_val: Optional[float] = 1
    nSims: int = 6000
    random_jitter: float = 0.3
    sobol_lb: list[float] = field(default_factory=lambda: [-0.7, -0.7, 0.0]) 
    sobol_ub: list[float] = field(default_factory=lambda: [0.7, 0.7, 360.0])
    #for 3D it would be
    #[-0.7, -0.7, -0.7, 0.0, 0.0]
    #[0.7, 0.7, 0.7, 360.0, 180.0]
    random_seed: Optional[int] = None

#%%
class NonAdaptiveTrialPlacement(ABC):
    """ 
    This class has common functionalities that can be shared by subclasses.
    The goal is to place trials near the thresholds given that we know what 
    the thresholds are. The ground truth thresholds can vary: 
        (1) CIELAB derived
        (2) Wishart fits to CIELAB derived thresholds
    The sampling method can vary:
        (a) fixed reference stimuli
        (b) Sobol-generated reference stimuli
    The stimulus space can vary:
        (i) GB, RG, RB planar slices or RGB cube
        (ii) Isoluminant plane
        
    Depending on the combinations, we choose different subclasses:
        TrialPlacement_sobolRef
        TrialPlacement_gridRef
        TrialPlacement_sobolRef_W
        TrialPlacement_gridRef_W
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
class TrialPlacement_sobolRef(NonAdaptiveTrialPlacement):
    def __init__(self, config: StimConfig_sobolref):
        """
        Non-adaptive trial placement using Sobol-sampled references and directions,
        with threshold distances defined by a CIE-based ΔE metric.

        High-level idea
        --------------
        Each simulated trial is generated as:

          1) Sample a reference location and a chromatic direction using a Sobol sequence.
             - 2D slice: sample (x, y, θ)
             - 3D volume: sample (x, y, z, θ, φ)

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

        Parameters
        ----------
        config : StimConfig_sobolref
            Stimulus configuration describing:
              - ndims (2 or 3), stim_space (e.g., 'Isoluminant plane' or RGB slices or RGB cube)
              - Sobol bounds (sobol_lb, sobol_ub), random_seed, nSims
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
            'vecDir': np.full((3, nSims), np.nan),
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
        # If we are operating in W-space, we need to map the stimulus into RGB space.
        if self.config.flag_W_space:
    
            # 2D case: `stim` is expected to be a 3-vector where one entry may be a
            # filler constant (e.g., 1) inserted upstream so that we can apply a 3x3
            # transform from 2D-W representation into RGB.
            if self.config.ndims == 2 and self.config.stim_space == 'Isoluminant plane':
                # Apply the linear transform (W -> RGB).
                rgb = self.config.M_2DWToRGB @ stim
    
            elif self.config.ndims == 3:
                # 3D case: `stim` is assumed to be in 3D W-space bounded in [-1, 1].
                # Convert to normalized RGB coordinates in [0, 1] via an affine mapping.
                rgb = (stim + 1) / 2
            else:
                raise ValueError(
                    "flag_W_space=True but unsupported config: "
                    f"ndims={self.config.ndims}, stim_space={self.config.stim_space!r}"
                )
    
        else:
            # If we are not in W-space, `stim` is already interpreted as RGB (normalized [0, 1]),
            # so no conversion is needed.
            rgb = stim
    
        # Return the stimulus in RGB coordinates.
        return rgb
        
    def _sobol_generate_ref(self):
        """
        Sample (reference location, direction angles) using a Sobol sequence.
        
        `sample_sobol` returns a quasi-random design matrix with shape (nSims, D),
        where D depends on dimensionality:
          - 2D: [x, y, theta]          -> D = 3
          - 3D: [x, y, z, theta, phi]  -> D = 5
        
        We then:
          1) split the Sobol samples into reference coordinates and direction angles,
          2) convert angles into a unit direction vector `vecDir`,
          3) validate/clip reference coordinates to the valid domain (W-space or RGB-space).
        """
        sobolRef_cat = sim_MOCS_trials.sample_sobol(self.config.nSims,
                                                    self.config.sobol_lb,
                                                    self.config.sobol_ub,
                                                    force_center=False,
                                                    seed=self.config.random_seed,
                                                    )
    
        # Indices of varying vs fixed dimensions for 2D slice experiments.
        # For 3D, `fixed_color_dim` is typically None and `varying_color_dim` spans all dims.
        vd = self.config.varying_color_dim
        fd = self.config.fixed_color_dim
    
        if self.config.ndims == 2:
            # 2D case:
            #   - ref coordinates: sobolRef_cat[:, :2] -> transpose to shape (2, nSims)
            #   - direction angle: sobolRef_cat[:,  2] in degrees
            ref = sobolRef_cat[:, :-1].T
            sobolAngle = sobolRef_cat[:, -1]
    
            # Build unit direction vectors in the 2D subspace.
            # column_stack -> (nSims, 2), transpose -> (2, nSims)
            self.sim['vecDir'][vd] = np.column_stack(
                (np.cos(np.radians(sobolAngle)), np.sin(np.radians(sobolAngle)))
            ).T
    
            # Ensure no motion along the fixed dimension in 2D RGB-slice settings.
            self.sim['vecDir'][fd] = 0
    
        else:
            # 3D case:
            #   - ref coordinates: sobolRef_cat[:, :3] -> transpose to shape (3, nSims)
            #   - angles: sobolRef_cat[:, 3:5] = [theta, phi] in degrees
            #       theta: azimuth (x-y plane)
            #       phi:   inclination from +z
            ref = sobolRef_cat[:, :-2].T
    
            sobolAngle = sobolRef_cat[:, -2:]
            theta_deg = sobolAngle[:, 0]
            phi_deg   = sobolAngle[:, 1]
    
            theta = np.radians(theta_deg)
            phi   = np.radians(phi_deg)
    
            # Convert spherical angles -> unit direction vectors (3, nSims)
            self.sim['vecDir'] = np.column_stack((
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            )).T
    
        #normalize vectors
        self.sim['vecDir_unit'] = self._normalize_vecDir()
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
        ref = np.asarray(ref)
        comp = np.asarray(comp)
    
        if ref.ndim == 1:
            # (3,)
            step = np.linalg.norm(comp - ref)  # scalar
            jitter = np.random.randn(*ref.shape) * step * self.config.random_jitter  # (3,)
    
        elif ref.ndim == 2:
            # (3, N)
            if ref.shape[0] != 3:
                raise ValueError(f"Expected shape (3, N) for batched inputs; got {ref.shape}.")
    
            # step per column -> shape (N,)
            step = np.linalg.norm(comp - ref, axis=0)
    
            # broadcast step to (1, N) so jitter becomes (3, N)
            jitter = np.random.randn(*ref.shape) * (step[None, :] * self.config.random_jitter)
    
        else:
            raise ValueError(f"`ref` and `comp` must be shape (3,) or (3, N). Got ndim={ref.ndim}.")
            
        fixed_dim = self.config.fixed_color_dim
        if fixed_dim is not None:
            jitter[fixed_dim] = 0
        return jitter
    
    def _normalize_vecDir(self, eps=1e-12):
        """
        Compute a unit-length version of the direction vectors in `self.sim['vecDir']`.
    
        `self.sim['vecDir']` is expected to have shape (ndims_total, nSims), where each
        column is one direction vector. This method does NOT modify `self.sim['vecDir']`;
        it stores the normalized vectors in `self.sim['vecDir_unit']`.
    
        Notes
        -----
        - In 2D-slice settings, the fixed dimension is typically already set to 0, so
          normalization keeps it at 0.
        - Vectors with norm < `eps` are left unchanged (i.e., divided by 1.0).
    
        Parameters
        ----------
        eps : float
            Small constant to avoid division by zero for near-zero vectors.
    
        Side effects
        ------------
        Adds:
            self.sim['vecDir_unit'] : ndarray, shape (ndims_total, nSims)
                Column-wise unit vectors derived from `self.sim['vecDir']`.
        """
        v = np.asarray(self.sim["vecDir"], dtype=float)
    
        norms = np.linalg.norm(v, axis=0, keepdims=True)  # (1, nSims)
        safe_norms = np.where(norms < eps, 1.0, norms)
    
        self.sim["vecDir_unit"] = v / safe_norms    
        
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
        Simulate trials using Sobol-sampled references.
    
        For each simulated trial:
          1) Choose a reference (from Sobol sampling).
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
        # Set the random seed if provided, otherwise generate a random seed
        np.random.seed(self.config.random_seed)
        
        # Precompute Sobol references and their sampling directions.
        self._sobol_generate_ref()
        
        # Normalize direction vectors (stored as `self.sim['vecDir_unit']`).
        self._normalize_vecDir()
        
        # indices of varying dimensions (e.g., [0, 1] in 2D)
        vd = self.config.varying_color_dim
    
        # Iterate over the grid points of the reference stimulus.
        for i in range(self.config.nSims):     
            # Reference stimulus for this trial (shape: (3,))
            ref_i = self.sim['ref'][:,i]
            
            # 1) Find a "threshold" comparison point (no jitter yet)
            if self.config.stim_space == 'Isoluminant plane':
                # Work in 2D W-space: use only the varying dimensions for the solver.
                ref_ii = ref_i[vd] # (2,)
                
                # find_threshold_point_on_isoluminant_plane returns several outputs;
                # we only need the threshold location in W-space (stim_thres, shape (2,))
                *_, stim_thres = sim_CIELab.find_threshold_point_on_isoluminant_plane(\
                                                    ref_ii, 
                                                    self.sim['vecDir_unit'][vd,i], 
                                                    self.config.M_RGBTo2DW,
                                                    self.config.M_2DWToRGB,
                                                    self.sim['deltaE_1JND'],
                                                    coloralg = self.config.gt
                                                    )
            else: 
                # RGB case: solve for threshold step length along the sampled direction.
                opt_vecLen = sim_CIELab.find_vecLen(ref_i, 
                                                    self.sim['vecDir_unit'][:,i],
                                                    self.sim['deltaE_1JND'],
                                                    coloralg = self.config.gt
                                                    )
                #stim_thres here is (3,) in RGB space
                stim_thres = opt_vecLen * self.sim['vecDir'][:,i] + ref_i
                
            # 2) Standardize to a full 3-vector + enforce plane definition/bounds
            # ensures shape (3,) and inserts fixed dim if needed
            stim_thres_v = self._validate_stim(stim_thres)
            
            # 3) Add jitter around threshold point (in the same space as ref/comp)
            jitter_i = self._jitter(ref_i, stim_thres_v)
            self.sim['comp'][:,i] = self._validate_stim(stim_thres_v + jitter_i)
            
            # 4) Compute ΔE, probability correct, and simulate a binary response
            self.sim['deltaE'][i], self.sim['probC'][i], self.sim['resp_binary'][i] = \
                self.run_sim_1ref(sim_CIELab, 
                                  self.sim['ref'][:,i], 
                                  self.sim['comp'][:,i]
                                  )
                
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
class TrialPlacement_gridRef(TrialPlacement_sobolRef):
    def __init__(self, gt_CIE, config: StimConfig):
        """
        Non-adaptive trial placement using precomputed CIE-based isothreshold contours.
    
        This class loads ground-truth discrimination contours (ellipse/ellipsoid parameters)
        from `gt_CIE` and uses them to generate simulated comparison stimuli around each
        reference point on a fixed grid. For each reference, it samples comparison points
        near the threshold contour, computes ΔE in CIELab (1976/1994/2000) between each pair
        of reference and sampled comparison stimuli, evaluates the probability of correct
        based on a specified psychometric function, and generates binary responses.
    
        Parameters
        ----------
        gt_CIE : dict
            Dictionary produced by the ground-truth scripts
            (e.g., `Isothreshold_ellipses_3slices_CIE1994.py`,
                   `Isothreshold_ellipses_isoluminant_CIE1994.py`,
                   `Isothreshold_ellipsoids_CIE1994.py`). 
    
        config : StimConfig
            Stimulus configuration describing the plane/slice, grid size, number of simulated
            trials per reference, jitter, and (if needed) W<->RGB transforms.
    
        """
        super().__init__(config=config)
        self._unpack_gt(gt_CIE)
        self._extract_ref_points()
            
    def _unpack_gt(self, gt_CIE):
        """    
        The precomputed gt_CIE dictionary contains multiple simulation settings. We pick
        the one corresponding to this run’s stimulus space and grid specification, and
        store three blocks as instance attributes:
            - self.gt_CIE_stim     : stimulus grid / reference locations used in the sim
            - self.gt_CIE_results  : simulation outputs (e.g., ellipses/ellipsoids or ΔE-derived results)
    
        Key convention
        -------------
        Entries are indexed by a string suffix `keystr`:
            - For full 3D settings (stim_space in {'cube', 'Isoluminant plane'}):
                  keystr = "grid{num_grid_pts}"
              (no fixed dimension is used, so no fixed_val appears in the key)
    
            - For 2D slices embedded in RGB (e.g., RG/RB/GB planes with one fixed channel):
                  keystr = "grid{num_grid_pts}_fixedVal{fixed_val}"
    
        Notes
        -----
        `fixed_val` must match the exact formatting used when creating the gt_CIE file
        (e.g., 0.15 vs 0.1500), otherwise key lookup will fail.
        
        """
        if self.config.stim_space in ['Isoluminant plane', 'cube']:
            keystr = f'grid{self.config.num_grid_pts}'
        else:
            keystr = f'grid{self.config.num_grid_pts}_fixedVal{self.config.fixed_val}'
        
        self.gt_CIE_stim = gt_CIE[f'stim_{keystr}']
        self.gt_CIE_results = gt_CIE[f'results_{keystr}']
                          
    def _extract_ref_points(self):
        """
        Load the reference stimulus grid and shared stimulus metadata from `self.gt_CIE_stim`.
    
        Behavior
        --------
        - 2D slice experiments (ndims == 2):
            `stim['ref_points']` is typically stored as a stack of slices (one per fixed RGB
            dimension/value combination). We select the slice specified by
            `self.config.fixed_color_dim`.
    
        - 3D experiments (ndims == 3):
            We use the full reference grid (no slicing).
    
        Side effects
        ------------
        Populates `self.sim` with:
          - 'ref_points'      : reference locations used for trial placement / evaluation
          - 'background_RGB'  : background color used in the CIE simulation
          - 'deltaE_1JND'     : target ΔE value defining the “1 JND” threshold distance
          
        """
        try: 
            stim = self.gt_CIE_stim
    
            # If 2D: pick one slice; otherwise: take everything
            sel = self.config.fixed_color_dim if self.config.ndims == 2 else slice(None)
    
            self.sim.update({
                'ref_points': stim['ref_points'][sel],
                'background_RGB': stim['background_RGB'],
                'deltaE_1JND': stim['deltaE_1JND'],
            })
            
        except KeyError as e:
            print(f"Error: Missing expected data in gt_CIE_stim - {e}")
        except IndexError as e:
            print(f"Error: Indexing issue with RGB plane - {e}")
        
    def _initialize(self):
        """
        Initializes simulation arrays to store computed data, including:
        - comparison values
        - Probability of correct response
        - Binary response (correct/incorrect)
        - deltaE color differences
    
        The arrays are preallocated with NaNs for efficient storage and processing.
        """
        # Determine the base shape based on 
        base_shape = (self.config.num_grid_pts,) * self.config.ndims
        nSims = self.config.nSims
    
        # Common arrays for all cases
        self.sim.update({
            'comp': np.full(base_shape + (3, nSims), np.nan),
            'probC': np.full(base_shape + (nSims,), np.nan),
            'resp_binary': np.full(base_shape + (nSims,), np.nan),
            'deltaE': np.full(base_shape + (nSims,), np.nan)
        })
            
    def _extract_gt_ellParam(self, ref_idx):
        """
        Extract the parameters that define the elliptical / ellipsoidal threshold contours
        
        """
        if self.config.ndims == 2:
            ellPara = self.gt_CIE_results['ellParams'][self.config.fixed_color_dim][*ref_idx]      
        else:
            # Use ellipsoidal parameters to generate comparison stimuli
            i,j,k = ref_idx
            ellPara = self.gt_CIE_results['ellParams'][i][j][k]
        return ellPara
                
    def _random_points_on_unit_circle(self):
        """
        Sample noisy points around the unit circle (2D).
        
        Procedure
        ---------
        1) Sample angles θ uniformly from [0, 2π).
        2) Convert to unit-circle coordinates: (cosθ, sinθ).
        3) Add i.i.d. Gaussian jitter to x and y:
               x = cosθ + N(0, jitter)
               y = sinθ + N(0, jitter)
        
        Returns both the noisy points and the underlying noise-free unit-circle points.
        
        """
        N = self.config.nSims
        jitter = self.config.random_jitter
           
        # Generate random angles and compute unit circle coordinates
        randTheta = np.random.rand(1, N) * 2 * np.pi
        randx_noNoise, randy_noNoise = np.cos(randTheta), np.sin(randTheta)
        
        # Apply independent Gaussian noise to x and y coordinates
        noise_x = np.random.randn(1, N) * jitter
        noise_y = np.random.randn(1, N) * jitter
        randx, randy = randx_noNoise + noise_x, randy_noNoise + noise_y
        
        return randx, randy, randx_noNoise, randy_noNoise

    def _random_points_on_unit_sphere(self):
        """
        Sample noisy points around the unit sphere surface (3D).
        
        Uniform sampling on the sphere
        ------------------------------
        To sample points uniformly on a sphere, we cannot sample φ uniformly in [0, π],
        because that would over-sample the poles. Instead we sample:
            u = cos(φ) ~ Uniform(-1, 1)
        and then set:
            φ = arccos(u)
        with θ ~ Uniform(0, 2π).
        
        We then convert spherical coordinates (θ, φ) to Cartesian coordinates on the
        unit sphere:
            x = sinφ cosθ
            y = sinφ sinθ
            z = cosφ
        
        Finally, we add i.i.d. Gaussian jitter to each coordinate:
            (x, y, z) += N(0, jitter^2)
        
        Returns both the noisy points and the underlying noise-free unit-sphere points.
        
        """
        N = self.config.nSims
        jitter = self.config.random_jitter
        
        #Uniformly distributed angles between 0 and 2*pi
        randtheta = np.random.rand(1, N) * 2 * np.pi
        
        #If you were to sample phi uniformly, you'd place too many points near 
        #the poles and too few points near the equator, because the surface area 
        #decreases near the poles. To correct for this, we uniformly sample 
        #cos(phi). This ensures that points are spaced evenly across the sphere's
        #surface because the cosine of phi (which ranges from -1 to 1) accounts 
        #for the different sizes of latitude bands as you move from pole to pole.
        #Uniformly sampled from [-1, 1] ensures uniform distribution along
        #the z-axis
        randphi_temp = np.random.uniform(-1, 1, N)    # cos(theta) for polar angle
        # Converted from costheta using the inverse cosine function to get the angle.
        randphi = np.arccos(randphi_temp)
        
        # Convert spherical coordinates to Cartesian (unit sphere)
        randx_noNoise = np.sin(randphi) * np.cos(randtheta)
        randy_noNoise = np.sin(randphi) * np.sin(randtheta)
        randz_noNoise = np.cos(randphi)
        
        # Apply independent Gaussian noise to each coordinate
        noise_x = np.random.randn(1, N) * jitter
        noise_y = np.random.randn(1, N) * jitter
        noise_z = np.random.randn(1, N) * jitter
        randx, randy, randz = randx_noNoise + noise_x, randy_noNoise + noise_y, randz_noNoise + noise_z
        
        return randx, randy, randz, randx_noNoise, randy_noNoise, randz_noNoise
         
    def sample_comp_2DNearContour(self, ref, paramEllipse):
        """
        Samples comparison stimuli near a 2D elliptical isothreshold contour.
        
        This function generates simulated comparison stimuli based on a reference
        point and the parameters of an ellipse. If the simulation is performed in 
        RGB space (e.g., RG/RB/GB planes), `ref` and outputs are RGB values bounded 
        between 0 and 1. If the simulation is on the isoluminant plane, the values 
        are in Wishart space, also bounded between -1 and 1, with the third dimension 
        filled with 1s.
        
        Parameters
        ----------
        ref : array-like, shape (3,)
            The reference stimulus. Only the varying dimensions are updated.
        paramEllipse : array-like, shape (5,)
            Ellipse parameters: [xc, yc, semi_axis1, semi_axis2, rotation_angle (deg)]
        
        Returns
        -------
        comp_sim : array, shape (3, nSims)
            Simulated comparison stimuli.
        rand_stretched : array, shape (2, nSims)
            Ellipse-transformed coordinates.
        rand_noisy : array, shape (2, nSims)
            Unit circle points with added noise.
        rand_noNoise : array, shape (2, nSims)
            Original unit circle points without noise.
        """
        
        # sanity check: ellipse center (xc, yc) should match the reference in varying dims
        vd = self.config.varying_color_dim
        if not np.allclose(ref[vd], paramEllipse[:2], rtol=1e-3, atol=1e-6):
            raise ValueError("Ellipse center (xc, yc) does not match the provided reference (varying dims).")
        
        # Initialize output matrix
        comp_sim = np.full((3, self.config.nSims), np.nan)
        
        # Generate noisy and noise-free points on a unit circle
        randx, randy, randx_noNoise, randy_noNoise = self._random_points_on_unit_circle()
        
        # Stretch unit circle to match ellipse shape
        randx_stretched, randy_stretched = stretch_unit_circle(
            randx, randy, paramEllipse[2], paramEllipse[3])
        
        # Rotate and translate points based on reference and ellipse parameters
        comp_sim[vd] = rotate_relocate_stretched_ellipse(
           randx_stretched, randy_stretched, paramEllipse[-1], *ref[vd])
        
        # Clip values to remain within the valid space
        comp_sim = self._validate_stim(comp_sim)
        
        # Return simulated data and intermediate transformations
        return comp_sim, \
            np.vstack((randx_stretched, randy_stretched)), \
            np.vstack((randx, randy)), \
            np.vstack((randx_noNoise, randy_noNoise))

    def sample_comp_3DNearContour(self, ref, paramEllipsoid):
        """
        Samples comparison stimuli near a 3D isothreshold ellipsoidal contour.
        
        This function generates simulated comparison stimuli around a reference
        point using ellipsoid parameters. 
        
        Parameters
        ----------
        ref : array-like, shape (3,)
            The reference stimulus (RGB or Wishart).
        paramEllipsoid : dict with keys ['radii', 'evecs']
            - radii: array-like, shape (3,)
                Semi-axis lengths of the ellipsoid.
            - evecs: array-like, shape (3, 3)
                Eigenvectors defining the ellipsoid orientation.
        
        Returns
        -------
        comp_sim : array, shape (3, nSims)
            Simulated comparison stimuli.
        rand_stretched : array, shape (3, nSims)
            Ellipsoid-transformed coordinates.
        rand_noisy : array, shape (3, nSims)
            Unit sphere points with noise.
        rand_noNoise : array, shape (3, nSims)
            Original unit sphere points without noise.
        """
        
        radii, eigenVec = paramEllipsoid['radii'], paramEllipsoid['evecs']
        
        # Generate noisy and noise-free points on a unit sphere
        randx, randy, randz, randx_noNoise, randy_noNoise, randz_noNoise = \
            self._random_points_on_unit_sphere()
        
        # Stretch to match ellipsoid dimensions
        randx_stretched, randy_stretched, randz_stretched = stretch_unit_circle(
            randx, randy, radii[0], radii[1], z=randz, ax_length_z=radii[2])
        
        # Stack coordinates
        xyz = np.vstack((randx_stretched, randy_stretched, randz_stretched))
        
        # Rotate and translate based on ellipsoid orientation and center
        comp_sim = eigenVec @ xyz + ref[:, None]
        
        # Clip values to remain in valid space
        comp_sim = self._validate_stim(comp_sim)
        
        # Return simulated data and intermediate transformations
        return comp_sim, \
            np.vstack((randx_stretched, randy_stretched, randz_stretched)), \
            np.vstack((randx, randy, randz)), \
            np.vstack((randx_noNoise, randy_noNoise, randz_noNoise))
            
    def run_sim_1ref(self, sim_CIELab, ref, comp):
        """
        Simulates responses for comparison stimuli around a single reference point.
        
        This method computes perceptual differences (ΔE), probability of correct 
        responses using a Weibull function, and simulates binary responses for a 
        set of comparison stimuli around a fixed reference.
        
        Parameters
        ----------
        sim_CIELab : SimThresCIELab object
            Used to compute perceptual differences (delta E).
        ref : array-like, shape (3,)
            Reference stimulus (in RGB or Wishart space).
        comp : array-like, shape (3, nSims)
            Comparison stimuli (in RGB or Wishart space).
        
        Returns
        -------
        deltaE : ndarray, shape (nSims,)
            Perceptual differences between reference and comparison stimuli.
        probC : ndarray, shape (nSims,)
            Probability of correct identification based on ΔE.
        resp_binary : ndarray, shape (nSims,)
            Simulated binary responses (1 = correct, 0 = incorrect).
        """
        N = self.config.nSims
        
        # Initialize outputs
        deltaE = np.full((N,), np.nan)
        probC = np.full((N,), np.nan)
        resp_binary = np.full((N,), np.nan)
        
        # Convert to RGB space if operating in Wishart space because we need 
        # stimuli to be in RGB space to compute CIEL*a*b* values
        rgb_ref = self._convert_stim_toRGB(ref)
        rgb_comp = self._convert_stim_toRGB(comp)
            
        # Compute deltaE and probability of correct response for each comparison
        for n in range(N):
            deltaE[n] = sim_CIELab.compute_deltaE(rgb_ref, None, None,
                comp_RGB=rgb_comp[:, n], method=self.config.gt
            )
            probC[n] = self.WeibullFunc(deltaE[n],
                                        self.sim['alpha'], 
                                        self.sim['beta'], 
                                        self.sim['guessing_rate']
            )
        
        # Simulate binary responses using Bernoulli sampling
        randNum = np.random.rand(N)
        resp_binary = (randNum < probC).astype(int)
        
        return deltaE, probC, resp_binary            

    def run_sim(self, sim_CIELab):
        """
        Run a full non-adaptive simulation over the reference grid.
        
        This method iterates over every reference stimulus in `self.sim["ref_points"]`.
        At each reference location, it:
        
          1) retrieves the precomputed ground-truth contour parameters stored in the
             gt_CIE tables (ellipse in 2D, ellipsoid in 3D),
          2) draws comparison stimuli near that contour using the subclass-specific
             sampler (e.g., random directions + jitter around the boundary),
          3) evaluates the trial outcome by computing ΔE(ref, comp), converting it to
             p(correct) via the Weibull psychometric function, and sampling a binary
             response.
        
        All generated stimuli and responses are written back into `self.sim` using the
        same grid indexing as `ref_points`, so downstream code can directly compare
        placement/response statistics across reference locations.
        
        Parameters
        ----------
        sim_CIELab : SimThresCIELab-like object
            Helper that computes color differences (ΔE) after converting stimuli to CIELab
            using our monitor calibration (primaries/SPD) and Stockman–Sharpe 2° cone
            fundamentals.
        """
        # Seed NumPy's RNG so sampling (angles/jitter/Bernoulli draws) is reproducible.
        np.random.seed(self.config.random_seed)
        
        # Reference grid resolution along each axis; total grid has shape (N,)*ndims.
        N = self.config.num_grid_pts
        grid_shape = (N,) * self.config.ndims
        
        # Loop over all reference indices (i,j) for 2D or (i,j,k) for 3D.
        for idx in np.ndindex(grid_shape):
            # Reference stimulus at this grid location.
            ref = self.sim["ref_points"][idx]
        
            # Ground-truth contour parameters at this reference:
            #   - 2D: ellipse parameters
            #   - 3D: ellipsoid parameters
            ellParam = self._extract_gt_ellParam(idx)
        
            # Sample comparison stimuli near the threshold contour.
            # Note: the sampling methods already call `_validate_stim`, so `comp` is in-bounds
            # and consistent with the configured slice/space.
            if self.config.ndims == 2:
                comp, *_ = self.sample_comp_2DNearContour(ref, ellParam)
            else:
                comp, *_ = self.sample_comp_3DNearContour(ref, ellParam)
        
            # Store sampled comparisons for this reference.
            self.sim["comp"][idx] = comp
        
            # Compute ΔE for each comparison, map to percent-correct via the psychometric
            # function, and draw binary responses.
            self.sim["deltaE"][idx], self.sim["probC"][idx], self.sim["resp_binary"][idx] = \
                self.run_sim_1ref(sim_CIELab, ref, comp)

#%% 
class TrialPlacement_sobolRef_W(TrialPlacement_sobolRef):
    def __init__(self, gt_Wishart, config: StimConfig_W_sobolref):
        """
        Non-adaptive trial placement using *Wishart-model* predicted thresholds on a
        fixed grid of references.
        
        This subclass mirrors `TrialPlacement_gridRef` (CIE-ground-truth version), but
        uses a fitted Wishart model to provide the local threshold contour at each
        reference. 
        
        Parameters
        ----------
        gt_Wishart : dict
            Container holding Wishart-model outputs evaluated on a grid of references.
            Expected keys include:
              - 'model_pred_Wishart' : fitted/predictive model object
              - 'model'              : model object
              - 'W_est'              : the best-fitting weight matrix
        
        config : StimConfig_W
            Simulation configuration (ndims, num_grid_pts, nSims, jitter, bounds, etc.).
        
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
          - vecDir/vecDir_unit: last row is fixed at 0 so directions lie in the 2D plane.
        
        Shapes
        ------
          - ref, comp, vecDir, vecDir_unit : (3, nSims)
          - probC, resp_binary             : (nSims,)
        """
        # Determine the base shape based on 
        nSims = self.config.nSims
                
        # Common arrays for all cases
        self.sim.update({
            'ref': np.full((3, nSims), np.nan),
            'comp': np.full((3, nSims), np.nan),
            'vecDir': np.full((3, nSims), np.nan),
            'vecDir_unit': np.full((3, nSims), np.nan),
            'probC': np.full((nSims,), np.nan),
            'resp_binary': np.full((nSims,), np.nan),
        })
        
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
        nSims = self.config.nSims
        
        # Stored stimulus dimensionality in self.sim['ref'] (often 3 because of a filler dim).
        nd = self.sim["ref"].shape[0]
        
        # Candidate step sizes along vecDir_unit, shape: (nRep,)
        vecLength_grid = np.asarray(self.gt_model_pred._set_up_grid_search(), dtype=float)
        nRep = vecLength_grid.size
        
        # Broadcast refs and directions to evaluate all (step length × sim) candidates.
        # self.sim['ref'] and self.sim['vecDir_unit'] are (nd, nSims).
        # After transpose: (nSims, nd); after adding leading axis: (1, nSims, nd);
        # after repeat: (nRep, nSims, nd).
        ref_rep = np.repeat(self.sim["ref"].T[None], nRep, axis=0)
        vecDir_rep = np.repeat(self.sim["vecDir_unit"].T[None], nRep, axis=0)
        
        # Build candidate comparisons for every step length, every sim:
        # vecLength_grid[:, None, None] broadcasts to (nRep, nSims, nd) automatically here.
        comp_rep = ref_rep + vecDir_rep * vecLength_grid[:, None, None]
        
        # Flatten for model evaluation, and keep only the varying dims:
        # (nRep, nSims, nd) -> (nRep*nSims, nd) -> select varying dims -> (nRep*nSims, |vd|)
        ref_test = np.reshape(ref_rep, (-1, nd))[:, vd]
        comp_test = np.reshape(comp_rep, (-1, nd))[:, vd]
        
        # Model-predicted percent-correct for every candidate pair, shape: (nRep*nSims,)
        probC = self.gt_model_pred._compute_pChoosingX1(ref_test, comp_test)
        
        # Reshape back to (nRep, nSims): rows=candidate step lengths, cols=sims
        probC_reshape = np.reshape(probC, (nRep, nSims))
        
        # For each sim (each column), pick the step length index whose pC is closest to target_pC
        idx = np.abs(probC_reshape - self.gt_model_pred.target_pC).argmin(axis=0)  # (nSims,)
        
        # Gather the selected comparison stimuli: (nSims, nd) -> return as (nd, nSims)
        comp_thres = comp_rep[idx, np.arange(nSims)]
        return comp_thres.T
    
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
        
        # Preallocate step vectors (ref -> threshold) in the varying subspace
        vecLength_thres_approx = np.full((nd, nSims), np.nan)
        
        # 2-5) For each ref:
        #   2) Σ -> ellipse params (axis lengths and rotation angles)
        #   3) scale axis lengths by `Sigmas_scaler`
        #   4) find distance r to exit ellipse / ellipsoid along dir_n
        #   5) step = r * dir_n, then comp = ref + step
        for n in range(nSims):
            if nd == 2:
                # Convert covariance -> ellipse axes length and rotation angle
                _, _, axisl, theta = covMat_to_ellParamsQ(Sigmas[n])
            else:
                _, _, axisl, theta, phi = covMat_to_ellParamsQ(Sigmas[n])
            # Enlarge the noise ellipse to approximate the threshold contour
            axisl_enlarged = axisl * self.Sigmas_scaler
        
            # Direction for this simulation in varying dims (shape: (d,))
            dir_n = self.sim["vecDir_unit"][vd, n]
        
            # Distance from ellipse / ellipsoid center to boundary along direction `dir_n`
            if nd == 2:
                r = distance_to_ellipse_boundary(*axisl_enlarged, theta, *dir_n)
            else:
                r = distance_to_ellipsoid_boundary(*axisl_enlarged, theta, phi, *dir_n)
        
            # Step vector in the varying subspace
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
        Simulate non-adaptive trials with Sobol-sampled references and directions.
        
        Pipeline
        --------
        1) Sample reference locations and direction angles via Sobol, then convert angles
           into direction vectors (`vecDir`).
        2) Normalize direction vectors to unit length (`vecDir_unit`).
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
        # for reproducibility
        np.random.seed(self.config.random_seed)
        
        # 1) Sobol sample reference locations + direction angles, then build `self.sim['vecDir']`.
        self._sobol_generate_ref()
        
        # 2) Normalize direction vectors (stored as `self.sim['vecDir_unit']`).
        self._normalize_vecDir()
        
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
            self.Sigmas_scaler, *_ = compute_radii_scaler_to_reach_targetPC(
                self.gt_model_pred.target_pC, ndims = self.config.ndims
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
        
#%%
class TrialPlacement_gridRef_W(TrialPlacement_gridRef):
    def __init__(self, gt_Wishart, config: StimConfig_W):
        """
        Non-adaptive trial placement using *Wishart-model* predicted thresholds on a
        fixed grid of references.
        
        This subclass mirrors `TrialPlacement_gridRef` (CIE-ground-truth version), but
        uses a fitted Wishart model to provide the local threshold contour at each
        reference. For each reference point, we:
          1) sample comparison stimuli near the predicted iso-threshold contour
             (ellipse in 2D, ellipsoid in 3D),
          2) evaluate the psychometric response model, and
          3) generate binary responses.
        
        Parameters
        ----------
        gt_Wishart : dict
            Container holding Wishart-model outputs evaluated on a grid of references.
            Expected keys include:
              - 'model_pred_Wishart' : fitted/predictive model object
              - 'grid'              : reference grid of shape (..., ndims)
        
        config : StimConfig_W
            Simulation configuration (ndims, num_grid_pts, nSims, jitter, bounds, etc.).
        
        """
        super().__init__(gt_CIE=gt_Wishart, config=config)  # pass it through

    def _unpack_gt(self, gt_Wishart):
        """
        Unpack Wishart-model ground truth from `gt_Wishart`.
        
        Side effects
        ------------
        Sets:
          - self.gt_model_pred : the predictive model object (deep-copied)
          - self.gt_ellParams  : per-reference contour parameters (ellipse/ellipsoid)
          - self.gt_grid       : reference grid locations
        """
        self.gt_model_pred = deepcopy(gt_Wishart['model_pred_Wishart'])
        
        # `params_ell` should contain ellipse params (2D) or ellipsoid params (3D)
        # indexed over the reference grid.
        self.gt_ellParams = self.gt_model_pred.params_ell
        
        # Reference grid (stored with stimulus dims on last axis: (..., ndims))
        if self.config.ndims == 2:
            self.gt_grid = gt_Wishart['grid']
        else:
            self.gt_grid = gt_Wishart['grid_trans']
            
    def _initialize(self):    
        """
        Preallocate simulation buffers (NaN-filled) for `nSims` trials per reference.
        
        This class operates in the model/W space but stores stimuli and directions as
        3-vectors for a consistent interface with RGB conversion. 
        
        Shapes
        ------
        base_shape = (num_grid_pts,) * ndims, where ndims is 2 or 3.
        Stored arrays:
          - comp        : base_shape + (3, nSims)   comparison stimuli (3-vector form)
          - probC       : base_shape + (nSims,)     p(correct) per simulated trial
          - resp_binary : base_shape + (nSims,)     simulated binary responses
          
        """
        nSims = self.config.nSims
        
        # Grid resolution inferred from previously extracted references.
        base_shape = (self.config.num_grid_pts,) * self.config.ndims
        
        self.sim.update({
            'comp': np.full(base_shape + (3, nSims,), np.nan),
            'probC': np.full(base_shape + (nSims,), np.nan),
            'resp_binary': np.full(base_shape + (nSims,), np.nan),
        })
        
    def _extract_ref_points(self):
        """
        Load and validate the reference grid from Wishart outputs.
        
        The stored grid convention is (..., ndims) but `_validate_stim` expects
        (ndims, ...). We therefore:
          1) move axis: (..., ndims) -> (ndims, ...)
          2) validate (insert 1's; clip to bounds)
          3) move axis back: (ndims, ...) -> (..., ndims)
        
        Side effects
        ------------
        Updates:
          - self.sim['ref_points'] : validated reference grid, shape (..., 3) after
            slice completion (if needed) and clipping.
        """
        grid = np.asarray(self.gt_grid)
        grid_valid = np.moveaxis(self._validate_stim(np.moveaxis(grid, -1, 0)), 0, -1)
        self.sim.update({"ref_points": grid_valid})

    def _extract_gt_ellParam(self, ref_idx):
        """
        Return the contour parameters at a given reference-grid index.
        
        Parameters
        ----------
        ref_idx : tuple
            Grid index:
              - 2D: (i, j)
              - 3D: (i, j, k)
        
        Returns
        -------
        ellPara : array-like (5 parameters for 2D) 
                 dict for 3D  that includes eigenvalues and eigenvectors
        """
        if self.config.ndims == 2:
            i,j = ref_idx
            ellPara = self.gt_ellParams[i][j]
        else:
            # Use ellipsoidal parameters to generate comparison stimuli
            i,j,k = ref_idx
            ellPara = self.gt_ellParams[i][j][k]
        return ellPara

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

    def run_sim(self):
        """
        Generate simulated trials around each reference stimulus and produce binary responses.
        
        For every reference location on the configured grid, this method:
          1) reads out the ground-truth isothreshold contour parameters (ellipse in 2D,
             ellipsoid in 3D),
          2) samples comparison stimuli near that contour (with the subclass-specific
             non-adaptive placement rule),
          3) compute the percent correct using the loaded best-fit Wishart model
          4) simulate a binary response
        
        """
        # Set the random seed if provided, otherwise generate a random seed
        np.random.seed(self.config.random_seed)
        
        N = self.config.num_grid_pts
        shape = (N,) * self.config.ndims
        vd = self.config.varying_color_dim
        
        # Iterate over the grid points of the reference stimulus.
        for idx in np.ndindex(*shape):
            # Reference stimulus at this grid point
            ref_ij = self.sim['ref_points'][idx][vd] 
        
            # Ground-truth ellipse parameters
            ellParam = self._extract_gt_ellParam(idx)
        
            #Generate comparison stimuli based on 2D or 3D sampling
            if self.config.ndims == 2:
                comp_ij, *_ = self.sample_comp_2DNearContour(ref_ij, ellParam)
            else:
                comp_ij, *_ = self.sample_comp_3DNearContour(ref_ij, ellParam)
        
            # Validate and store
            self.sim['comp'][idx] = self._validate_stim(comp_ij)
        
            # Repeat reference to match number of simulated comparisons
            ref_v = np.repeat(ref_ij[None, vd], self.config.nSims, axis=0) 
        
            # Use only varying dimensions for the psychometric evaluation
            comp_v = self.sim['comp'][idx][vd].T
        
            self.sim['probC'][idx], self.sim['resp_binary'][idx] = \
                self.run_sim_1ref(ref_v, comp_v)
