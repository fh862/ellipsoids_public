#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:18:05 2025

@author: fangfang
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from matplotlib.ticker import MaxNLocator
from core.model_predictions import wishart_model_pred
from analysis.ellipses_tools import fit_2d_isothreshold_contour, UnitCircleGenerate
from colormath.color_diff import delta_e_cie2000, delta_e_cie1994, delta_e_cie1976
from colormath.color_objects import LabColor

#%%
class wishart_model_pred_inCIELab(wishart_model_pred):
    def __init__(self, *args, cfg, flag_debug_plot = False, **kwargs):
        """
        Extension of `wishart_model_pred` for working in CIELAB coordinates.

        This subclass keeps all core functionality from the base predictor
        (e.g., stored fitted parameters, Wishart weights, grid handling,
        chromatic-direction setup), and adds a CIELAB-specific configuration
        object `cfg` that describes how a 2D slice (or grid) is embedded in the
        full 3D Lab space.

        Parameters
        ----------
        *args
            Positional arguments forwarded directly to `wishart_model_pred.__init__`.
            In your current base class, these are:

                model : object
                    The fitted Wishart-process model object. Must expose
                    `model.num_dims` (the dimensionality of the fitted model
                    in its internal coordinate system, e.g., 2 for a plane).

                opt_params : any
                    Optimized / fitted parameters returned by your optimizer.

                w_init_key : any
                    RNG key used to initialize the Wishart weights (for
                    reproducibility).

                opt_key : any
                    RNG key used during optimization (for reproducibility).

                W_init : array-like
                    Initial Wishart weight matrix (before fitting).

                W_est : array-like
                    Best-fit Wishart weight matrix (after fitting).

                Sigmas_noise_grid : array-like
                    Noise covariance matrices evaluated on the grid points
                    (i.e., predicted sensory noise for each reference location).

        cfg : object, including:
              - fixed_dim       : which Lab coordinate is held constant (L*, a*, or b*)
              - vis_range       : (min, max, n) for the fixed dimension grid (e.g., L* from 10 to 120 in 23 steps)
              - fixed_slc_idx   : which indices of the fixed-dimension grid to use (multiple slices can be evaluated)
              - varied_slc      : list of 2D reference locations (in the two varying dimensions) to analyze/plot
              - vectors_centroid: optional centroid used for defining/centering direction vectors in the 2D plane

        flag_debug_plot: bool, if True, generate diagnostic 3D scatter plots of
              comparison stimuli and fitted contours in both CIELAB
              space and model (W) space for each grid point.
                  
        **kwargs
            Keyword arguments forwarded to `wishart_model_pred.__init__`.
            These override entries in `self.params` from the base class, e.g.:
              - n_theta / nTheta
              - n_phi / nPhi
              - bds_bruteforce
              - flag_force_centered_ref
              - etc.

        """

        # Using super() ensures we inherit base behavior without duplicating it.
        super().__init__(*args, **kwargs)

        # `cfg` describes how we embed/interpret the model's coordinate system
        # within Lab (e.g., isoluminant plane: fixed L*, varying a*/b*).
        self.cfg = cfg

        # This is separate from `self.ndims` (the model dimensionality).
        # Example: self.ndims may be 2 (a slice), but Lab is always 3.
        self.ndims_lab = 3
        
        self.flag_debug_plot = flag_debug_plot
        
    def _init_model_pred_list(self):
        # Ellipse parameter containers for each grid point on the slice.
        # Use per-index lists to store fits; length = num_grid_pts1.
        self.params_ell      = [[] for _ in range(self.num_grid_pts1)]
        self.params_ell_CLab = [[] for _ in range(self.num_grid_pts1)]

        shape_base = (self.num_grid_pts1, self.num_grid_pts2)
        
        #save deltaE (they meant to be constant, but since we do grid search, they might not match it exactly)
        self.deltaE = np.full(shape_base + (self.params['n_theta'], ), np.nan)

        # These arrays store the *fitted* ellipse contours at each slice grid point.
        # Even though the fit is 2D (in the varying dimensions), we store contours
        # in full Lab (L*, a*, b*) so downstream plotting is consistent across
        # different slice choices.
        #
        # Shape convention:
        #   (G1, G2, 3, nTheta)
        #     - 3 corresponds to (L*, a*, b*)
        #     - nTheta is the fine angular sampling used to render a smooth contour
        #
        # For the fixed Lab dimension, values are filled using the fixed value(s)
        # specified by `cfg` for that slice.
        self.fitEll_unscaled_WPPM = np.full(shape_base + (self.ndims_lab, self.params['nTheta']), np.nan)
        self.fitEll_unscaled_CLab = np.full(self.fitEll_unscaled_WPPM.shape, np.nan)

        # Same ellipses, scaled for visualization only (e.g., clearer overlays).
        self.fitEll_scaled_WPPM = np.full(self.fitEll_unscaled_WPPM.shape, np.nan)
        self.fitEll_scaled_CLab = np.full(self.fitEll_unscaled_WPPM.shape, np.nan)

        # Threshold *points* (unscaled) sampled along directions before ellipse fit.
        # Shape matches ellipse arrays but stores raw contour samples.
        self.Lab_thres_unscaled_WPPM = np.full(shape_base + (self.ndims_lab, self.params['n_theta']), np.nan)
        self.Lab_thres_unscaled_CLab = np.full(self.Lab_thres_unscaled_WPPM.shape, np.nan)

        # Threshold points after applying the visualization scaler.
        self.Lab_thres_scaled_WPPM = np.full(self.Lab_thres_unscaled_WPPM.shape, np.nan)
        self.Lab_thres_scaled_CLab = np.full(self.Lab_thres_unscaled_WPPM.shape, np.nan)
        
        # Brute-force simulation bookkeeping:
        #   n_theta          = number of radial directions in plane
        #   ngrid_bruteforce = number of radial steps per direction
        #   nRepeats         = total comparison samples per reference
        # Example (defaults): 16 × 1000 = 16000.
        self.nRepeats = self.params['n_theta'] * self.params['ngrid_bruteforce']

        # Storage for reference/compare stimuli in both W-space and Lab/RGB:
        #   w_ref        : reference in W-space (1 × 3) per grid point
        #   rgb_ref      : reference in RGB
        self.w_ref   = np.full(shape_base + (1, self.ndims_lab,), np.nan)
        self.rgb_ref = np.full(self.w_ref.shape, np.nan)
            
    def _init_model_pred_list_for_fixed_dim(self):
        """
        Initialize buffers for storing threshold points along the *fixed* Lab dimension.
    
        This is used when we additionally compute an out-of-plane threshold for each
        reference on the slice grid, i.e., moving along the fixed coordinate
        (L*, a*, or b*) while holding the other two dimensions constant.
    
        Buffers created
        ---------------
        Lab_thres_at_fd_unscaled_WPPM : array, shape (G1, G2, 3)
            Threshold comparison point in full Lab coordinates (L*, a*, b*).
        Lab_thres_at_fd_scaled_WPPM : array, shape (G1, G2, 3)
            Same point after applying visualization scaling about the reference.
    
        Returns
        -------
        None
        """
        shape_grid = (self.num_grid_pts1, self.num_grid_pts2)
        shape_base = shape_grid + (self.ndims_lab,)
    
        # Threshold comparison point along the fixed dimension (full Lab: L*, a*, b*)
        self.Lab_thres_at_fd_unscaled_WPPM = np.full(shape_base, np.nan)
        self.Lab_thres_at_fd_scaled_WPPM = np.full(shape_base, np.nan)
        
    def _debug_plot(self, Lab_comp, w_comp, rgb_comp, fit_ell_lab = None):
        """
        Debug visualization for a set of comparison stimuli.
    
        Plots the candidate comparison points in:
          (1) CIELAB space (L*, a*, b*), optionally overlaying a fitted ellipse/contour.
          (2) Model space after transformation.
    
        Args:
            Lab_comp    : (N, 3) comparison points in CIELAB coordinates.
            w_comp      : (N, 3) same points represented in model space coordinates.
            rgb_comp    : (N, 3) RGB colors (0–1) used to color the scatter points.
            fit_ell_lab : (M, 3) ellipse/contour points in the
                          2D plane defined by the varied dimensions. If provided,
                          this method inserts the fixed Lab coordinate to construct
                          full 3D Lab points for plotting.
        """
        nPts = Lab_comp.shape[0]
    
        # Figure 1: Scatter the candidate points in CIELAB space
        fig1 = plt.figure(figsize=(7, 7))
        ax1 = fig1.add_subplot(111, projection="3d")
    
        # Color each point by its rendered RGB appearance.
        ax1.scatter(*Lab_comp.T, c=rgb_comp, s=1 / nPts * 1000)
    
        # Optionally overlay the fitted ellipse/contour in CIELAB space.
        if fit_ell_lab is not None:
            ax1.plot(*fit_ell_lab.T, color='k')
    
        # Label axes and keep tick density manageable.
        ax1.set_xlabel('L*')
        ax1.set_ylabel('a*')
        ax1.set_zlabel('b*')
        ax1.xaxis.set_major_locator(MaxNLocator(4))
        ax1.yaxis.set_major_locator(MaxNLocator(4))
        ax1.zaxis.set_major_locator(MaxNLocator(4))
        ax1.view_init(elev=20, azim=75)
        plt.show()
    
        # Figure 2: Scatter the same points in model space
        fig2 = plt.figure(figsize=(7, 7))
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.scatter(*w_comp.T, c=rgb_comp, s=1 / nPts * 1000)
        ax2.set_xlabel('Model space dim 1')
        ax2.set_ylabel('Model space dim 2')
        ax2.set_zlabel('Model space dim 3')
        ax2.xaxis.set_major_locator(MaxNLocator(4))
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        ax2.zaxis.set_major_locator(MaxNLocator(4))
        ax2.view_init(elev=20, azim=75)
        plt.show()
        
    def setup_bruteforce_comp(self, Lab_ref, CLabTool):
        """
        Build vectorized comparison stimuli around a single Lab reference by
        sampling many radial directions and many radii within the selected slice.
        The radial directions and finely sampled comparison stimuli are in L*a*b*
        space, and then converted to rgb and model spaces.
    
        Returns (in order):
            w_ref        : (3,)      reference in model (W) space
            w_ref_rep    : (R,3)     reference repeated R times (R = n_theta * ngrid_bruteforce)
            w_comp_rep   : (R,3)     comparison stimuli in W space
            rgb_ref      : (3,)      reference in RGB (N space)
            rgb_comp_rep : (R,3)     comparison stimuli in RGB
            Lab_comp_rep : (R,3)     comparison stimuli in Lab
        """
        # Vector lengths to probe per direction (uses params['bds_bruteforce'] and ['ngrid_bruteforce'])
        # Shape: (ngrid_bruteforce,)
        vecLength_test = self._set_up_grid_search()  # e.g., (1000,)
    
        # Angular directions within the 2D slice (unit vectors on the varied axes)
        # Shape: (2, n_theta)
        grid_chromdir = UnitCircleGenerate(self.params['n_theta']+1)[:,:-1]
    
        # Broadcast directions across radii for vectorized ops
        #   vecDir_org:   (2, n_theta, ngrid_bruteforce)
        #   vecDir_trans: (ngrid_bruteforce, n_theta, 2)
        vecDir_org   = np.tile(grid_chromdir[:, :, None], (1, 1, self.params['ngrid_bruteforce']))
        vecDir_trans = np.transpose(vecDir_org, [1, 2, 0])
    
        # Insert a zero component along the fixed dimension to lift (2D → 3D) in Lab
        # fill_zeros:      (ngrid_bruteforce, n_theta, 1)
        # vecDir_concat:   (ngrid_bruteforce, n_theta, 3)
        fill_zeros   = np.full(vecDir_trans.shape[:2], 0)[:, :, None]
        vecDir_concat = CLabTool.fill_in_vals(self.cfg.fixed_dim, vecDir_trans, fill_zeros)
    
        # Collapse to a flat list of 3D direction vectors
        # Shape: (R, 3) where R = n_theta * ngrid_bruteforce
        vecDir_rep = vecDir_concat.reshape((self.nRepeats, self.ndims_lab))
    
        # Broadcast vector lengths to match directions
        #   vecLength_org:    (n_theta, ngrid_bruteforce)
        #   vecLength_reshape:(R, 1)
        #   vecLength_rep:    (R, 3)  (same length applied to each Lab axis of the 3D direction)
        vecLength_org     = np.tile(vecLength_test, (self.params['n_theta'], 1))
        vecLength_reshape = np.reshape(vecLength_org, (self.nRepeats, 1))
        vecLength_rep     = np.tile(vecLength_reshape, (1, self.ndims_lab))
    
        # Repeat the Lab reference to align with all (direction × radius) samples
        Lab_ref_rep = jnp.tile(Lab_ref, (self.nRepeats, 1)) #(R, 3)
    
        # Generate comparison Lab samples: Lab_ref + (direction × length) 
        Lab_comp_rep = Lab_ref_rep + vecDir_rep * vecLength_rep #(R, 3)
    
        # Convert comparisons to RGB (for gamut checks and later display-space modeling)
        rgb_comp_rep, *_ = CLabTool.convert_lab_rgb(Lab_comp_rep) #(R, 3)
    
        # Map RGB from normalized display space (N: [0,1]) to model W space
        w_comp_rep = self.color_thres_data.N_unit_to_W_unit(rgb_comp_rep) #(R, 3)
    
        # Reference conversions (single → repeated to match comparisons)
        # rgb_ref: (3,), w_ref: (3,), w_ref_rep: (R,3)
        rgb_ref, *_ = CLabTool.convert_lab_rgb(Lab_ref)
        w_ref = self.color_thres_data.N_unit_to_W_unit(rgb_ref)
        w_ref_rep = jnp.tile(w_ref, (self.nRepeats, 1))
    
        return w_ref, w_ref_rep, w_comp_rep, rgb_ref, rgb_comp_rep, Lab_comp_rep
    
    @staticmethod
    def fill_in_fixed_dim(M, idx, fixed_val):
        """
        Insert a fixed row (all = fixed_val) into M at row index idx.
    
        Parameters
        ----------
        M : array-like, shape (m, N)
        idx : int
            Insertion index in [0, m]. (0 prepends, m appends)
        fixed_val : scalar
            Value to fill the inserted row with.
    
        Returns
        -------
        out : ndarray, shape (m+1, N)
        """
        M = np.asarray(M)
    
        if M.ndim != 2:
            raise ValueError(f"M must be 2D (m, N). Got shape {M.shape}.")
        if not (0 <= idx <= M.shape[0]):
            raise IndexError(f"idx must be between 0 and {M.shape[0]} (inclusive). Got {idx}.")
    
        # Create the fixed row
        F = np.full((1, M.shape[1]), fixed_val, dtype=M.dtype)
    
        # Insert along the first dimension (rows)
        return np.concatenate([M[:idx, :], F, M[idx:, :]], axis=0)
        
    def compute_thres_for_fixed_dim_inLab(self, Lab_ref, CLabTool, vis_scaler = 1):
        """
        Compute the threshold comparison point when moving *only* along the slice's
        fixed Lab dimension (i.e., orthogonal to the 2D plane).
    
        Conceptually:
          - Start at a reference stimulus `Lab_ref` (L*, a*, b*).
          - Move along the fixed dimension direction (e.g., +L* if fixed_dim = L)
            with a set of candidate step sizes (grid search).
          - For each candidate, compute oddity-task performance under the WPPM.
          - Pick the candidate whose predicted performance is closest to `target_pC`.
    
        Parameters
        ----------
        Lab_ref : array-like, shape (3,)
            Reference stimulus in CIELAB coordinates (L*, a*, b*).
        CLabTool : object
            Utility providing Lab<->RGB conversion methods:
              - convert_lab_rgb
              - convert_rgb_lab
        vis_scaler : float, optional
            Purely for visualization: rescales the Lab offset from Lab_ref
            (does NOT affect the threshold search). Default is 1 (no scaling).
    
        Returns
        -------
        comp_thres_unscaled : array-like, shape (3,)
            Threshold comparison in Lab (as found by the model/grid-search).
        comp_thres_scaled : array-like, shape (3,)
            Same comparison, but offset from Lab_ref is multiplied by `vis_scaler`.
        """
    
        # Uses params['bds_bruteforce'] and params['ngrid_bruteforce'] to produce
        # a 1D array of candidate vector lengths (e.g., linearly spaced).
        vecLength_test = self._set_up_grid_search()  # shape (N,)
    
        # fixed_dim.value picks which coordinate (L*, a*, or b*) is "fixed" for the
        # slice; here we move along that axis (out-of-plane direction).
        vecDir = np.zeros((self.ndims_lab,), dtype=float)
        vecDir[self.cfg.fixed_dim.value] = 1.0  # e.g., [1,0,0] for L*, [0,1,0] for a*
    
        # Number of candidate radii in the grid search
        nRepeats = self.params['ngrid_bruteforce']
    
        # Replicate direction, lengths, and reference to vectorize computation.
        #   vecDir_rep    : (N, 3)
        #   vecLength_rep : (N, 3)  (same length copied into each coordinate)
        #   Lab_ref_rep   : (N, 3)
        vecDir_rep = np.tile(vecDir[None], (nRepeats, 1))
        vecLength_rep = np.tile(vecLength_test[:, None], (1, self.ndims_lab))
        Lab_ref_rep = jnp.tile(Lab_ref[None], (nRepeats, 1))
    
        # Candidate comparisons: Lab_ref + length * direction
        # Shape: (N, 3)
        Lab_comp_rep = Lab_ref_rep + vecDir_rep * vecLength_rep
    
        # Convert Lab -> RGB -> W (model space)
        #
        # We evaluate the oddity-task model in W-space, so we convert the Lab
        # comparisons to RGB (for gamut handling and display-space mapping),
        # then map RGB in normalized display units N ([0,1]) to W.
        rgb_comp_rep, *_ = CLabTool.convert_lab_rgb(Lab_comp_rep)     # (N, 3)
        w_comp_rep = self.color_thres_data.N_unit_to_W_unit(rgb_comp_rep)  # (N, 3)
    
        # Reference: Lab -> RGB -> W, then repeat to match comparison batch size.
        rgb_ref, *_ = CLabTool.convert_lab_rgb(Lab_ref)               # (3,)
        w_ref = self.color_thres_data.N_unit_to_W_unit(rgb_ref)       # (3,)
        w_ref_rep = np.tile(w_ref[None], (nRepeats, 1))               # (N, 3)
    
        # Predict oddity-task performance for each candidate comparison
        #
        # pChoosingX1: probability that the observer selects the comparison as odd
        # for each candidate radius. Shape: (N,)
        pChoosingX1 = self._compute_pChoosingX1(w_ref_rep, w_comp_rep)
    
        # Pick the candidate whose performance is closest to target criterion.
        min_idx = np.argmin(np.abs(pChoosingX1 - self.target_pC))
    
        # If the best point is at the boundary, the search range is likely too tight.
        if min_idx in (0, self.params['ngrid_bruteforce'] - 1):
            raise ValueError("Threshold hit the grid-search boundary. Consider expanding bds_bruteforce.")
    
        # Convert threshold candidate back to Lab
        w_comp_thres = w_comp_rep[min_idx]                            # (3,)
        rgb_comp_thres = self.color_thres_data.W_unit_to_N_unit(w_comp_thres)  # (3,)
        Lab_comp_thres, *_ = CLabTool.convert_rgb_lab(rgb_comp_thres) # (3,)
    
        # Optional visualization scaling (does not affect the threshold)
        comp_thres_unscaled = Lab_comp_thres
        comp_thres_scaled = (Lab_comp_thres - Lab_ref) * vis_scaler + Lab_ref
    
        return comp_thres_unscaled, comp_thres_scaled

    def convert_Sig_2DisothresholdContour_oddity_inLab(self, Lab_ref, CLabTool, 
                                                       ell_vis_scaler=1):
        """
        Compute the 2D isothreshold contour (oddity task) in a CIELAB slice around a 
        single reference.
    
        Steps:
          1) Generate comparison stimuli radiating from Lab_ref along n_theta directions and
             ngrid_bruteforce radii (via setup_bruteforce_comp). Convert them to values in the
             model space.
          2) Evaluate oddity-task choice probabilities for each (direction, radius) in the 
             model space.
          3) For each direction, find the radius whose performance is closest to target_pC.
          4) Convert those threshold comparisons to from the model space to Lab values and 
             drop the fixed dimension to obtain 2D slice coordinates.
          5) Fit an ellipse to the threshold samples in Lab space and optionally scale for 
             visualization.
          6) Inset fixed value to fixed dimension
    
        Args:
            Lab_ref        : (3,)      Reference point in Lab for the current slice.
            CLabTool       : helper for Lab↔RGB conversions and slice utilities.
            ell_vis_scaler : float     Scale factor applied to the fitted ellipse and the
                                       threshold samples for plotting (no effect on estimates).
    
        Returns:
            fitEll_scaled        : (3, nTheta)    Fine ellipse points scaled for visualization.
            fitEll_unscaled      : (3, nTheta)    Fine ellipse points at the true threshold scale.
            comp_thres_scaled    : (3, n_theta)   Threshold samples after scaling.
            ellParams            : list           Parameters of the fitted ellipse (center, axes, angle, ...).
            comp_thres_unscaled  : (3, n_theta)   Raw threshold samples in slice coordinates.
            w_ref                : (3,)           Reference in W (model) space.
            w_comp_rep           : (R, 3)         All comparison stimuli in W space (R = n_theta * ngrid).
            rgb_ref              : (3,)           Reference in RGB.
            rgb_comp_rep         : (R, 3)         All comparison stimuli in RGB.
            Lab_comp_rep         : (R, 3)         All comparison stimuli in Lab.
        """
        # Build dense comparison set around Lab_ref (vectorized over directions × radii)
        w_ref, w_ref_rep, w_comp_rep, rgb_ref, rgb_comp_rep, Lab_comp_rep = \
            self.setup_bruteforce_comp(Lab_ref, CLabTool)
    
        # Model choice probability for oddity task: P(choose X1 as odd)
        # Shape returned: (R,)
        pChoosingX1 = self._compute_pChoosingX1(w_ref_rep, w_comp_rep)
    
        # Reshape to (n_theta, ngrid_bruteforce): rows = directions, cols = radii
        base_shape = (self.params['n_theta'], self.params['ngrid_bruteforce'])
        pChoosingX1_org = pChoosingX1.reshape(base_shape)
    
        # For each direction, pick the radius whose performance is closest to the criterion target_pC
        # min_idx: (n_theta,) indices into the radial grid
        min_idx = np.argmin(np.abs(pChoosingX1_org - self.target_pC), axis=1)
        
        # Boundary check:
        # If the selected index lands on the first/last grid point, we should 
        # adjust the search bounds
        if np.any(np.isin(min_idx, [0, self.params['ngrid_bruteforce'] - 1])):
            raise ValueError('Hitting the boundary. Consider increasing the bounds.')
    
        # Gather W-space comparisons at threshold radius for each direction
        # w_comp_thres_temp: (n_theta, ngrid_bruteforce, 3)
        # w_comp_thres:      (n_theta, 3)
        w_comp_thres_temp = w_comp_rep.reshape(base_shape + (self.ndims_lab,))
        w_comp_thres = w_comp_thres_temp[list(range(self.params['n_theta'])), min_idx]
    
        # Convert threshold comparisons to display space (RGB) and then to Lab
        rgb_comp_thres = self.color_thres_data.W_unit_to_N_unit(w_comp_thres)
        Lab_comp_thres = CLabTool.convert_rgb_lab(rgb_comp_thres)[0].T
        
        #extract fixed, varied dims and vals
        vd = self.cfg.varied_dims
        fd = self.cfg.fixed_dim.value
        fv = Lab_ref[self.cfg.fixed_dim.value].item()
    
        # Fit an ellipse to the threshold points in the slice.
        #   center = Lab_ref projected onto the same 2D slice
        fitEll_scaled, fitEll_unscaled, ellParams, Lab_comp_thres_scaled = \
            fit_2d_isothreshold_contour(
                Lab_ref[vd],
                Lab_comp_thres[vd],    # 2×n_theta threshold samples (naming kept for compatibility)
                ellipse_scaler=ell_vis_scaler,
                flag_force_centered_ref = True
            )
            
        #insert the fixed dimension
        fitEll_scaled_f = self.fill_in_fixed_dim(fitEll_scaled, fd, fv)
        fitEll_unscaled_f = self.fill_in_fixed_dim(fitEll_unscaled, fd, fv)
        Lab_comp_thres_scaled_f = self.fill_in_fixed_dim(Lab_comp_thres_scaled, fd, fv)
            
        # debug plot
        if self.flag_debug_plot:
            self._debug_plot(Lab_comp_rep, w_comp_rep, rgb_comp_rep)
            self._debug_plot(Lab_comp_thres.T, w_comp_thres, rgb_comp_thres,
                             fit_ell_lab=fitEll_unscaled_f.T)
    
        return (
            fitEll_scaled_f, fitEll_unscaled_f, ellParams,
            Lab_comp_thres_scaled_f, Lab_comp_thres, w_ref, rgb_ref
        )
    
    def find_thres_CLab_givenDeltaE(self, ref_Lab, comp_Lab, target_deltaE, coloralg):
        """
        Given a Lab reference and a set of candidate Lab comparison points, find the
        comparison whose ΔE (per the chosen color-difference formula) is closest to
        `target_deltaE`.
    
        Args:
            ref_Lab       : (3,) Lab reference (L*, a*, b*).
            comp_Lab      : (N,3) candidate Lab comparisons.
            target_deltaE : float, target color difference (e.g., 2.0).
            coloralg      : str in {'CIE2000','CIE1994','CIE1976' (default)}.
    
        Returns:
            comp_Lab_thres : (3,) the comparison Lab with ΔE closest to target.
            min_idx        : int, index into `comp_Lab` of the chosen point.
        """
        
        num_pairs = comp_Lab.shape[0]
        
        ref_Lab_rep = np.tile(ref_Lab[None], (num_pairs, 1))
        deltaE = compute_thres_in_delta_batch(ref_Lab_rep, comp_Lab, coloralg)
    
        # Pick the candidate closest to the target ΔE.
        min_idx = np.argmin(np.abs(deltaE - target_deltaE))
        
        # Boundary check:
        # If the selected index lands on the first/last grid point, we should 
        # adjust the search bounds
        if np.any(np.isin(min_idx, [0, self.params['ngrid_bruteforce'] - 1])):
            raise ValueError('Hitting the boundary. Consider increasing the bounds.')
            
        comp_Lab_thres = comp_Lab[min_idx]
        return comp_Lab_thres, min_idx, deltaE[min_idx]
    
    def convert_Sig_2DisothresholdContour_oddity_batch_inLab(self, Lab_ref,
                                                             CLabTool, 
                                                             ell_vis_scaler=1,
                                                             flag_compute_thres_fd = True
                                                             ):
        """
        Batched isothreshold estimation (oddity task) on a CIELAB slice using WPPM.
    
        For each reference stimulus on a G1 × G2 Lab grid:
          - Generate comparison stimuli along n_theta directions and
            ngrid_bruteforce radii.
          - Find the radius where predicted performance is closest to target_pC.
          - Convert threshold comparisons to Lab and fit a 2D ellipse on the slice.
          - Store both unscaled and visualization-scaled contours.
    
        Optionally, also compute the threshold point along the *fixed* Lab dimension
        (i.e., orthogonal to the 2D slice).
    
        Parameters
        ----------
        Lab_ref : array, shape (G1, G2, 3)
            Grid of Lab reference stimuli defining the slice.
        CLabTool : object
            Helper for Lab ↔ RGB conversions.
        ell_vis_scaler : float, optional
            Visualization-only scaling factor for ellipses and threshold points.
        flag_compute_thres_fd : bool, optional
            Whether to also compute thresholds along the fixed Lab dimension.
    
        Side effects
        ------------
        Fills preallocated buffers created by `_init_model_pred_list()`, including:
          - fitEll_(un)scaled_WPPM      : fitted ellipse contours
          - Lab_thres_(un)scaled_WPPM   : raw threshold samples
          - params_ell                  : ellipse fit parameters
          - w_ref, rgb_ref              : reference stimuli (model & RGB space)
          - (optional) Lab_thres_at_fd  : fixed-dimension thresholds
        """
    
        # Initialize storage arrays and grid dimensions from Lab_ref
        self.num_grid_pts1, self.num_grid_pts2 = Lab_ref.shape[:2]
        self._init_model_pred_list()
    
        # Optionally initialize buffers for fixed-dimension thresholds
        if flag_compute_thres_fd:
            self._init_model_pred_list_for_fixed_dim()
    
        # Loop over all reference locations on the slice grid
        for i in range(self.num_grid_pts1):
            self.params_ell[i] = [[] for _ in range(self.num_grid_pts2)]
    
            for j in range(self.num_grid_pts2):
                print(f"Processing grid point: {np.round(Lab_ref[i, j], 2)}")
    
                # Compute 2D isothreshold contour and ellipse fit for this reference
                (self.fitEll_scaled_WPPM[i, j],
                 self.fitEll_unscaled_WPPM[i, j],
                 self.params_ell[i][j],
                 self.Lab_thres_scaled_WPPM[i, j],
                 self.Lab_thres_unscaled_WPPM[i, j],
                 self.w_ref[i, j],
                 self.rgb_ref[i, j]) = self.convert_Sig_2DisothresholdContour_oddity_inLab(
                    Lab_ref[i, j], CLabTool, ell_vis_scaler
                )
    
                # Optionally compute threshold along the fixed Lab dimension
                if flag_compute_thres_fd:
                    (self.Lab_thres_at_fd_unscaled_WPPM[i, j],
                     self.Lab_thres_at_fd_scaled_WPPM[i, j]) = \
                        self.compute_thres_for_fixed_dim_inLab(
                            Lab_ref[i, j],
                            CLabTool,
                            vis_scaler=ell_vis_scaler
                        )

    def convert_Sig_2DisothresholdContour_oddity_batch_LabPred(
        self, Lab_ref, CLabTool, target_deltaE=2.5, coloralg='CIE1994', ell_vis_scaler=1
    ):
        """
        Batched ΔE-based isothreshold contours on the same CIELAB slice.
    
        For each reference stimulus on a (G1 × G2) Lab grid:
          - Reuse the same densely sampled comparison points (same directions × radii)
            used in the WPPM oddity-task computation.
          - For each direction, select the comparison whose ΔE from the reference is
            closest to `target_deltaE` (using the chosen ΔE formula).
          - Fit a 2D ellipse to the resulting ΔE-threshold samples on the slice.
          - Store both unscaled and visualization-scaled versions in full 3D Lab
            (by filling in the fixed dimension).
    
        Parameters
        ----------
        Lab_ref : array, shape (G1, G2, 3)
            Grid of Lab reference stimuli defining the slice.
        CLabTool : object
            Helper for Lab ↔ RGB conversions (passed through to sampling helpers).
        target_deltaE : float
            Target ΔE radius for selecting threshold points (e.g., 2.5).
        coloralg : {'CIE1976', 'CIE1994', 'CIE2000'}
            ΔE formula used to compute color difference.
        ell_vis_scaler : float
            Visualization-only scaling factor applied to the fitted ellipse and samples.
    
        """
    
        # Loop over all reference locations on the slice grid
        for i in range(self.num_grid_pts1):
            self.params_ell_CLab[i] = [[] for _ in range(self.num_grid_pts2)]
    
            for j in range(self.num_grid_pts2):
                # Reference Lab center for this grid point
                Lab_ref_ij = Lab_ref[i, j]
    
                # Generate / reuse dense Lab comparisons along rays (directions × radii)
                # setup_bruteforce_comp returns several arrays; we only need Lab comparisons here.
                *_, Lab_comp_rep_ij = self.setup_bruteforce_comp(Lab_ref_ij, CLabTool)
    
                # Reshape to: (n_theta directions, ngrid_bruteforce radii, 3 Lab dims)
                Lab_comp_reshape = np.reshape(
                    Lab_comp_rep_ij,
                    (self.params['n_theta'], self.params['ngrid_bruteforce'], -1)
                )
    
                # For each direction, choose the sample whose ΔE is closest to target_deltaE
                for c in range(self.params['n_theta']):
                    Lab_comp_ijc = Lab_comp_reshape[c]  # (ngrid_bruteforce, 3)
    
                    # Returns:
                    #   - selected Lab threshold point (3,)
                    #   - (optional other outputs)
                    #   - achieved ΔE (may deviate slightly due to discretization)
                    self.Lab_thres_unscaled_CLab[i, j, :, c], _, self.deltaE[i, j, c] = \
                        self.find_thres_CLab_givenDeltaE(
                            Lab_ref_ij,
                            Lab_comp_ijc,
                            target_deltaE=target_deltaE,
                            coloralg=coloralg
                        )
    
                # Fit a 2D ellipse in the slice coordinates (two varying dimensions only)
                vd = self.cfg.varied_dims
                (fitEll_scaled_CLab_ij, fitEll_unscaled_CLab_ij,
                 self.params_ell_CLab[i][j], Lab_thres_scaled_CLab_ij) = \
                    fit_2d_isothreshold_contour(
                        Lab_ref_ij[vd],                          # 2D center
                        self.Lab_thres_unscaled_CLab[i, j, vd],  # 2×n_theta samples
                        ellipse_scaler=ell_vis_scaler,
                        flag_force_centered_ref=True
                    )
    
                # Convert fitted 2D results back to full 3D Lab by inserting the fixed dimension value
                fd = self.cfg.fixed_dim.value
                fv = Lab_ref_ij[fd].item()
    
                self.fitEll_scaled_CLab[i, j] = self.fill_in_fixed_dim(fitEll_scaled_CLab_ij, fd, fv)
                self.fitEll_unscaled_CLab[i, j] = self.fill_in_fixed_dim(fitEll_unscaled_CLab_ij, fd, fv)
                self.Lab_thres_scaled_CLab[i, j] = self.fill_in_fixed_dim(Lab_thres_scaled_CLab_ij, fd, fv)
                
#%%
def compute_thres_in_delta_batch(M1, M2, color_alg):
    """
    Compute color-difference thresholds (ΔE) between corresponding rows of M1 and M2.

    Parameters
    ----------
    M1, M2 : array-like, shape (N, 3)
        CIELAB coordinates [L*, a*, b*] for two sets of stimuli.
        Each row in M1 is compared with the corresponding row in M2.
    color_alg : str, optional
        Color-difference metric to use:
        'CIE1976', 'CIE1994', or 'CIE2000' (default: 'CIE2000').

    Returns
    -------
    dE : ndarray, shape (N,)
        ΔE values for each stimulus pair.
    """

    # Convert inputs to NumPy arrays
    M1 = np.asarray(M1)
    M2 = np.asarray(M2)

    # Select the ΔE function based on the chosen algorithm
    if color_alg == 'CIE2000':
        de_fn = delta_e_cie2000
    elif color_alg == 'CIE1994':
        de_fn = delta_e_cie1994
    elif color_alg == 'CIE1976':
        de_fn = delta_e_cie1976
    else:
        raise ValueError('unrecognized color difference algorithm.')

    # Initialize output array
    dE = np.zeros(M1.shape[0])

    # Loop over all stimulus pairs
    for i in range(M1.shape[0]):
        # Convert each Lab triplet to a LabColor object
        c1 = LabColor(*M1[i])
        c2 = LabColor(*M2[i])

        # Compute the color difference for this pair
        dE[i] = de_fn(c1, c2)

    return dE