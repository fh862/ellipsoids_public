#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:25:41 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import os
from core import oddity_task, chebyshev
from analysis.ellipses_tools import fit_2d_isothreshold_contour, ellParams_to_covMat,\
    covMat3D_to_2DsurfaceSlice, UnitCircleGenerate, ellParamsQ_to_covMat
from analysis.ellipsoids_tools import fit_3d_isothreshold_ellipsoid, UnitCircleGenerate_3D

#%% helping functions
class wishart_model_pred():
    def __init__(self, model, opt_params, w_init_key, opt_key, W_init, W_est,
                 Sigmas_noise_grid, color_thres_data = None, target_pC = 0.667,
                 **kwargs):        
        # Core model and dimensionality
        self.model               = model
        self.ndims               = model.num_dims
        
        # Fitted parameters and RNG keys used during optimization / initialization
        self.opt_params          = opt_params
        self.w_init_key          = w_init_key
        self.opt_key             = opt_key
        
        # Wishart process initial and best-fit weight matrix
        self.W_init              = W_init
        self.W_est               = W_est
        
        # Noise covariance matrices on the grid 
        self.Sigmas_noise_grid = Sigmas_noise_grid
        
        # Experimental / data-related configuration
        
        self.color_thres_data    = color_thres_data
        self.target_pC           = target_pC     
        
        # Default simulation and contour-fitting parameters
        self.params = {
            'simulation_func': oddity_task.simulate_oddity, #oddity task with no fixed reference stimulus
            'n_phi': 9,                      #number of grid points from north to south pole (only relevant for 3D)
            'n_theta':16,                    #number of angular steps to compute the ellipse
            'nPhi':100,                      #finer grid 
            'nTheta': 200,                   #finer grid
            'scaler_x1': 1,                  # optional: ellipses/ellipsoids were scaled up/down
            'ngrid_bruteforce': 1000,        # Number of brute force steps based on vector length
            'bds_bruteforce': [0.001, 0.25], # Bounds for vector length search
            'flag_force_centered_ref': True  #force the center to be at the ref when fitting ellipses / ellipsoids
            }
    
        # Allow caller to override any of the default parameters
        self.params.update(kwargs)
    
        # Derived quantities based on model and grid
        self._extract_grid_points()
        self._set_chromatic_dir()
        
    def _extract_grid_points(self):
        """
        Infer the spatial grid dimensions and covariance dimensionality
        from `self.W_est.shape`.
    
        Expected shapes:
            2D grid: (g1, g2, ndims_cov, ndims_cov + extra)
            3D grid: (g1, g2, g3, ndims_cov, ndims_cov + extra)
    
        Sets:
            self.num_grid_pts1, self.num_grid_pts2, [self.num_grid_pts3]
            self.ndims_cov
            self.grid_shape  # (g1, g2) or (g1, g2, g3)
        """
        # Split the leading grid dimensions from the trailing covariance dims
        *grid_shape, self.ndims_cov, _ = self.Sigmas_noise_grid.shape   # (g1, g2) or (g1, g2, g3)
    
        # Sanity check: the number of spatial grid dimensions must match the
        # dimensionality of the basis functions (self.ndims).
        assert len(grid_shape) == self.ndims, (
            "The number of grid dimensions must match the number of basis-function dimensions."
        )    
        
        # Unpack grid size; third dimension is optional
        self.num_grid_pts1, self.num_grid_pts2 = grid_shape[:2]
        self.num_grid_pts3 = grid_shape[2] if len(grid_shape) == 3 else None  
    
        # Ensure the covariance dimensionality matches the model specification
        assert self.ndims_cov == self.model.num_dims_cov, (
            f"num_dims_cov mismatch: W_est has {self.ndims_cov}, "
            f"model has {self.model.num_dims_cov}"
        )
    
        # Store the grid shape in its native form: (g1, g2) or (g1, g2, g3)
        self.grid_shape = tuple(grid_shape)
        
    def _set_chromatic_dir(self):
        """
        Precompute unit-length chromatic directions in model space.
    
        For ndims_cov = 2:
            - grid_color has shape (n_theta, 2)
              and samples directions on the unit circle.
    
        For ndims_cov = 3:
            - grid_color has shape (n_phi, n_theta, 3)
              and samples directions on the unit sphere.
        These directions are later scaled to find threshold-length vectors.
        """
        if self.ndims_cov == 2:
            # 2D: directions on the unit circle
            grid_color = UnitCircleGenerate(self.params['n_theta'])
        else:
            # 3D: directions on the unit sphere (elevation × azimuth)
            grid_color = UnitCircleGenerate_3D(
                self.params['n_theta'],
                self.params['n_phi']
            )
    
        self.params['grid_color'] = grid_color
        
    def _init_list_ellparams(self):
        """
        Initialize a nested list structure to store ellipse/ellipsoid parameters
        at each grid point.
    
        For ndims == 2:
            params_ell[i][j] holds the parameters at grid (i, j)
    
        For ndims == 3:
            params_ell[i][j][k] holds the parameters at grid (i, j, k)
        """
        if self.ndims == 2:
            # 2D grid: create a (num_grid_pts1 x num_grid_pts2) array of empty lists
            self.params_ell = [
                [[] for _ in range(self.num_grid_pts2)]
                for _ in range(self.num_grid_pts1)
            ]
        elif self.ndims == 3:
            # 3D grid: create a (num_grid_pts1 x num_grid_pts2 x num_grid_pts3)
            # array of empty lists
            self.params_ell = [
                [
                    [[] for _ in range(self.num_grid_pts3)]
                    for _ in range(self.num_grid_pts2)
                ]
                for _ in range(self.num_grid_pts1)
            ]
        else:
            # Only 2D or 3D grids are currently supported
            raise ValueError(f"Unsupported ndims={self.ndims}")
        
    def _init_model_pred_list(self):
        """
        Allocate arrays for model predictions at each grid point.
    
        For ndims_cov = 2 (ellipse in 2D subspace):
            - fitEll_scaled[g1, g2, (g3), dim, nTheta]
                Dense samples along the fitted elliptical contour.
    
        For ndims_cov = 3 (ellipsoid in 3D subspace):
            - fitEll_scaled[g1, g2, g3, dim, nPhi * nTheta]
                Dense samples on the fitted ellipsoidal surface.
    
        """
        # Initialize arrays that store the (scaled) contour points
        if self.ndims_cov == 2:
            # 2D covariance: elliptical contours
            self.fitEll_scaled = np.full(
                self.grid_shape + (self.ndims_cov, self.params['nTheta']),  # dense points on ellipse
                np.nan,
            )
    
        elif self.ndims_cov == 3:
            # 3D covariance: ellipsoidal contours
            self.fitEll_scaled = np.full(
                self.grid_shape + (self.ndims_cov, self.params['nPhi'] * self.params['nTheta']),  # dense points on ellipsoid
                np.nan,
            )
    
        # Initialize array for predicted covariance matrices at each grid point.
        #
        # Examples:
        #   - 3D stimulus, 2×2 covariance at each location in a slice:
        #         Sigmas_thres_grid.shape -> (g1, g2, g3, 2, 2)
        #   - 3D stimulus, 3×3 covariance at each location in the cube:
        #         Sigmas_thres_grid.shape -> (g1, g2, g3, 3, 3)
        #   - 2D stimulus, 2×2 covariance:
        #         Sigmas_thres_grid.shape -> (g1, g2, 2, 2)
        #
        # By construction, ndims_cov ≤ ndims, so we will never have something like (g1, g2, 3, 3)
        # for a 2D stimulus.
        self.Sigmas_thres_grid = np.full(
            self.grid_shape + (self.ndims_cov, self.ndims_cov),
            np.nan,
        )
    
        # Unscaled contour points (same shape as fitEll_scaled)
        self.fitEll_unscaled = np.full(self.fitEll_scaled.shape, np.nan)
    
    def _set_up_grid_search(self):
        """
        Sets up a grid for a brute force search over a predefined range of vector lengths. 
        This method generates a linearly spaced array of vector lengths within specified 
        bounds, which will be used to determine optimal vector lengths in chromatic direction 
        computations for perceptual threshold tasks.
        
        The bounds and number of points in the grid are defined in the class's parameters.
        
        Returns:
            np.ndarray: A 1D array of vector lengths that are linearly spaced between the 
                        lower and upper bounds defined in the class's parameters.
    
        """
        # Generate a linearly spaced array of vector lengths within the specified bounds.
        vecLength_grid = np.linspace(*self.params['bds_bruteforce'], 
                                     self.params['ngrid_bruteforce'])
            
        return vecLength_grid
    
    def _compute_pChoosingX1(self, w_ref, w_comp):
        Uref          = self.model.compute_U(self.W_est, w_ref)
        U1            = self.model.compute_U(self.W_est, w_comp)
        # Simulate the oddity task trial and compute the signed difference
        pChoosingX1   = oddity_task.oddity_prediction(
                                    (w_ref, w_comp, Uref, U1),
                                    jax.random.split(self.opt_key, num = w_comp.shape[0]),
                                    self.opt_params['mc_samples'], 
                                    self.opt_params['bandwidth'],
                                    self.model.diag_term,
                                    self.params['simulation_func'])
        return pChoosingX1
    
    def _convert_Sig_2DisothresholdContour_oddity(self, w_ref, vecLength_test):
        """
        Simulates an oddity task to determine chromatic differences at a perceptual 
        threshold, and computes isothreshold contours for comparison stimuli. The 
        contours represent the boundaries at which chromatic differences are just 
        noticeable.
        
        Notes
        -----
        This method employs JAX's vmap for efficient computation across multiple 
        simulations. Extensive reshaping of input data structures is performed to 
        fit the requirements of vectorized operations.
    
        Parameters
        ----------
        w_ref : array_like, shape (ndims,)
            Reference stimulus in model space. Only the first `ndims_cov` dimensions
            are used for defining chromatic directions; remaining dimensions (if any)
            are treated as fixed coordinates.
            
        vecLength_test: A 1D array with length equal to nSteps_bruteforce, representing 
            the magnitudes of the vector lengths to test in each chromatic direction.
    
        Returns
        -------
        fitEll_scaled: A 2D array (2, nThetaEllipse) containing the coordinates 
            of the scaled fitted ellipse.
            
        fitEll_unscaled: A 2D array (2, nThetaEllipse) containing the 
            coordinates of the unscaled fitted ellipse.
        
        w_comp_scaled: A 2D array (2, numDirPts) containing the scaled w 
            components of the comparison stimuli. (*the variable naming is bad here!)
                
        [xCenter, yCenter, majorAxis, minorAxis, theta]: parameters of ellipses
        
        """
        
        # Calculate the total number of simulations based on grid and brute force steps.
        # ---- if we take the default params value, nRepeats = 16 x 1000 = 16000
        nRepeats = self.params['n_theta'] * self.params['ngrid_bruteforce']
        
        # Prepare reference values repeated across all simulations.
        # ---- shape of w_ref_rep is (16000, 2)
        # Only the first ndims_cov coordinates are used (2D covariance subspace).
        w_ref_rep   = jnp.tile(w_ref[:self.ndims_cov],(nRepeats,1))  
        
        # Prepare chromatic direction data formatted for vectorized operations.
        # ---- shape of self.params['grid_theta_xy'] is (2,16)
        # ---- shape of vecDir_org is (2, 16, 1000)
        vecDir_org    = np.tile(self.params['grid_color'][:,:,None],
                                (1,1, self.params['ngrid_bruteforce']))
        # ---- shape of vecDir_trans is (2, 16, 1000)
        vecDir_trans  = np.transpose(vecDir_org, [1,2,0])
        # ---- shape of vecDir_rep is (16000, 2)
        vecDir_rep    = vecDir_trans.reshape((nRepeats, self.ndims_cov))
        
        # Prepare vector lengths formatted for simulations, matching chromatic direction formatting.
        # ---- shape of vecLength_org is (16,1000)
        vecLength_org = np.tile(vecLength_test, (self.params['n_theta'], 1))
        # ---- shape of vecLength_reshape is (16000, 1)
        vecLength_reshape = np.reshape(vecLength_org,(nRepeats,1))
        # ---- shape of vecLength_rep is (16000, 2)
        vecLength_rep = np.tile(vecLength_reshape,(1, self.ndims_cov))
        
        # Calculate comparison stimuli w values based on reference and direction magnitudes.
        w_comp_rep  = w_ref_rep + vecDir_rep * vecLength_rep 
    
        # If the covariance lives in a lower-dimensional subspace (ndims_cov < ndims),
        # append the remaining (fixed) coordinates so that we can evaluate the full model.
        if self.ndims_cov < self.ndims:
            # Remaining coordinates are treated as fixed across all simulations.
            # Shape: (nRepeats, ndims - ndims_cov)
            w_ref_last_dim = np.ones(
                (nRepeats, self.ndims - self.ndims_cov)
            ) * w_ref[self.ndims_cov:][None, :]
    
            w_ref_rep_full = np.hstack((w_ref_rep, w_ref_last_dim))
            w_comp_rep_full = np.hstack((w_comp_rep, w_ref_last_dim))
        else:
            # Full dimensionality already matches
            w_ref_rep_full = w_ref_rep
            w_comp_rep_full = w_comp_rep
        
        # Comparison points in the ndims_cov-dimensional subspace.
        w_comp_rep = w_ref_rep + vecDir_rep * vecLength_rep
        
        # Compute the U matrices for reference and comparison stimuli.
        pChoosingX1 = self._compute_pChoosingX1(w_ref_rep_full, w_comp_rep_full)
            
        #reshape the probability of choosing x1 from
        #(16000, ) to (16, 1000)
        pChoosingX1_org = pChoosingX1.reshape((self.params['n_theta'],
                                               self.params['ngrid_bruteforce']))
        # Identify the vector length at the perceptual threshold for each direction.
        min_idx = np.argmin(np.abs(pChoosingX1_org - self.target_pC), axis = 1) 
        recover_vecLength = vecLength_org[np.arange(self.params['n_theta']),min_idx]
        
        # Compute and store the comparison w component estimate
        recover_w_comp_est = w_ref[:self.ndims_cov,None] + \
            self.params['grid_color'] * recover_vecLength[None,:]
        
        # Fit ellipses to the chromatic differences at the perceptual threshold.
        # fitEll_params: [xCenter, yCenter, majorAxis, minorAxis, theta]
        fitEll_scaled, fitEll_unscaled, fitEll_params, _ = \
                fit_2d_isothreshold_contour(
                    w_ref[:self.ndims_cov], 
                    recover_w_comp_est,
                    ellipse_scaler = self.params['scaler_x1'],
                    nTheta = self.params['nTheta'],
                    flag_force_centered_ref = self.params['flag_force_centered_ref']
                    )
    
        return fitEll_scaled, fitEll_unscaled, fitEll_params
            
    def _convert_Sig_3DisothresholdContour_oddity(self, w_ref, vecLength_test):
    
        # Number of total simulations
        nRepeats = self.params['n_phi'] * self.params['n_theta'] * self.params['ngrid_bruteforce']
        
        # Repeat the vector w_ref by nRepeats; final size = (nRepeats, 3)
        w_ref_rep = jnp.tile(w_ref[:self.ndims_cov], (nRepeats, 1))
    
        # Chromatic directions 
        vecDir_org = np.tile(self.params['grid_color'][:,:,None,:],
                             (1,1,self.params['ngrid_bruteforce'],1))
        
        vecDir_rep = vecDir_org.reshape((nRepeats, self.ndims_cov))
        # vector length (how far do we go along each chromatic direction)
        vecLength_org = np.tile(vecLength_test, (self.params['n_phi'], 
                                                 self.params['n_theta'], 1))
        vecLength_rep = np.tile(np.reshape(vecLength_org, (nRepeats,1)),
                                (1, self.ndims_cov))
        # comparison stimuli
        w_comp_rep  = w_ref_rep + vecDir_rep * vecLength_rep 
        # Compute the U matrices for reference and comparison stimuli.
        pChoosingX1 = self._compute_pChoosingX1(w_ref_rep, w_comp_rep)
        
        #reshape the probability of choosing x1 from
        #(nRepeats, 3) to (n_phi, n_theta, nSteps_bruteforce, 3)
        pChoosingX1_org = pChoosingX1.reshape((self.params['n_phi'],
                                               self.params['n_theta'],
                                               self.params['ngrid_bruteforce']))
        min_idx = np.argmin(np.abs(pChoosingX1_org - self.target_pC), axis = 2) 
        i, j = np.ogrid[:self.params['n_phi'], :self.params['n_theta']]
        recover_vecLength = vecLength_org[i,j,min_idx]
              
        # Compute and store the comparison component estimate
        recover_w_comp_est = w_ref[None,None,:self.ndims_cov] + \
            self.params['grid_color'] * recover_vecLength[:,:,None]
            
        #fit an ellipse to the estimated comparison stimuli
        fitEll_scaled, fitEll_unscaled, ellipsoidParams, _ =\
            fit_3d_isothreshold_ellipsoid(w_ref[:self.ndims_cov], 
                                          recover_w_comp_est, 
                                          nTheta = self.params['nTheta'],
                                          nPhi = self.params['nPhi'],
                                          ellipsoid_scaler = self.params['scaler_x1'],
                                          flag_force_centered_ref = self.params['flag_force_centered_ref']
                                          )
        return fitEll_scaled, fitEll_unscaled, ellipsoidParams        
    
    def _convert_Sig_2DisothresholdContour_oddity_batch(self, w_ref):
        """ 
        Processes a batch of reference w values to compute isothreshold contours
        (ellipses or ellipsoids) for an oddity task.
    
        This method loops over all spatial grid points and, for each point,
        fits a contour based on the simulated oddity thresholds.
    
        Args:
            w_ref (np.ndarray):
                Reference stimulus in model space, with shape
                    (ndims_cov, *grid_shape)
                where grid_shape is (g1, g2) for 2D or (g1, g2, g3) for 3D.
    
        Side effects:
            Fills in:
                - self.fitEll_scaled[grid_idx]
                - self.fitEll_unscaled[grid_idx]
                - self.w_comp_scaled[grid_idx]
                - self.params_ell[...]
        """
    
        # Loop over all grid indices: (i, j) for 2D or (i, j, k) for 3D
        for grid_idx in np.ndindex(*self.grid_shape):
            # Extract w_ref at this grid point: shape (ndims_cov,)
            w_ref_ijk = w_ref[grid_idx]
    
            print(f"Processing grid point {grid_idx}: {np.round(w_ref_ijk, 2)}")
    
            # Determine the vector lengths to test using a brute-force grid search.
            vecLength_test = self._set_up_grid_search()
    
            # Fit contour for this grid point
            fit_scaled, fit_unscaled, params_ell = \
                self._convert_Sig_2DisothresholdContour_oddity(
                    w_ref_ijk, vecLength_test
                )
    
            # Save results into the arrays (handles both 2D and 3D via tuple indexing)
            self.fitEll_scaled[grid_idx]   = fit_scaled
            self.fitEll_unscaled[grid_idx] = fit_unscaled
    
            # Save ellipse/ellipsoid parameters into params_ell (nested lists)
            if self.ndims == 2:
                i, j = grid_idx
                self.params_ell[i][j] = params_ell
            elif self.ndims == 3:
                i, j, k = grid_idx
                self.params_ell[i][j][k] = params_ell
                            
            #check if there are nan's. If there is, throw an error
            #and then probably adjust the bds_bruteforce to fix the problem
            #if np.isnan(self.params_ell[i][j]['radii']).any():
            #    raise ValueError("Input array contains NaN values.")
                
            # Convert the ellipsoid parameters to covariance matrices.
            # Note: this covariance matrix has a similar but not identical shape compared to Sigma_noise_grid
            self.Sigmas_thres_grid[grid_idx] = ellParamsQ_to_covMat(*params_ell[2:])
    
    def _convert_Sig_3DisothresholdContour_oddity_batch(self, w_ref):        
        """
        Similar to the 2D method, but processes a batch of 3D reference values in the 
        model space.
        
        See `_convert_Sig_2DisothresholdContour_oddity_batch` for detailed comments.
        
        This function extends the 2D processing to a third dimension, fitting ellipsoids
        across a 3D grid of comparison stimuli based on perceptual thresholds.
        """
        
        # Loop over all grid points (i, j, k) in the 3D grid
        for grid_idx in np.ndindex(*self.grid_shape):
            i, j, k = grid_idx
        
            # Extract the specific values in the model space for the current grid point
            w_ref_scaled_ijk = jnp.array(w_ref[grid_idx])
        
            print(f"Processing grid point {grid_idx}: {np.round(w_ref_scaled_ijk, 2)}")
        
            # Determine the vector lengths to test using a brute-force grid search
            vecLength_test = self._set_up_grid_search() 
        
            # Fit ellipsoids to the estimated comparison stimuli
            (
                self.fitEll_scaled[grid_idx],
                self.fitEll_unscaled[grid_idx],
                self.params_ell[i][j][k],
            ) = self._convert_Sig_3DisothresholdContour_oddity(
                w_ref_scaled_ijk,
                vecLength_test,
            )
        
            # Optional NaN check on parameters
            # if np.isnan(self.params_ell[i][j][k]['radii']).any():
            #     raise ValueError("Ellipse parameter 'radii' contains NaN values.")
        
            # Convert the ellipsoid parameters to covariance matrices.
            # Note: this covariance matrix has a similar but not identical shape compared to Sigma_noise_grid
            self.Sigmas_thres_grid[grid_idx] = ellParams_to_covMat(
                self.params_ell[i][j][k]['radii'],
                self.params_ell[i][j][k]['evecs'],
            )
        
        # Compute the 2D ellipse slices from the 3D covariance matrices
        # for both ground truth and predictions.
        self.pred_slice_2d_ellipse = covMat3D_to_2DsurfaceSlice(self.Sigmas_thres_grid)
                            
    #%%
    def convert_Sig_Threshold_oddity_batch(self, w_ref):
        """
        Dispatch method for computing isothreshold contours for the oddity task.
    
        This method:
            1. Initializes storage for model predictions and fitted contour parameters.
            2. Chooses the appropriate routine (2D vs 3D) based on `ndims_cov`.
            3. Processes all reference points in `w_ref` to obtain isothreshold
               contours (ellipses or ellipsoids) in model space.
    
        Args:
            w_ref (np.ndarray):
                Reference stimuli in model space. Expected shape is
                (ndims_cov, *grid_shape), where grid_shape is (g1, g2) or (g1, g2, g3),
                consistent with how the W field is defined.
    
        Returns:
            None
                Results are written into instance attributes, e.g.:
                    - self.fitEll_scaled
                    - self.fitEll_unscaled
                    - self.params_ell
                    - self.Sigmas_thres_grid (and derived quantities)
    
        Notes:
            - `ndims` is the dimensionality of the stimulus space.
            - `ndims_cov` is the dimensionality of the covariance matrix.
    
            Typically, `ndims_cov == ndims`, but it is also possible to work with a
            lower-dimensional covariance. For example:
                - A 3D stimulus space (ndims = 3) with a 2×2 covariance (ndims_cov = 2)
                  defined on each slice of a 2D plane in that 3D space.
            In that case, the contour is computed in the lower-dimensional covariance
            subspace, but still indexed over the full spatial grid.
        """
        
        # Initialize model prediction lists for storing the ellipse and the data in the model space.
        self._init_model_pred_list()
        self._init_list_ellparams()
    
        if self.ndims_cov == 2:
            self._convert_Sig_2DisothresholdContour_oddity_batch(w_ref)
        elif self.ndims_cov == 3:
            self._convert_Sig_3DisothresholdContour_oddity_batch(w_ref)
        else:
            print('Currently do not support higher dimensionality of cov matrix.')
    
    #%%
    @staticmethod
    def compute_Mahalanobis_distance_one_pair(xref, x1, Uref, U1):
        """
        Computes the Mahalanobis distance between two points with associated covariance matrices.
    
        Parameters:
        ----------
        xref : jnp.ndarray
            A reference point of shape (ndims,).
        x1 : jnp.ndarray
            A comparison point of shape (ndims,).
        Uref : jnp.ndarray
            Covariance matrix of the reference point, shape (ndims, ndims).
        U1 : jnp.ndarray
            Covariance matrix of the comparison point, shape (ndims, ndims).
    
        Returns:
        -------
        d_M : float
            The Mahalanobis distance between the two points considering their covariance matrices.
    
        Notes:
        -----
        - The Mahalanobis distance incorporates the covariance structure of the points, making it
          suitable for measuring distances in the presence of correlated variables.
        - The computation is based on the combined covariance matrix:
            sigma_avg = (Uref + U1) / 2
        - If the covariance matrices are singular or ill-conditioned, the method may fail unless
          handled explicitly (e.g., using pseudo-inverse instead of inverse).
        """
        # Compute the average covariance matrix between the two points
        sigma_avg = (Uref + U1) / 2
        
        # Compute the difference vector between the two points
        delta = xref - x1
        
        # Compute the inverse of the average covariance matrix
        sigma_avg_inv = jnp.linalg.inv(sigma_avg)
        
        # Compute the Mahalanobis distance using the formula: sqrt(delta.T @ sigma_avg_inv @ delta)
        d_M = jnp.sqrt(delta.T @ sigma_avg_inv @ delta)
        
        return d_M
    
    def compute_Mahalanobis_distance_batch(self, xref_all, x1_all, Uref_all, U1_all):
        """
        Computes the Mahalanobis distances for a batch of reference and comparison points,
        each with their associated covariance matrices.
        
        Parameters:
        ----------
        xref_all : jnp.ndarray
            Batch of reference points, shape (N, ndims).
        x1_all : jnp.ndarray
            Batch of comparison points, shape (N, ndims).
        Uref_all : jnp.ndarray
            Batch of covariance matrices for the reference points, shape (N, ndims, ndims).
        U1_all : jnp.ndarray
            Batch of covariance matrices for the comparison points, shape (N, ndims, ndims).
        
        Returns:
        -------
        None
            The computed Mahalanobis distances are stored in the instance attribute:
            self.mahalanobis_distances, which has a shape of (N,).
        
        Notes:
        -----
        - This method uses `jax.vmap` to vectorize the computation of Mahalanobis distances
          across all input batches, making it highly efficient for large datasets.
        - The resulting distances are stored as an instance attribute for later use.
        - Assumes that all inputs are valid and have consistent dimensions.
        """
        # Vectorized computation of Mahalanobis distances across the batch
    
        self.mahalanobis_distances = jax.vmap(
            self.compute_Mahalanobis_distance_one_pair,
            in_axes=(0, 0, 0, 0)
        )(xref_all, x1_all, Uref_all, U1_all)
    
#%%
def rerun_model_pred_wExisting_model(grid, model_pred, color_thres_data, 
                                     ngrid_bruteforce = 1000, 
                                     bds_bruteforce = [0.0005, 0.25]):
    """
    Recomputes model predictions on a new grid using an existing Wishart model 
    and its fitted parameters.
    
    Parameters
    ----------
    grid : ndarray (N x N x 2 for 2d; N x N x N x 3 for 3D)
        The stimulus grid over which to recompute model predictions.
    model_pred : wishart_model_pred object
        The previously fitted Wishart model containing W_est, keys, and opt_params.
    color_thres_data : object
    ngrid_bruteforce : int, optional
        Number of samples used for brute-force threshold search (default: 1000).
    bds_bruteforce : list of float, optional
        Lower and upper bounds for the brute-force threshold search (default: [0.0005, 0.25]).
    
    Returns
    -------
    model_pred_new : wishart_model_pred object
        Updated model prediction object based on the new grid.
    grid_trans : ndarray
        The transposed version of the grid, used for batch processing.
    """

    model = model_pred.model
    Sigmas_noise_grid = model.compute_Sigmas(model.compute_U(model_pred.W_est, grid))
    model_pred_new = wishart_model_pred(model, model_pred.opt_params,
                                        model_pred.w_init_key,
                                        model_pred.opt_key, model_pred.W_init,
                                        model_pred.W_est, Sigmas_noise_grid,
                                        color_thres_data,
                                        target_pC = model_pred.target_pC,
                                        ngrid_bruteforce = ngrid_bruteforce,
                                        bds_bruteforce = bds_bruteforce,
                                        simulation_func = model_pred.params['simulation_func'])
    model_pred_new.convert_Sig_Threshold_oddity_batch(grid)   
    
    return model_pred_new
        





