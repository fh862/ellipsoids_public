#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:48:25 2024

@author: fangfang

The class `fit_PMF_MOCS_trials` analyzes psychophysical data collected using the 
Method of Constant Stimuli (MOCS) in either a 3AFC oddity task or a 2AFC 
supra-threshold judgment task. 

- In the **3AFC oddity task**, three stimuli are presented (two references and 
  one comparison). The participant’s task is to identify the odd one out. The 
  reference is fixed while the comparison stimulus is varied along a specified 
  chromatic direction.

- In the **2AFC supra-threshold judgment task**, three stimuli are presented 
  (reference, comp#1, comp#2). The reference is always positioned at the top. 
  The participant judges which comparison is more different from the reference. 
  Comp#1 has a fixed surface color, while comp#2 varies along a specified 
  chromatic direction.

This class is flexible in the choice of distance metric used to fit the 
psychometric function. The Weibull psychometric function can be fit as a 
function of one of the following distances between the reference and the 
comparison stimulus:
    1. Euclidean distance
    2. Mahalanobis distance
    3. Geodesic distance
    
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from tqdm import trange
import copy
import os
# Optional import: geodesics (not needed on all machines)
try:
    from core import geodesics
    HAS_GEODESICS = True
except Exception as e:
    geodesics = None
    HAS_GEODESICS = False
    print(f"[WARN] Optional module `core.geodesics` not available. "
          f"Geodesic-related functionality will be disabled. ({e})")

#%%
class fit_PMF_MOCS_trials():
    def __init__(self, nDim: int, stim: np.ndarray, resp: np.ndarray, nLevels: int,
                 guess_rate: float = 1/3, target_pC: float = 0.667, 
                 dist_metric: str = 'Euclidean', flag_pad_dist0 = True, **kwargs: dict):
        """
        Fit a psychometric function to MOCS (Method of Constant Stimuli) trials 
        and optionally perform bootstrapping to estimate confidence intervals.
        
        Parameters
        ----------
        nDim : int
            Number of dimensions of the stimulus space (e.g., 2 for 2D, 3 for 3D).
        stim : np.ndarray
            Stimulus array of shape (N, nDim), where N is the number of trials. 
            Each row corresponds to one trial. Stimuli are expected to be centered 
            at the origin. The order matches the experimental presentation sequence 
            and is not pre-sorted.
        resp : np.ndarray
            Array of binary responses (0 or 1) for each trial.
        nLevels : int
            Number of stimulus levels tested along the chosen chromatic direction.
        guess_rate : float, optional
            Lapse/guess rate:
              - 1/3 for 3AFC oddity tasks (chance level is 1 out of 3).
              - 0 for 2AFC supra-threshold judgments, where the dependent measure 
                is the probability of choosing comp#2 as more different. 
                At the point where comp#2 matches the reference, the probability is 0.
        target_pC : float, optional
            Target proportion correct for threshold estimation:
              - Default is 0.667 for 3AFC oddity tasks.
              - 0.5 for 2AFC supra-threshold tasks (corresponding to the PSE, 
                where comp#2 is perceived as equally different from the reference 
                as comp#1).
        dist_metric : str, optional
            Distance metric to use when fitting the psychometric function:
              - 'Euclidean'   : L2 norm of the stimulus vector relative to the origin.
              - 'Mahalanobis' : Mahalanobis distance between reference and 
                                comparison stimuli, using covariance structure 
                                from a Wishart model.
              - 'Geodesic'    : (if implemented) Geodesic distance along a 
                                curved manifold in stimulus space.
        flag_pad_dist0: bool, default=True
            Whether to pad a single filler trial for distance = 0.
            Notes:
            - 3AFC oddity: Typically set to True. We often append a single
              (distance=0, p=guess_rate) filler to anchor the PMF at chance, but
              that point is not an actual trial and should be excluded from resampling.
            - 2AFC supra-threshold: Usually set to False, since trials at
              distance=0 can be observed (no feedback), so no filler is needed.
        **kwargs : dict
            Additional optional arguments:
              - nInitializations : int
                  Number of random initializations used in parameter fitting.
              - bounds : list of tuples
                  Lower and upper bounds for parameters of the psychometric function.
              - nGridPts : int
                  Number of points for reconstructing and plotting the fitted 
                  psychometric function.
        """

        self.nDim       = nDim #whether we are in 2D W space or 3D
        self.stim       = stim #stimulus has to be centered to the origin
        self.resp       = resp
        self.target_pC  = target_pC
        self.nLevels    = nLevels
        self.guess_rate = guess_rate
        self.dist_metric = dist_metric
        self.flag_pad_dist0 = flag_pad_dist0
        #Validate inputs to ensure correctness
        self._validate_inputs()
        self.unique_stim, self.nTrials_perLevel, self.pC_perLevel, self.stim_org,\
            self.resp_org = self._get_unique_stim()
        if self.nLevels != self.unique_stim.shape[0]:
            raise ValueError("The number of unique stimuli does not match the number of input levels!")
                        
        # Set number of initializations from kwargs, or use default values if not provided
        self.nInitializations = kwargs.get('nInitializations', 20)  
        # Set bounds from kwargs, or use default bounds if not provided
        self.bounds   = kwargs.get('bounds', [(1e-4, 0.5), (1e-1, 5)]) 
        self.nGridPts = kwargs.get('nGridPts', 1200)
                
    def _validate_inputs(self):
        """
        Validate stimuli and responses for dimensionality, alignment, centering, 
        and distance metric consistency.
        """
        # --- Check stimulus dimensionality ---
        if self.stim.shape[1] != self.nDim:
            raise ValueError(
                f"Stimulus dimensionality mismatch: expected {self.nDim}, "
                f"but got {self.stim.shape[1]}"
            )
    
        # --- Check matching number of trials and responses ---
        if self.stim.shape[0] != self.resp.shape[0]:
            raise ValueError(
                f"Trial count mismatch: {self.stim.shape[0]} stimuli but "
                f"{self.resp.shape[0]} responses"
            )
    
        # --- Check if stimuli are centered on the origin ---
        # At least one stimulus should lie on each axis (within tolerance)
        tol = 1e-6
        has_x_axis = np.any(np.isclose(self.stim[:, 0], 0.0, atol=tol))
        has_y_axis = np.any(np.isclose(self.stim[:, 1], 0.0, atol=tol))
        if not (has_x_axis and has_y_axis):
            raise ValueError(
                "Stimuli are not centered on the origin: expected at least one "
                "stimulus near x = 0 and one near y = 0 (within tolerance)."
            )
    
        # --- Check distance metric validity ---
        if self.dist_metric not in {"Euclidean", "Mahalanobis", "Geodesic"}:
            raise ValueError(
                f"Unsupported distance metric: '{self.dist_metric}'. "
                "Currently supported: 'Euclidean', 'Mahalanobis', 'Geodesic'."
            )

    def _get_unique_stim(self, tol = 1e-8):
        """
        Generalized method to retrieve unique stimulus groups and compute statistics at each unique level.
        
        This method groups stimulus values in the first column within the specified tolerance
        and aggregates associated values in the other columns. Additionally, it calculates
        the number of trials (`nTrials_perLevel`) and the proportion correct (`pC_perLevel`) for 
        each unique stimulus group.
        
        Parameters:
        - tol (float): The tolerance for grouping stimulus values in the first column. 
                       Values within this tolerance are considered identical.
        
        Returns:
        - unique_stim (np.ndarray): An array of shape (M, K), where M is the number of unique groups 
                                     and K is the number of dimension (self.nDims). Each row 
                                     contains a unique stimulus group.
        - nTrials_perLevel (np.ndarray): A 1D array of shape (M,) representing the number of trials for 
                                         each unique stimulus group.
        - pC_perLevel (np.ndarray): A 1D array of shape (M,) representing the proportion correct for 
                                     each unique stimulus group.
        """
        # Extract the first column for grouping (dim1)
        stim_dim1 = self.stim[:, 0]
        
        # Scale and round values in the first column for grouping
        rounded_stim_dim1 = np.round(stim_dim1 / tol) * tol
        
        # Identify unique values in the rounded first column
        unique_stim_dim1 = np.sort(np.unique(rounded_stim_dim1))
        num_unique = len(unique_stim_dim1)  # Number of unique groups
        
        # Initialize arrays for results
        unique_stim      = np.full((num_unique, self.stim.shape[1]), np.nan)  # Unique rows for all columns
        nTrials_perLevel = np.full(num_unique, np.nan)                        # Number of trials per level
        pC_perLevel      = np.full(num_unique, np.nan)                        # Proportion correct per level
        stim_org         = np.full(self.stim.shape, np.nan)                   # undo shuffling
        resp_org         = np.full(self.resp.shape, np.nan)
        
        # Loop through each unique value in the first column
        idx_counter = 0
        for n, value in enumerate(unique_stim_dim1):
            # Find indices where the current unique value matches in the rounded array
            idx_n = np.where(value == rounded_stim_dim1)[0]
            
            # Store the aggregated values for each column
            unique_stim[n, 0] = value  # First column uses the unique rounded value
            for col in range(1, self.stim.shape[1]):  # For other columns, take the first occurrence
                unique_stim[n, col] = self.stim[idx_n[0], col]
            
            # Count the number of trials corresponding to the current unique stimulus group
            nTrials_perLevel[n] = len(idx_n)
            
            # organize the trials
            stim_org[int(idx_counter): int(idx_counter + nTrials_perLevel[n])] = unique_stim[n]
            resp_org[int(idx_counter): int(idx_counter + nTrials_perLevel[n])] = self.resp[idx_n]
            idx_counter += nTrials_perLevel[n]
            
            # Compute the proportion correct for the current unique stimulus group
            pC_perLevel[n] = np.sum(self.resp[idx_n]) / nTrials_perLevel[n]
        # Return the unique stimulus groups, number of trials, and proportion correct
        return unique_stim, nTrials_perLevel.astype(int), pC_perLevel, stim_org, resp_org
    
    def _fit_PsychometricFunc(self, dist_xref_x1, resp_org):
        """
        Fit a Weibull psychometric function to the data using maximum likelihood estimation (MLE).

        The function minimizes the negative log-likelihood (nLL) of the observed responses by 
        optimizing the Weibull parameters. To reduce the risk of converging to local minima, 
        the optimization is repeated with multiple random initializations within the parameter bounds.

        Parameters
        ----------
        dist_xref_x1 : np.ndarray
            1D array of distances between the reference and comparison stimuli.
            The distance metric can be Euclidean, Mahalanobis or Geodesic
        resp_org : np.ndarray
            1D array of binary responses (0 or 1) for each trial.
            
        Note: Both inputs are arranged in blocks — the same stimulus distance is 
        repeated for its corresponding N trials, followed by the next distance value, 
        and so on.

        Returns
        -------
        bestfit_result : OptimizeResult
            The optimization result containing:
            - x : Best-fitting parameters [threshold (a), steepness (b)]
            - fun : The minimized negative log-likelihood (bestfit_nLL)
            - success : Boolean indicating if the optimization converged
            - message : Description of the optimizer's exit status

        """
        
        # Set an extremely high nLL that can be easily defeated
        bestfit_nLL = 1000
        bestfit_result = None
    
        # Perform multiple random initializations
        for n in range(self.nInitializations):       
            # Draw random initialization within the specified bounds
            initial_params_n = [np.random.uniform(*self.bounds[0]), 
                                np.random.uniform(*self.bounds[1])]
            
            # Perform optimization using `minimize`
            result = minimize(
                self.nLL_Weibull,                # Objective function
                initial_params_n,                # Initial parameters
                args=(dist_xref_x1, resp_org), 
                method='L-BFGS-B',               # Optimization method
                bounds=self.bounds               # Parameter bounds
            )
            
            # Update best fit if this result is successful and has a lower nLL
            if result.success and result.fun < bestfit_nLL:
                bestfit_nLL = result.fun
                bestfit_result = result
            
        # Check if the optimization was successful
        if bestfit_result is not None:
            return bestfit_result
        else:
            # Raise an error if all attempts fail
            raise ValueError("Optimization failed: " + result.message)
    
    def _find_stim_at_targetPC(self, predPC):
        """
        Find the stimulus value corresponding to the target proportion correct (target_pC).
        
        This method identifies the stimulus value from the fine grid of predicted probabilities
        (`predPC`) that is closest to the target proportion correct (`self.target_pC`).
        
        Backward compatibility
        ----------------------
        - If `self.dist_metric` is not defined (older objects), the method defaults 
          to using `self.fineVal` (Euclidean case).
        
        Parameters
        ----------
        predPC : np.ndarray
            A 1D array of predicted probabilities corresponding to the finely sampled 
            stimulus values (`self.fineVal`).
        
        Returns
        -------
        float
            The stimulus value (Euclidean / Mahalanobis  / Geodesic distance) that 
            best matches the target proportion correct.
        
        """
        idx = np.argmin(np.abs(predPC - self.target_pC))    
        return self.fineVal[idx]
    
    def nLL_Weibull(self, params, dist_xref_x1, resp):
        """
        Compute the negative log-likelihood (nLL) for a Weibull psychometric function.
        
        This method calculates the negative log-likelihood of a Weibull psychometric 
        function given the observed stimulus-response data. It uses the predicted probabilities
        of correct responses (`pC_hyp`) and compares them to the observed responses (`resp`) 
        to evaluate the model's fit.
        
        Parameters
        ----------
        params : np.ndarray
            A 1D array containing the parameters of the Weibull function:
            - params[0]: Threshold (a), controls the point where the function transitions.
            - params[1]: Steepness (b), controls the slope of the curve.
        stim : np.ndarray
            A 2D or 3D array containing the stimulus coordinates for each trial.
        resp : np.ndarray
            A 1D array of binary responses (0 or 1) corresponding to each stimulus.
        
        Returns
        -------
        nLL : float
            The negative log-likelihood value. Lower values indicate a better fit 
            between the model predictions and the observed data.
        """
        
        # Compute predicted probabilities of correct responses (pC_hyp)
        pC_hyp = self.pC_Weibull(params, dist_xref_x1, guess_rate= self.guess_rate)
        
        # Compute the negative log-likelihood
        nLL = -np.sum(resp * np.log(pC_hyp) + (1 - resp) * np.log(1 - pC_hyp))
        
        return nLL
            
    def fit_PsychometricFunc_toData(self):
        """
        Fit a psychometric function (PMF) to the original dataset.
    
        This method serves as a shortcut for fitting a PMF to the original stimulus-response 
        data. It uses `self._fit_PsychometricFunc`, which is also utilized for bootstrapped 
        datasets, to ensure consistency in the fitting process.
        """
        #compute the Euclidean / Mahalanobis / Geodesic distance
        #the default is Euclidean
        self.compute_dist_given_metric() 
        
        # Each unique stimulus is repeated `nTrials_perLevel` times, producing
        # a trial-level distance vector aligned with the response data.
        self.dist_xref_x1 = self.remap_distances(self.nTrials_perLevel)
        
        # Responses (`self.resp_org`) are already sorted by stimulus order, with 
        # trials from the same stimulus grouped in miniblocks.
        self.bestfit_result = self._fit_PsychometricFunc(self.dist_xref_x1, self.resp_org) 
    
    def find_stim_at_targetPC_givenData(self):
        self.stim_at_targetPC = self._find_stim_at_targetPC(self.fine_pC)
        
    def reconstruct_PsychometricFunc_givenData(self):
        self.fine_pC = self._reconstruct_PsychometricFunc(self.bestfit_result.x)
        
    @staticmethod
    def pC_Weibull(weibull_params, dist, guess_rate=1/3, eps=1e-10):
        """
        Compute the probability of a correct response (pC) for multiple trials 
        using the Weibull psychometric function.
    
        This function models the probability of a correct response as a function of 
        the Euclidean distance (L2 norm) of the stimulus coordinates from a reference, 
        based on the Weibull psychometric function.
    
        Parameters
        ----------
        weibull_params : np.ndarray (2,)
            Parameters of the Weibull function:
            - a : float, controls the threshold (the distance at which the response probability 
                  reaches a certain level, e.g., 82% for a 2AFC task).
            - b : float, controls the steepness (how quickly the probability changes 
                  around the threshold).
    
        dist : np.ndarray
            A 1D array of shape (N,), where N is the number of trials. This input represents
            the distance between the reference and comparison stimuli computed using some
            metric ('Euclidean', 'Mahalanobis', or 'Geodesic')
    
        eps : float, optional
            A small value to clip the output probabilities and prevent extreme values 
            (e.g., exactly 0 or 1) that could cause numerical issues. Defaults to 1e-4.
    
        Returns
        -------
        pC_weibull : np.ndarray
            A 1D array of shape (N,) containing the probability of a correct response for 
            each trial, based on the Weibull psychometric function.
    
        """
        
        # Unpack Weibull parameters: 'a' controls threshold, 'b' controls steepness
        a, b = weibull_params
        
        # Compute the probability of a correct response (pC) using the Weibull function
        pC_weibull = 1 - (1 - guess_rate) * np.exp(- (dist / a)**b)
        
        # Clip probabilities to avoid numerical instability
        return np.clip(pC_weibull, eps, 1 - eps)

    def _derive_fineVal(self):
        """
        Build a finely spaced grid over stimulus magnitudes (Euclidean norms).
        Ensures 0 is included in the grid.
        
        """
        vmin = float(np.nanmin(self.dist_xref_x1))
        vmax = float(np.nanmax(self.dist_xref_x1))
        
        # Build finely spaced grid across observed range
        self.fineVal = np.linspace(vmin, vmax, self.nGridPts)
    
    def _reconstruct_PsychometricFunc(self, bestfit_params):
        """
        Reconstruct the psychometric function at finely sampled grid points.
        
        This method uses the best-fitting parameters from the optimization (`self.bestfit_result.x`) 
        to reconstruct the psychometric function. It computes the predicted probability of a correct 
        response (`pC`) for a set of finely sampled stimulus values, allowing visualization of the 
        psychometric function.
        
        """
        
        # Compute the distance given a metric (could be Mahalanobis or Euclidean or Geodesic distance)
        self._derive_fineVal()

        # Compute predicted pC values at the finely sampled stimulus magnitudes
        fine_pC = self.pC_Weibull(bestfit_params, self.fineVal, guess_rate = self.guess_rate)
        return fine_pC
    
    @staticmethod
    def compute_Euclidean_dist(vals):
        """
        Compute the Euclidean distance of each point from the origin.
    
        Parameters
        ----------
        vals : np.ndarray of shape (N, 2)
            Array of N stimulus coordinates. Each row corresponds to a point in 2D space.
    
        Returns
        -------
        np.ndarray of shape (N,)
            Euclidean distances of each point in `vals` from the origin (0, 0).
        """
        return np.linalg.norm(vals, axis=1)
    
    def compute_dist_given_metric(self):
        # Reduce stimulus to 1D distance values
        # Depending on the chosen metric, compute either:
        #   - Euclidean distances from the origin
        #   - Mahalanobis distances using reference and comparison stimuli
        if self.dist_metric == 'Euclidean':
            # Magnitudes of unique stimuli (reduces to a 1D radial axis)
            self.unique_stim_dist = self.compute_Euclidean_dist(self.unique_stim)
        else:
            # --- Construct reference and comparison stimuli in original coordinates ---
            # Repeat the reference stimulus so it matches the number of unique stimuli
            xref_org = np.repeat(self.stim_ref[0][None], self.nLevels, axis=0)
            # Comparison stimuli are the reference plus the displacements
            x1_org = xref_org + self.unique_stim
            
            if self.dist_metric == 'Mahalanobis':        
                # --- Compute Mahalanobis distances for each pair ---
                self.unique_stim_dist = self.compute_Mahalanobis_dist(xref_org, x1_org)        
            elif self.dist_metric == 'Geodesic':
                # --- Compute Geodesic distances for each pair ---
                self.unique_stim_dist = self.compute_Geodesic_dist(xref_org, x1_org)
    
    def remap_distances(self, nTrials_rep):
        """
        Repeat per-unique distances to match per-trial data layout.
    
        Parameters
        ----------
        nTrials_rep : int or array-like of int
            If int: same repeat count for all unique stimuli.
            If array: per-unique repeat counts; length must equal len(self.unique_stim_dist).
    
        Returns
        -------
        dist_xref_x1 : np.ndarray, shape (sum(nTrials_rep),)
            Per-trial distance vector.
        """
        if not hasattr(self, "unique_stim_dist") or self.unique_stim_dist is None:
            raise RuntimeError("Call compute_dist_given_metric() before remapping.")
    
        repeats = np.asarray(nTrials_rep)
        if repeats.shape[0] != self.nLevels:
            raise ValueError(
                f"Length mismatch: repeats len={repeats.shape[0]} vs "
                f"unique distances len={self.nLevels}"
            )
    
        dist_xref_x1 = np.repeat(self.unique_stim_dist, repeats, axis=0)
        return dist_xref_x1
    
    #%% Mahalanobis-distance specific methods
    def load_Wishart_modelpred(self, modelpred_Wishart, W_est, xref):
        """
        Store the Wishart model predictions and the estimated weight matrix for 
        later use in Mahalanobis / Geodesic distance computations.
    
        Parameters
        ----------
        modelpred_Wishart : object
            A Wishart model prediction object that provides methods for computing 
            covariance matrices (e.g., `compute_U`, `compute_Sigmas`, 
            `compute_Mahalanobis_distance_batch`).
        W_est : np.ndarray
            Estimated weight matrix of the basis functions used in the Wishart model. 
            Shape (d, d, nDims, nBasis) where:
                - d : polynomial degree of the basis functions (e.g., 5)
                - nDims : number of stimulus dimensions (e.g., 2)
                - nBasis : number of basis functions (e.g., 3)
            This matrix represents the best-fit weights for reconstructing the 
            psychometric field.
            
        xref : np.ndarray
            Reference stimulus location(s).  
            Note: stimuli in `fit_PMF_MOCS_trials` are centered at the origin 
            (reference-less). To compute Mahalanobis or Geodesic distances, the 
            absolute reference location must be reintroduced, which is supplied here.
        """
        
        self.modelpred_Wishart = modelpred_Wishart
        self.model = modelpred_Wishart.model
        self.W_est = W_est
        self.stim_ref = xref
        if self.stim_ref.shape != self.stim.shape:
            raise ValueError(
                f"`stim_ref.shape` {self.stim_ref.shape} must equal `stim.shape` {self.stim.shape}."
            )
            
    def compute_Mahalanobis_dist(self, xref, x1):
        """
        Compute Mahalanobis distances between reference and comparison stimuli
        under the Wishart model.
    
        Unlike other methods in this class, the inputs `xref` and `x1` should be
        specified in their original (uncentered) coordinates. This is necessary
        because Mahalanobis distances depend on the local covariance structure
        at the absolute stimulus positions, not on relocated (centered) values.
    
        Parameters
        ----------
        xref : np.ndarray of shape (N, 2)
            Array of reference stimulus positions.
        x1 : np.ndarray of shape (N, 2)
            Array of comparison stimulus positions.
    
        Returns
        -------
        np.ndarray
            Array of Mahalanobis distances of length N, one for each pair of
            reference and comparison stimuli.
        """
        # --- Create independent copies of the Wishart model and prediction object ---
        model = copy.deepcopy(self.modelpred_Wishart.model)
        modelp = copy.deepcopy(self.modelpred_Wishart)
    
        # --- Covariance at reference stimuli ---
        # Add a leading batch axis (shape becomes (1, N, 2)) so it matches model.compute_U's input format
        xref_batched = xref[None, ...]
        # Compute covariance matrices for each reference stimulus (shape: (1, N, 2, 2))
        sigma_xref = model.compute_Sigmas(model.compute_U(self.W_est, xref_batched))
    
        # --- Covariance at comparison stimuli ---
        x1_batched = x1[None, ...]   # shape: (1, N, 2)
        sigma_x1 = model.compute_Sigmas(model.compute_U(self.W_est, x1_batched))  # shape: (1, N, 2, 2)
    
        # --- Compute Mahalanobis distances in batch ---
        # Remove the extra leading axis (index 0) to match what compute_Mahalanobis_distance_batch expects
        modelp.compute_Mahalanobis_distance_batch(xref, x1,
                                                  sigma_xref.squeeze(0), 
                                                  sigma_x1.squeeze(0))
        return modelp.mahalanobis_distances
        
    #%% Geodesic-distance specific methods
    def set_geodesic_params(self, num_gen=30, pop_size=50, key=jax.random.PRNGKey(1), tol=1e-4):
        """
        Set parameters for geodesic distance calculations.
    
        Parameters
        ----------
        num_gen : int, optional
            Number of generations for the optimizer (default: 30).
        pop_size : int, optional
            Population size for the optimizer (default: 50).
        key : jax.random.PRNGKey, optional
            PRNG key for reproducibility (default: jax.random.PRNGKey(1)).
        tol : float, optional
            Tolerance for optimizer convergence (default: 1e-4).
        """
        self.geo_num_gen  = num_gen
        self.geo_pop_size = pop_size
        self.geo_key      = key
        self.geo_tol      = tol        
    
    def precision_field(self, x):
        """
        Compute the precision matrices (inverse covariance matrices) at specified 
        stimulus positions under the Wishart model. The precision field describes 
        the local shape of the psychometric function in terms of inverse variance. 
        It is used in geodesic distance computations.
        
        """
        # Compute the covariance matrices Σ(x) for the given stimuli x
        return jnp.linalg.inv(self.model.compute_Sigmas(self.model.compute_U(self.W_est, x)))
        
    def compute_Geodesic_dist(self, xref, x1):
        """
        Compute geodesic distances between reference and comparison stimuli under 
        the Wishart model, and store geodesic paths in a dictionary.
    
        Parameters
        ----------
        xref : np.ndarray of shape (N, 2)
            Reference stimuli.
        x1 : np.ndarray of shape (N, 2)
            Comparison stimuli.
    
        Returns
        -------
        np.ndarray of shape (N,)
            Geodesic distances for each reference–comparison pair.
        """
        nLevels = xref.shape[0]
        geodesic_dist = np.full((nLevels,), np.nan)
        # If dicts don’t exist yet, initialize them
        if not hasattr(self, "_geodesic_path_dict"):
            self._geodesic_path_dict = {}
        if not hasattr(self, "_geodesic_dist_dict"):
            self._geodesic_dist_dict = {}
        if not hasattr(self, "geo_num_gen"):
            self.set_geodesic_params()
            
        for n in range(nLevels):
            # Use tuple of points as dictionary key
            key_pair = (tuple(xref[n]), tuple(x1[n]))
    
            if key_pair in self._geodesic_path_dict or key_pair in self._geodesic_dist_dict:
                geodesic_dist[n] = self._geodesic_dist_dict[key_pair]
            else:          
                state, metrics = geodesics.estimate_v0(
                    xref[n], x1[n], self.precision_field, self.geo_key,
                    num_generations= self.geo_num_gen, popsize= self.geo_pop_size, 
                    tolerance= self.geo_tol, solution_dim = self.model.num_dims,
                )
                v0 = metrics["best_solution"]
                xt_n, ts_n = geodesics.shooting_geodesic(xref[n], v0, self.precision_field)
                cost_n = geodesics.estimate_path_cost(xt_n, ts_n, self.precision_field)
        
                # Save distance
                geodesic_dist[n] = cost_n
    
                #store it in a dictionary
                self._geodesic_path_dict[key_pair] = xt_n
                self._geodesic_dist_dict[key_pair] = cost_n
    
        return geodesic_dist
    
    def get_geodesic_path(self, xref, x1):
        key_pair = (tuple(xref), tuple(x1))
        return self._geodesic_path_dict.get(key_pair, None)

    def get_geodesic_dist(self, xref, x1):
        key_pair = (tuple(xref), tuple(x1))
        return self._geodesic_dist_dict.get(key_pair, None)
                
    #%% bootstrap-related methods
    @staticmethod
    def shuffle_indices_once(cumsum_starts, cumsum_ends, nTrials_perStim, rng):
        """
        Block-wise bootstrap (with replacement) of trial indices.
    
        For each stimulus level k, sample `nTrials_perStim[k]` indices uniformly
        from the half-open interval [cumsum_starts[k], cumsum_ends[k]) and
        concatenate the results across levels. This preserves the number of trials
        per stimulus while resampling within each stimulus block.
    
        Parameters
        ----------
        cumsum_starts : np.ndarray, shape (N,)
            Start index (0-based, inclusive) of each stimulus’s trial block in the
            concatenated trial array.
        cumsum_ends : np.ndarray, shape (N,)
            End index (0-based, exclusive) of each stimulus’s trial block.
        nTrials_perStim : np.ndarray, shape (N,)
            Number of trials for each stimulus block. The sum equals the total
            number of trials.
    
        Returns
        -------
        np.ndarray, shape (sum(nTrials_perStim),)
            Concatenated bootstrap sample of trial indices. Within each block the
            sampling is with replacement. The order of blocks is preserved.
    
        Notes
        -----
        - `np.random.randint(low, high, size)` draws from [low, high).
        - For reproducibility, set a seed before calling (e.g., `np.random.seed(...)`).
        """
        # Sample indices within each block, with replacement
        indices_parts = [rng.integers(start, end, size=count) 
                 for start, end, count in zip(cumsum_starts, cumsum_ends, nTrials_perStim)]
        return np.concatenate(indices_parts)
        
    def bootstrap_and_refit(self, nBtst = 120, seed = None):
        """
        Bootstrap responses and refit the psychometric function.
    
        This method creates bootstrapped datasets by resampling trial indices with
        replacement and, for each bootstrap replicate, fits a Weibull PMF and
        reconstructs the predicted curve on a fine grid. It uses the same fitting
        routine as the original (non-bootstrapped) data to keep results consistent.
        
        perform block-wise (within-stimulus) bootstrap: sample within
        each stimulus block so the per-stimulus trial counts are preserved.
    
        Parameters
        ----------
        nBtst : int, default=120
            Number of bootstrap iterations.
        seed : int or None, default=None
            Seed for reproducible resampling. When provided, a local RNG is used so
            results are repeatable without altering global NumPy state.

        Saving the following attributes:
        - self.resp_btst : np.ndarray
            Bootstrapped response datasets.
        - self.bestfit_result_btst : list
            Fitted parameters and minimal nLL for each bootstrap iteration.
        - self.fine_pC_btst : np.ndarray
            Reconstructed psychometric function for each bootstrap iteration.
        - self.stim_at_targetPC_btst : np.ndarray
            Stimulus value corresponding to the target performance level for each iteration.
        """
        #save them to the class
        self.nBtst = nBtst
        self.seed_btst = seed
        
        rng = np.random.default_rng(seed)  # reproducible & local
        #number of total trials 
        nT = self.resp.shape[0]
        
        # Get start and end indices for each interval
        cumTrials_end = np.cumsum(self.nTrials_perLevel)
        
        cumTrials_start = np.concatenate(([0], cumTrials_end[:-1]))
        # Repeat 120 times and stack into (120, nT)
        shuffled_idx = np.stack([self.shuffle_indices_once(cumTrials_start,
                                                           cumTrials_end, 
                                                           self.nTrials_perLevel,
                                                           rng) \
                                 for _ in range(self.nBtst)], axis=0)
    
        # Initialize arrays to store bootstrap results
        self.dist_xref_x1_btst = np.full((self.nBtst, nT), np.nan)         # Bootstrapped distance
        self.resp_btst = np.full((self.nBtst,) + self.resp.shape, np.nan)  # Bootstrapped responses
        self.bestfit_result_btst = []                                      # Fitted parameters
        self.fine_pC_btst = np.full((self.nBtst, self.nGridPts), np.nan)   # Reconstructed psychometric function
        self.stim_at_targetPC_btst = np.full((self.nBtst,), np.nan)        # Stimuli at target performance level
    
        # Perform bootstrap iterations
        for n in trange(self.nBtst):            
            self.resp_btst[n] = self.resp_org[shuffled_idx[n]]
            self.dist_xref_x1_btst[n] = self.dist_xref_x1[shuffled_idx[n]]
                
            # Fit a psychometric function to the bootstrapped dataset
            fit_btst_n = self._fit_PsychometricFunc(self.dist_xref_x1_btst[n], self.resp_btst[n])
    
            # Extract fitted parameters and negative log-likelihood
            self.bestfit_result_btst.append(fit_btst_n)
    
            # Reconstruct the psychometric function on a finer stimulus grid
            self.fine_pC_btst[n] = self._reconstruct_PsychometricFunc(fit_btst_n.x)
    
            # Identify the stimulus corresponding to the target performance level (e.g., 0.667)
            self.stim_at_targetPC_btst[n] = self._find_stim_at_targetPC(self.fine_pC_btst[n])
            
    def compute_95btstCI(self):
        """
        Compute the 95% bootstrap confidence intervals for the psychometric function predictions.
    
        This method calculates confidence intervals based on the bootstrapped estimates of the 
        stimulus corresponding to the target probability of correct responses (e.g., 66.7%). 
        It also computes the confidence intervals for the model-predicted probability of correct 
        responses at finer grid points.
    
        Attributes Updated
        ------------------
        - self.stim_at_targetPC_95btstCI : np.ndarray
            A 1D array containing the lower and upper bounds of the 95% confidence interval 
            for the stimulus corresponding to the target performance level.
        - self.stim_at_targetPC_95btstErr : np.ndarray
            A 1D array containing the lower and upper error bounds relative to the central 
            stimulus estimate for the target performance level.
        - self.fine_pC_95btstCI : np.ndarray
            A 2D array of shape (2, nGridPts), where the first row contains the lower bound 
            and the second row contains the upper bound of the 95% confidence intervals 
            for the probability correct at each grid point.
        """
        # Step 1: Sort the bootstrapped stimulus values in ascending order
        val_sorted = np.sort(self.stim_at_targetPC_btst)
        
        # Step 2: Compute the indices corresponding to the 2.5% and 97.5% percentiles
        idx_lb = int(np.ceil(self.nBtst * 0.025))  # Lower bound index
        idx_ub = int(np.floor(self.nBtst * 0.975)) - 1  # Upper bound index (convert to 0-based indexing)
        
        # Step 3: Extract the 95% confidence interval for the stimulus at the target performance level
        self.stim_at_targetPC_95btstCI = val_sorted[[idx_lb, idx_ub]]
        
        # Step 4: Compute the error bounds relative to the central stimulus estimate
        self.stim_at_targetPC_95btstErr = np.array([
            self.stim_at_targetPC - self.stim_at_targetPC_95btstCI[0],  # Lower error
            self.stim_at_targetPC_95btstCI[1] - self.stim_at_targetPC   # Upper error
        ])
        
        # Step 5: Sort the bootstrapped probability correct predictions (finer grid)
        # The array has shape (nBtst, nGridPts); sort along the bootstrap axis (axis=0)
        arr_sorted = np.sort(self.fine_pC_btst, axis=0)
        
        # Step 6: Extract the 95% confidence interval for the probability correct at each grid point
        self.fine_pC_95btstCI = arr_sorted[[idx_lb, idx_ub]]

#%%
def compute_Wishart_based_pCorrect_atMOCS(nLevels, fit_PMF_MOCS, 
                                          xref_unique, model_pred,
                                          ndims = 2):
    """
    Compute model-predicted percent-correct curves along each MOCS direction
    and extract the corresponding threshold distance and stimulus coordinates.

    This helper is dimension-agnostic and can be used for 2D or 3D MOCS
    stimulus sets, provided that each MOCS condition consists of one reference
    location plus comparison stimuli sampled along a single direction at
    multiple levels.

    Threshold distance is interpreted in Euclidean stimulus space. This helper
    does not currently implement alternative distance metrics such as a
    Mahalanobis-based threshold readout.

    Returns:
        dict:
            - ``pChoosingX1_Wishart``: model-predicted psychometric curves
              evaluated on a dense stimulus line for each reference condition.
            - ``vecLen_at_targetPC_Wishart``: threshold distance at the target
              performance level for each reference condition.
            - ``stim_at_targetPC_Wishart``: stimulus coordinates at that
              threshold in ``ndims``.

    """
    
    # Initialize arrays to store results
    nRefs = len(fit_PMF_MOCS)
    pChoosingX1_Wishart          = np.full((nRefs, fit_PMF_MOCS[0].nGridPts), np.nan)
    vecLen_at_targetPC_Wishart   = np.full((nRefs,), np.nan)
    stim_at_targetPC_Wishart     = np.full((nRefs, ndims), np.nan)

    for n in trange(nRefs):
        # Sort the tested comparison vectors by distance from the origin so we
        # can recover the outermost MOCS direction for this condition.
        sorted_indices = np.argsort(-np.linalg.norm(fit_PMF_MOCS[n].unique_stim, axis=1))
        sorted_array = fit_PMF_MOCS[n].unique_stim[sorted_indices]

        # Build a dense stimulus line along that same direction so the model can
        # be evaluated more finely than at the original discrete MOCS levels.
        finer_stim = sim_MOCS_trials.create_discrete_stim(
            sorted_array[0], 
            fit_PMF_MOCS[n].nGridPts,
            ndims= ndims
        )

        # Evaluate model-predicted proportion correct for stimuli centered on
        # the current MOCS reference.
        pChoosingX1_Wishart[n] = model_pred._compute_pChoosingX1(
            np.full(finer_stim.shape, 0) + xref_unique[n], 
            finer_stim + xref_unique[n]
        )

        # Read out the threshold distance corresponding to the target
        # performance level from the predicted psychometric curve.
        vecLen_at_targetPC_Wishart[n] = fit_PMF_MOCS[n]._find_stim_at_targetPC(pChoosingX1_Wishart[n])

        # Convert the scalar threshold distance back into stimulus coordinates
        # by projecting along the canonical MOCS direction for this condition.
        stim_at_targetPC_Wishart[n] = vecLen_at_targetPC_Wishart[n] * (
            fit_PMF_MOCS[n].unique_stim[nLevels // 2] /
            np.linalg.norm(fit_PMF_MOCS[n].unique_stim[nLevels // 2])
        ) + xref_unique[n]

    Wishart_based_thres_atMOCS = {
        "pChoosingX1_Wishart": pChoosingX1_Wishart,
        "vecLen_at_targetPC_Wishart": vecLen_at_targetPC_Wishart,
        "stim_at_targetPC_Wishart": stim_at_targetPC_Wishart,
    }
    return Wishart_based_thres_atMOCS
    
#%%            
class sim_MOCS_trials:
    @staticmethod
    def generate_vectors_min_angle(min_angle_degrees=60, max_angle_degrees=160,
                                   ndims=2, seed=None):
        """        
        Generate two random vectors in a given dimension (2D or 3D) such that their angle is 
        at least `min_angle_degrees` apart and at most `max_angle_degrees` apart.

        Args:
            min_angle_degrees (float): The minimum angle (in degrees) between the two vectors.
            max_angle_degrees (float): The maximum angle (in degrees) between the two vectors.
            ndims (int): Dimension of the vectors (2 for plane, 3 for RGB cube).
            seed (int, optional): Seed for the random number generator for reproducibility.

        Returns:
            tuple: Two numpy arrays representing the two vectors.
            
        """
        if seed is not None:
            np.random.seed(seed)

        # Convert angles to radians
        min_angle_radians = np.radians(min_angle_degrees)
        max_angle_radians = np.radians(max_angle_degrees)

        while True:
            # Generate two random vectors in the specified dimension
            vector1 = np.random.randn(ndims)
            vector2 = np.random.randn(ndims)

            # Normalize the vectors to make them unit vectors
            vector1 /= np.linalg.norm(vector1)
            vector2 /= np.linalg.norm(vector2)

            # Compute the cosine of the angle between the two vectors
            cos_theta = np.dot(vector1, vector2)
            cos_theta = np.clip(cos_theta, -1, 1)  # Ensure numerical stability
            angle = np.arccos(cos_theta)  # Get the angle in radians

            # Check if the vectors satisfy the angle constraints
            if min_angle_radians <= angle <= max_angle_radians:
                return vector1, vector2

    @staticmethod
    def sim_binary_trials(p, N, seed=None):
        """
        Simulate binary responses based on a probability `p` for `N` trials.

        Args:
            p (float): Probability of success (1) for each trial (0 ≤ p ≤ 1).
            N (int): Number of trials to simulate.
            seed (int, optional): Seed for reproducibility.

        Returns:
            numpy.ndarray: A 1D array of binary responses (0 or 1).
        """
        # if seed is not None:
        #     np.random.seed(seed)

        # # Generate binary responses directly using a binomial distribution
        # resp = np.random.binomial(1, p, N)
        
        rng = np.random.default_rng(seed)  # Ensures rng is always defined
        random_values = rng.random(N)  # Generate N random numbers between 0 and 1
        resp = (random_values < p).astype(int)  # Convert to binary responses
        return resp, np.mean(resp)
        
    @staticmethod
    def create_discrete_stim(endpoint, num_pts, startpoint = None, ndims = 2):
        startpoint = np.array([0]*ndims)
        if endpoint.shape[0] != startpoint.shape[0] != ndims:
            raise ValueError('The dimensions of points do not match!')
         
        for n in range(ndims):
            discrete_dim_n = np.linspace(startpoint[n], endpoint[n], num_pts)
            if n == 0:
                discrete_stim = discrete_dim_n[:,None]
            else:
                discrete_stim = np.hstack((discrete_stim, discrete_dim_n[:,None]))
        return discrete_stim
        
    @staticmethod
    def sample_sobol(N, lb, ub, force_center=False, seed=None):
        """
        Generate N Sobol-sequenced points within a bounded space in arbitrary dimensions,
        optionally forcing the first point to be at the center.
    
        Args:
            N (int): Number of points to sample.
            lb (array-like): Lower bounds for each dimension.
            ub (array-like): Upper bounds for each dimension.
            force_center (bool): If True, the first point is set at the center.
            seed (int, optional): Random seed for reproducibility.
    
        Returns:
            np.ndarray: (N, len(lb)) array of Sobol samples.
        """
        lb = np.array(lb)
        ub = np.array(ub)
        ndims = len(lb)  # Determine number of dimensions from bounds
    
        if N < 1:
            raise ValueError("N must be at least 1.")
        if len(lb) != len(ub):
            raise ValueError("Lower and upper bounds must have the same length.")
        
        # Initialize Sobol sequence generator
        sobol_sampler = Sobol(d=ndims, scramble=True, seed=seed)
    
        # Generate N Sobol points in [0,1]^ndims
        samples = sobol_sampler.random(N)
    
        # Scale to [lb, ub] for each dimension
        samples = lb + (ub - lb) * samples  
    
        if force_center:
            samples[0] = (lb + ub) / 2  # Force first point to be at the center
    
        return samples
        
        
        
    
    
    
    
    
    
    
        
        
        
