#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:15:58 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
from scipy.linalg import sqrtm
import numpy as np
from scipy.stats import special_ortho_group
import os
from analysis.ellipses_tools import ellParams_to_covMat, rotAngle_to_eigenvectors, ellParamsQ_to_covMat 

#%%
class ModelPerformance():
    def __init__(self, color_dimension, gt_ellParams, nLevels = 1):
        """
        Initializes the ModelPerformance object with the necessary data for 
        evaluating model performance against ground-truth ellipses or ellipsoids.
        
        Parameters:
        - color_dimension: int, dimensionality of the color space (2 for ellipses, 3 for ellipsoids).
        = gt_ellParams: list, ground truth ellipses
            2D size: len(gt_ellParams) = 7, len(gt_ellParams[0]) = 7, len(gt_ellParams[0],[0]) = 5 parameters 
            
        """
        self.ndims        = color_dimension 
        self.ellParams_gt = gt_ellParams
        self.nRefs        = len(gt_ellParams)
        self.nLevels      = nLevels
        
    def _initialize(self):
        """
        This method initializes arrays that save model performance results
        BW_distance: Bures-Wasserstein distance between the ground-truth and model predictions
        BW_distance_maxEigval: BW distance between the ground-truth and ellipses/ellipsoids with the largest eigenvalue
        BW_distance_minEigval: BW distance between the ground-truth and ellipses/ellipsoids with the smallest eigenvalue
        LU_distance: Log-Euclidean distance between the ground-truth and model predictions
        LU_distance_maxEigval: LU distance between the ground-truth and ellipses/ellipsoids with the largest eigenvalue
        LU_distance_minEigval: LU distance between the ground-truth and ellipses/ellipsoids with the smallest eigenvalue

        """
        self.BW_distance           = np.full((self.nLevels, self.nRefs), np.nan)
        self.BW_distance_maxEigval = np.full((self.nRefs,), np.nan)
        self.BW_distance_minEigval = np.full((self.nRefs,), np.nan)
        
        self.LU_distance           = np.full((self.nLevels, self.nRefs), np.nan)
        self.LU_distance_maxEigval = np.full((self.nRefs,), np.nan)
        self.LU_distance_minEigval = np.full((self.nRefs,), np.nan)
        
        #ground truth covariance matrices and model predictions
        self.covMat_gt             = np.full((self.nRefs, self.ndims, self.ndims), np.nan)
        self.covMat_modelPred      = np.full((self.nLevels, self.nRefs, self.ndims, self.ndims), np.nan)
    
    def load_modelPreds_ellParams(self, ellParams_set, verbose = False):
        """
        Loads model predictions (ellipsoid or ellipse parameters) from the Wishart model
        and converts them to covariance matrices.
        
        Scales the radii by 1/2 because the Wishart model operates in the W space [-1, 1],
        while the ground-truth ellipses/ellipsoids are in the N space [0,1].
        """
        try:
            for l in range(self.nLevels): 
                ellParams_l = ellParams_set[l]
                for ii in range(self.nRefs):
                    if self.ndims == 2:
                        # Retrieve predicted ellipse parameters (rotation → eigenvectors)
                        eigVec_ii = rotAngle_to_eigenvectors(ellParams_l[ii][-1])
                        radii_ii = np.array(ellParams_l[ii][2:4])
                    elif self.ndims == 3:
                        eigVec_ii = ellParams_l[ii]['evecs']
                        radii_ii = ellParams_l[ii]['radii']
                        
                    # Sort radii and eigenvectors (enforce major/minor axis ordering)
                    radii_ii, eigVec_ii = ModelPerformance.sort_eig(radii_ii, eigVec_ii)
            
                    # Convert sorted ellipse params into covariance matrix
                    self.covMat_modelPred[l, ii] = ellParams_to_covMat(radii_ii, eigVec_ii)
            
                    # If verbose mode is enabled, print radius comparison for first level
                    if verbose and l == 0:
                        _, radii_gt = self._convert_ellParams_to_covMat(self.ellParams_gt[ii])
                        print(f"{ii}:")
                        print(f"Ground truths: {np.sort(radii_gt)}")
                        print(f"W Model preds: {np.sort(radii_ii)}")
        except:
            print(f'l: {l}; ii: {ii}')
            print('Cannot find ell parameters.')
                                
    def _convert_ellParams_to_covMat(self, ellParams):
        """
        Converts ellipse or ellipsoid parameters into a covariance matrix.
        
        Scales the radii by a specified factor (default is 5) and sorts the radii and
        corresponding eigenvectors in descending order.
        """
        if self.ndims == 2:
            _, _, a, b, R = ellParams
            radii = np.array([a, b])
            eigvecs = rotAngle_to_eigenvectors(R)
        else:
            radii = ellParams['radii']
            eigvecs = ellParams['evecs']
        
        # Sort radii and eigenvectors in descending order.
        radii, eigvecs = ModelPerformance.sort_eig(radii, eigvecs)
        
        # Convert to covariance matrix.
        covMat = ellParams_to_covMat(radii, eigvecs)
        return covMat, radii
                
    def compare_with_extreme_ell(self, ell1Params):  
        """
        Compares the ground-truth ellipse/ellipsoid to extreme cases (largest and smallest eigenvalue),
        generating covariance matrices for bounding spheres and computing performance metrics.
        
        Returns covariance matrix, Bures-Wasserstein and Log-Euclidean distances for both extreme cases.
        """

        # Use the eigenvalues and eigenvectors to derive the cov matrix
        covMat_gt, radii_gt = self._convert_ellParams_to_covMat(ell1Params)
            
        #--------- Benchmark for evaluating model performance ------------
        # Evaluate using maximum eigenvalue (creates a bounding sphere)
        radii_gt_max = np.ones((self.ndims))*np.max(radii_gt)
        covMat_max = ellParams_to_covMat(radii_gt_max, np.eye(self.ndims))
        
        # Evaluate using minimum eigenvalue (creates an inscribed sphere)
        radii_gt_min = np.ones((self.ndims))*np.min(radii_gt)
        covMat_min = ellParams_to_covMat(radii_gt_min, np.eye(self.ndims))
        
        #compute Bures-Wasserstein distance between the ground truth cov matrix
        #and the smallest sphere that can just contain the ellipsoid
        BW_distance_maxEigval = self.compute_Bures_Wasserstein_distance(\
            covMat_gt,covMat_max)
        #compute Bures-Wasserstein distance between the ground truth cov matrix
        #and the largest sphere that can just be put inside the ellipsoid
        BW_distance_minEigval = self.compute_Bures_Wasserstein_distance(\
            covMat_gt,covMat_min)
            
        LU_distance_maxEigval = self.log_operator_norm_distance(covMat_gt, covMat_max)       
        LU_distance_minEigval = self.log_operator_norm_distance(covMat_gt, covMat_min)     
        
        return covMat_gt, BW_distance_minEigval, BW_distance_maxEigval,\
            LU_distance_minEigval, LU_distance_maxEigval
    
    def compare_with_corner_ell(self, covMat_gt, covMat_corner):
        """
        Compares the ground-truth covariance matrix with corner ellipsoids and computes the
        Bures-Wasserstein and Log-Euclidean distances for each corner.
        
        Returns arrays of distances for each corner.
        """
        #initialize
        BW_distance_corner = np.full((self.nCorners,), np.nan)
        LU_distance_corner = np.full((self.nCorners,), np.nan)
        #Compute normalized bures similarity and Bures-Wasserstein distance
        #between ground truth and ellipsoids at selected corner locations
        for m in range(self.nCorners):
            BW_distance_corner[m] = self.compute_Bures_Wasserstein_distance(covMat_corner[m],covMat_gt)
            LU_distance_corner[m] = self.log_operator_norm_distance(covMat_corner[m],covMat_gt)
        return BW_distance_corner, LU_distance_corner
        
    def compare_gt_model_pred_one_instance(self, covMat_gt, covMat_modelPred):
        """
        Compare a ground-truth covariance matrix to model-predicted covariance matrices 
        across multiple levels (e.g., bootstrapped estimates or prediction samples).
    
        This method computes two types of distances for each level:
            - Bures-Wasserstein (BW) distance
            - Log-Euclidean (LU) distance
    
        Args:
            covMat_gt (np.ndarray): 
                Ground-truth covariance matrix of shape (2, 2) or (3, 3).
            covMat_modelPred (np.ndarray): 
                Model-predicted covariance matrices of shape (N, 2, 2) or (N, 3, 3),
                where N is the number of prediction levels (e.g., bootstrap samples).
    
        Returns:
            BW_distance (np.ndarray): 
                Array of shape (N,) containing Bures-Wasserstein distances between 
                the ground truth and each predicted covariance matrix.
            LU_distance (np.ndarray): 
                Array of shape (N,) containing Log-Euclidean distances between 
                the ground truth and each predicted covariance matrix.
        """
        BW_distance = np.full((self.nLevels,), np.nan)
        LU_distance = np.full((self.nLevels,), np.nan)
        for l in range(self.nLevels):    
            BW_distance[l] = self.compute_Bures_Wasserstein_distance(\
                covMat_gt,covMat_modelPred[l])
            LU_distance[l] = self.log_operator_norm_distance(covMat_gt,
                                                             covMat_modelPred[l])
        return BW_distance, LU_distance

    def evaluate_model_performance(self, ellParams_set, covMat_corner = None):
        """
        Evaluate model performance by comparing predicted ellipsoids to ground-truth ellipsoids
        across all reference locations and bootstrap levels.
    
        This method computes two distance metrics between the predicted and ground-truth
        covariance matrices:
            - Bures-Wasserstein (BW) distance
            - Log-Euclidean (LU) distance
    
        Optionally, the method also compares the ground-truth ellipsoids to "corner" ellipsoids,
        which serve as a baseline reference to illustrate the range of expected distances 
        (i.e., how bad a poor prediction might be).
    
        Args:
            ellParams_set (List[List[ndarray]]): 
                Model predictions. The outer list has length N (number of bootstrap levels),
                and each inner list has length M (number of reference stimuli).
                Each element contains the ellipse parameters for a predicted ellipsoid.
    
            covMat_corner (List[ndarray], optional): 
                Ground-truth covariance matrices at a few "corner" locations.
                Used as additional baseline references. Each matrix is 2x2 or 3x3.
        
        Raises:
            ValueError: If the number of levels or number of reference stimuli does not match 
            expected values.
        """
        if len(ellParams_set) != self.nLevels:
            raise ValueError('Mismatch in the number of bootstrap datasets!')
        
        if len(ellParams_set[0]) != self.nRefs:
            raise ValueError('Mismatch in the number of reference stimuli!')
    
        # Initialize internal arrays to store computed distance metrics
        self._initialize()
    
        # Convert predicted ellipse parameters to covariance matrices
        self.load_modelPreds_ellParams(ellParams_set)
    
        # Initialize arrays for corner comparisons, if provided
        if covMat_corner is not None:
            self.nCorners = len(covMat_corner)
            self.BW_distance_corner = np.full((self.nCorners,) + self.BW_distance_maxEigval.shape, np.nan)
            self.LU_distance_corner = np.full(self.BW_distance_corner.shape, np.nan)
    
        for idx in range(self.nRefs):
            try:
                # Compare each ground-truth ellipse to its inscribed and circumscribed circles
                # These serve as baselines to interpret model performance (best and worst cases)
                self.covMat_gt[idx], self.BW_distance_minEigval[idx], \
                    self.BW_distance_maxEigval[idx], self.LU_distance_minEigval[idx], \
                    self.LU_distance_maxEigval[idx] = self.compare_with_extreme_ell(
                        self.ellParams_gt[idx])
    
                # Optionally compare ground-truth ellipses to corner ellipses
                # This gives additional sense of variability / performance range
                if covMat_corner is not None:
                    self.BW_distance_corner[:, idx], self.LU_distance_corner[:, idx] = \
                        self.compare_with_corner_ell(self.covMat_gt[idx], covMat_corner)
    
                # Compare model-predicted ellipses to ground-truth ellipse for each bootstrap level
                self.BW_distance[:, idx], self.LU_distance[:, idx] = \
                    self.compare_gt_model_pred_one_instance(self.covMat_gt[idx], 
                                                            self.covMat_modelPred[:, idx])
            except:
                print('Cannot find ellipse parameters for reference index', idx)
            
    def concatenate_benchamrks(self):
        #we pick multiple ellipses/ellipsoids for computing benchmarks, including
        #the 
        self.BW_benchmark = np.concatenate((self.BW_distance_minEigval[np.newaxis],\
                                           self.BW_distance_maxEigval[np.newaxis],\
                                           self.BW_distance_corner), axis = 0)
            
        self.LU_benchmark = np.concatenate((self.LU_distance_minEigval[np.newaxis],\
                                       self.LU_distance_maxEigval[np.newaxis],\
                                       self.LU_distance_corner), axis = 0)

#%%
    @staticmethod
    def sort_eig(radii, eigvecs, order='descending'):
        # Sort radii and eigenvectors
        sorted_indices = np.argsort(radii)
        if order == 'descending':
            sorted_indices = sorted_indices[::-1]  # Reverse for descending order
        radii_sorted = radii[sorted_indices]
        eigvecs_sorted = eigvecs[:, sorted_indices]
        return radii_sorted, eigvecs_sorted
    
    @staticmethod
    def compute_Bures_Wasserstein_distance(M1, M2):
        # Compute the square root of M1
        sqrt_M1 = sqrtm(M1)
        # Compute the product sqrt(M1) * M2 * sqrt(M1)
        product = sqrt_M1 @ M2 @ sqrt_M1
        # Compute the square root of the product
        sqrt_product = sqrtm(product)
        # Ensure the result is real
        if np.iscomplexobj(sqrt_product):
            sqrt_product = np.real(sqrt_product)
            print(M1)
            print(M2)
            
        # Calculate the Bures-Wasserstein distance
        trace_diff = np.trace(M1) + np.trace(M2) - 2 * np.trace(sqrt_product)
        trace_diff = max(0, trace_diff)  # Avoid negative values under sqrt
        
        BW_distance = np.sqrt(trace_diff)        
        return BW_distance
            
    @staticmethod
    def compute_normalized_Bures_similarity(M1, M2):
        M1 = np.asarray(M1)
        M2 = np.asarray(M2)
    
        if M1.shape != M2.shape:
            raise ValueError(f"M1 and M2 must have the same shape, got {M1.shape} and {M2.shape}")
        if M1.ndim < 2 or M1.shape[-1] != M1.shape[-2]:
            raise ValueError("M1 and M2 must have shape (..., d, d)")
            
        # Compute the product inside the trace
        inner_product = sqrtm(sqrtm(M1) @ M2 @ sqrtm(M1))  
        # Calculate the trace of the product
        trace_value = np.trace(inner_product)    
        # Normalize by the geometric mean of the traces of M1 and M2
        normalization_factor = np.sqrt(np.trace(M1) * np.trace(M2))    
        # Calculate NBS
        NBS = trace_value / normalization_factor    
        return NBS
    
    @staticmethod
    def compute_normalized_Bures_similarity_batch(M1, M2, eps=1e-12):
        """
        Compute normalized Bures similarity (NBS) for batches of covariance matrices.
        This method is suitable for fast computing the NBS between two sets of cov matrices
    
        Parameters
        ----------
        M1, M2 : ndarray
            Arrays of shape (..., d, d), where each pair M1[..., :, :] and
            M2[..., :, :] is one covariance-matrix pair.
        eps : float
            Small value for numerical stability when clipping eigenvalues.
    
        Returns
        -------
        NBS : ndarray
            Array of shape (...) containing the NBS for each matrix pair.
        """
        M1 = np.asarray(M1)
        M2 = np.asarray(M2)
    
        if M1.shape != M2.shape:
            raise ValueError(f"M1 and M2 must have the same shape, got {M1.shape} and {M2.shape}")
        if M1.ndim < 2 or M1.shape[-1] != M1.shape[-2]:
            raise ValueError("M1 and M2 must have shape (..., d, d)")
    
        # sqrt(M1) via batched eigendecomposition
        evals1, evecs1 = np.linalg.eigh(M1)
        evals1 = np.clip(evals1, eps, None)
        sqrt_evals1 = np.sqrt(evals1)
    
        # sqrt(M1) = V diag(sqrt(lambda)) V^T
        sqrt_M1 = (evecs1 * sqrt_evals1[..., None, :]) @ np.swapaxes(evecs1, -1, -2)
    
        # A = sqrt(M1) @ M2 @ sqrt(M1)
        A = sqrt_M1 @ M2 @ sqrt_M1
    
        # trace(sqrt(A)) = sum(sqrt(eigvals(A)))
        evalsA = np.linalg.eigvalsh(A)
        evalsA = np.clip(evalsA, eps, None)
        trace_sqrt_A = np.sum(np.sqrt(evalsA), axis=-1)
    
        # normalization factor = sqrt(trace(M1) * trace(M2))
        trace_M1 = np.trace(M1, axis1=-2, axis2=-1)
        trace_M2 = np.trace(M2, axis1=-2, axis2=-1)
        normalization_factor = np.sqrt(np.clip(trace_M1 * trace_M2, eps, None))
    
        return trace_sqrt_A / normalization_factor
    
    @staticmethod
    def log_psd_matrix(S, tol=1e-4):
    	v, U = np.linalg.eigh(S)
    	d = np.log(np.clip(v, tol, None))
    	return U @ np.diag(d) @ U.T
    
    @staticmethod
    def log_operator_norm_distance(A, B):
    	lgA = ModelPerformance.log_psd_matrix(A)
    	lgB = ModelPerformance.log_psd_matrix(B)
    	return np.linalg.norm(lgA - lgB, 2)

#%%
def compute_95CI_BWD_multipleConditions(BWD):
    """
    Compute the 95% confidence interval of Bures-Wasserstein Distances (BWD) for each condition
    by flattening across bootstrap levels and reference stimuli.

    Parameters
    ----------
    BWD : np.ndarray of shape (nConditions, nLevels, nRefs)
        Bures-Wasserstein distances between model-predicted ellipses and ground-truth ellipses
        for multiple experimental conditions.

    Returns
    -------
    CI95 : np.ndarray of shape (2, nConditions)
        The lower (0.025) and upper (0.975) bounds of the 95% confidence interval for each condition.

    yerr : np.ndarray of shape (2, nConditions)
        Asymmetric error bars computed as:
            - yerr[0] = median - lower bound
            - yerr[1] = upper bound - median

    BWD_median : np.ndarray of shape (nConditions,)
        Median BWD for each condition.
    """
    # Compute median across all levels and reference locations for each condition
    BWD_median = np.nanmedian(BWD, axis=(1, 2))

    nConditions = BWD.shape[0]

    # Flatten across levels and reference locations: shape becomes (nConditions, nLevels * nRefs)
    BWD_flat_sorted = np.sort(np.reshape(BWD, (nConditions, -1)), axis=1)

    # Count non-NaN values per condition
    valid_counts = np.sum(~np.isnan(BWD_flat_sorted), axis=1)

    # Compute indices for 2.5th and 97.5th percentiles
    CI_idx_bounds = np.round(
        np.outer(valid_counts, [0.025, 0.975])
    ).astype(int)

    # Extract lower and upper CI bounds for each condition
    CI95_lower = BWD_flat_sorted[np.arange(nConditions), CI_idx_bounds[:, 0]]
    CI95_upper = BWD_flat_sorted[np.arange(nConditions), CI_idx_bounds[:, 1]]
    CI95 = np.vstack((CI95_lower, CI95_upper))

    # Compute asymmetric error bars for plotting
    yerr = np.vstack((BWD_median - CI95[0], CI95[1] - BWD_median))

    return CI95, yerr, BWD_median

def generate_ellipses_within_BWdistance(ellipse_gt, target_bw_dist, min_axis_len, max_axis_len,
                     max_trials=10000, tol=1e-4, seed=None, num_ellipses=1):
    """
    Generate ellipses whose Bures-Wasserstein distance to the ground truth ellipse
    is close to the target distance.

    Parameters:
        ellipse_gt: tuple (center, (a_gt, b_gt), theta_gt)
        target_bw_dist: float
        min_axis_len: float
        max_axis_len: float
        max_trials: int
        tol: float
        seed: int or None
        num_ellipses: int (number of ellipses to generate)

    Returns:
        ellipses: list of tuples [(center, (a, b), theta), ...]
        distances: list of corresponding BW distances
    """
    if seed is not None:
        np.random.seed(seed)
    
    a_gt, b_gt, theta_gt, center_x_gt, center_y_gt = ellipse_gt
    cov_gt = ellParamsQ_to_covMat(a_gt, b_gt, theta_gt)
    
    ellipses = []
    distances = []
    
    attempts = 0
    while len(ellipses) < num_ellipses and attempts < max_trials:
        # Randomly sample axis lengths and angle within bounds
        a = np.random.uniform(min_axis_len, max_axis_len)
        b = np.random.uniform(min_axis_len, max_axis_len)
        theta = np.random.uniform(0, 180)
        
        cov_sim = ellParamsQ_to_covMat(a, b, theta)
        
        bw_dist = ModelPerformance.compute_Bures_Wasserstein_distance(cov_gt, cov_sim)
        
        if np.isclose(bw_dist, target_bw_dist, atol=tol):
            ellipses.append((center_x_gt,center_y_gt, a, b, theta))
            distances.append(bw_dist)
        
        attempts += 1
    
    if len(ellipses) < num_ellipses:
        print(f"Only found {len(ellipses)} ellipse(s) within {max_trials} trials.")
    
    return ellipses, distances

def generate_ellipsoids_within_BWdistance(gt_ellipsoid, target_bw_dist, min_axis_len, max_axis_len,
                                          max_trials=10000, tol=1e-4, seed=None, num_ellipsoids=1):
    """
    Generate ellipsoids in 3D whose BW distance to the ground truth ellipsoid is close to the target.

    Parameters:
        gt_ellipsoid: dict with 'radii', 'evecs', and 'center'
        target_bw_dist: float
        min_axis_len: float
        max_axis_len: float
        max_trials: int
        tol: float
        seed: int or None
        num_ellipsoids: int

    Returns:
        ellipsoids: list of dicts with 'radii', 'evecs', 'center'
        distances: list of BW distances
    """
    if seed is not None:
        np.random.seed(seed)

    # Ground truth covariance matrix: Σ = R * diag(radii^2) * R^T
    radii_gt = gt_ellipsoid['radii']
    evecs_gt = gt_ellipsoid['evecs']
    center_gt = gt_ellipsoid['center'].flatten()

    cov_gt = evecs_gt @ np.diag(radii_gt**2) @ evecs_gt.T

    ellipsoids = []
    distances = []

    attempts = 0
    while len(ellipsoids) < num_ellipsoids and attempts < max_trials:
        # Sample radii and a random rotation matrix
        radii = np.random.uniform(min_axis_len, max_axis_len, size=3)
        evecs = special_ortho_group.rvs(3)  # Random orthonormal 3x3 matrix

        cov_sim = evecs @ np.diag(radii**2) @ evecs.T
        bw_dist = ModelPerformance.compute_Bures_Wasserstein_distance(cov_gt, cov_sim)

        if np.isclose(bw_dist, target_bw_dist, atol=tol):
            ellipsoids.append({
                'radii': radii,
                'evecs': evecs,
                'center': center_gt.copy()
            })
            distances.append(bw_dist)

        attempts += 1

    if len(ellipsoids) < num_ellipsoids:
        print(f"Only found {len(ellipsoids)} ellipsoid(s) within {max_trials} trials.")

    return ellipsoids, distances