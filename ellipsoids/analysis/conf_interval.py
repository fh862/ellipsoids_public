#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 12:26:25 2025

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from shapely.geometry import Polygon
import os
from analysis.ellipses_tools import find_inner_outer_contours, distance_to_ellipse_boundary

#%%
def find_inner_outer_contours_for_gridRefs(p_ell, nTheta=1000):
    """    
    Computes the inner and outer contour points (confidence intervals) for the model-predicted ellipses 
    at each grid point, generalized to an n-dimensional grid.

    Args:
        p_ell (ndarray): Ellipse parameters for all grid points.
            Shape: (ng1, ng2, ..., ngN, nBtst, 5)
            - ng1, ng2, ..., ngN: Grid dimensions (for ndims=N).
            - nBtst: Number of bootstrapped datasets.
            - 5 parameters per ellipse: [x0, y0, a, b, rotAngle].
        nTheta (int): Maximum number of contour points to preallocate for each ellipse.

    Returns:
        tuple:
            ell_min (ndarray): Inner contour (lower confidence bound) for each grid point.
                Shape: (ng1, ng2, ..., ngN, 2, nTheta)
            ell_max (ndarray): Outer contour (upper confidence bound) for each grid point.
                Same shape as ell_min.
    """
    # Get grid shape for first ndims dimensions
    grid_shape = p_ell.shape[:-2]

    # Preallocate output arrays
    ell_min = np.full((*grid_shape, 2, nTheta), np.nan)
    ell_max = np.full(ell_min.shape, np.nan)

    # Iterate over all grid indices
    for grid_idx in np.ndindex(*grid_shape):
        # Extract ellipse parameters at this grid index
        params_ij = p_ell[*grid_idx] # shape: (nBtst, 5)
        
        # Handle missing values (NaNs) in ellipse parameters:
        # - Remove any bootstrap samples where the first parameter (center_x) is NaN
        # - This ensures only valid ellipses are used to compute confidence intervals
        nan_rows = np.any(np.isnan(params_ij), axis=1)  # boolean mask of rows with at least 1 NaN
        params_ij_removeNaN = params_ij[~nan_rows]

        if params_ij.shape[0] != params_ij_removeNaN.shape[0]:
            print('NaN values detected in ellipse parameters at grid index', grid_idx,
                  '— these samples were excluded from CI computation.')
            
        # Compute the inner and outer contours for the current grid point.
        # The function find_inner_outer_contours returns:
        #   xu_ij, yu_ij: x and y coordinates of the outer (upper) contour.
        #   xi_ij, yi_ij: x and y coordinates of the inner (lower) contour.
        xu_ij, yu_ij, xi_ij, yi_ij = find_inner_outer_contours(params_ij_removeNaN)

        # Determine how many points were computed for each contour.
        idx_u = xu_ij.shape[0]  # Number of points in the outer contour.
        idx_i = xi_ij.shape[0]  # Number of points in the inner contour.

        # Store the outer contour coordinates.
        ell_max[*grid_idx, 0, :idx_u] = xu_ij  # X-coordinates of the outer contour.
        ell_max[*grid_idx, 1, :idx_u] = yu_ij  # Y-coordinates of the outer contour.

        # Store the inner contour coordinates.
        ell_min[*grid_idx, 0, :idx_i] = xi_ij  # X-coordinates of the inner contour.
        ell_min[*grid_idx, 1, :idx_i] = yi_ij  # Y-coordinates of the inner contour.

    return ell_min, ell_max

def find_distance_to_ellipse_boundary_for_gridRef(p_ell, chromDir, bds_idx=[0, -1]):
    """
    Computes the bootstrapped confidence intervals on threshold distances 
    along a given chromatic direction for each grid point.
    
    Args: 
        p_ell (ndarray): Ellipse parameters for all grid points.
            Shape: (ng1, ng2, nBtst, 5)
            - ng1, ng2: Grid dimensions
            - nBtst: Number of bootstrapped datasets
            - Last dimension: [x0, y0, a, b, rotAngle]
        chromDir (ndarray): Chromatic direction vectors for each grid point.
            Shape: (ng1, ng2, 2)
            - Last dimension is the (dx, dy) vector
        bds_idx (list or tuple): Indices for lower and upper bounds in the sorted bootstrapped values.
            Default: [0, -1] → full min-max interval
    
    Returns:
        tuple:
            opt_vec (ndarray): Distance to ellipse boundary for each bootstrap.
                Shape: (ng1, ng2, nBtst)
            opt_vec_sorted (ndarray): Sorted distances for each grid point.
            opt_vec_sorted_lb (ndarray): Lower bound across bootstraps for each grid point.
                Shape: (ng1, ng2)
            opt_vec_sorted_ub (ndarray): Upper bound across bootstraps for each grid point.
                Shape: (ng1, ng2)
    """
    ng1, ng2, nBtst = p_ell.shape[0:3]
    opt_vec = np.full((ng1, ng2, nBtst), np.nan)
    
    for i in range(ng1):
        for j in range(ng2):
            # Normalize chromatic direction at this grid point
            dx, dy = chromDir[i, j]
            norm = np.linalg.norm([dx, dy])
            if norm < 1e-10:
                continue  # Skip if direction vector is zero
            dx /= norm
            dy /= norm
    
            for k in range(nBtst):
                _, _, a, b, angle = p_ell[i, j, k]
                opt_vec[i, j, k] = distance_to_ellipse_boundary(a, b, angle, dx, dy)
    
    # Sort distances across bootstraps (axis=-1)
    opt_vec_sorted = np.sort(opt_vec, axis=-1)
    
    # Extract lower and upper bounds
    opt_vec_sorted_lb = opt_vec_sorted[..., bds_idx[0]]
    opt_vec_sorted_ub = opt_vec_sorted[..., bds_idx[1]]
    
    return opt_vec, opt_vec_sorted, opt_vec_sorted_lb, opt_vec_sorted_ub

def intervals_overlap(ci1, ci2):
    """
    Determine overlap between corresponding rows of two (N, 2) arrays
    of confidence intervals.
    
    Parameters
    ----------
    ci1, ci2 : array_like, shape (N, 2)
        Arrays in which each row contains the lower and upper bounds
        of a 1-D confidence interval.
    
    Returns
    -------
    overlaps : ndarray, shape (N,)
        Boolean array indicating for each row whether the two intervals
        overlap or touch (True) or are disjoint (False).
    num_overlaps : int
        Total count of interval pairs that overlap or touch.
    p_overlaps : float
        Proportion of overlapping pairs, i.e. ``num_overlaps / N``.
    
    Raises
    ------
    ValueError
        If the inputs do not have identical shape, are not of shape
        (N, 2), or if any row violates the condition lower ≤ upper.
    """
    ci1 = np.asarray(ci1, dtype=float)
    ci2 = np.asarray(ci2, dtype=float)
    
    # ─── Shape checks ─────────────────────────────────────────────────────────
    if ci1.shape != ci2.shape:
        raise ValueError("ci1 and ci2 must have the same shape.")
    if ci1.ndim != 2 or ci1.shape[1] != 2:
        raise ValueError("Each input must be an array of shape (N, 2).")
    
    # ─── Validate that each interval satisfies lower ≤ upper ─────────────────
    if np.any(ci1[:, 1] < ci1[:, 0]) or np.any(ci2[:, 1] < ci2[:, 0]):
        raise ValueError("Each interval must satisfy lower ≤ upper.")
    
    # ─── Vectorized overlap test ─────────────────────────────────────────────
    # Two intervals [a, b] and [c, d] overlap (or touch) if max(a, c) ≤ min(b, d)
    overlaps = np.maximum(ci1[:, 0], ci2[:, 0]) <= np.minimum(ci1[:, 1], ci2[:, 1])
    
    # Count and proportion of overlaps
    num_overlaps = int(np.sum(overlaps))          # total overlapping pairs
    p_overlaps = num_overlaps / ci1.shape[0]      # proportion of overlaps
    
    return overlaps, num_overlaps, p_overlaps

def find_inner_outer_contours_nonellipse(pts_list, central_fraction = 0.95):
    """
    Compute the union and intersection contours for a set of 2D closed shapes,
    after discarding outlier contours based on their deviation from the first
    valid contour.
    
    Parameters
    ----------
    pts_list : list of array-like
        A list of 2D contours. Each element pts_list[i] is an array of shape
        (Ni, 2) containing the (x, y) coordinates of the i-th contour's
        vertices, ordered along the boundary. Ni may differ between contours.
    
    central_fraction : float, optional
        Fraction of contours (by count) to keep after sorting by their
        deviation from the first valid contour. Must be in (0, 1].
        - 1.0  -> keep all valid contours
        - 0.95 -> keep the central 95% (discard ~2.5% most deviant on each side).
    
    Returns
    -------
    xu, yu : np.ndarray
        1D arrays of x- and y-coordinates for the outer boundary of the
        union of all kept shapes.
    xi, yi : np.ndarray
        1D arrays of x- and y-coordinates for the outer boundary of the
        intersection of all kept shapes.
    
    Notes
    -----
    - Contours that are too short (< 4 points), contain NaNs/Infs, or have
      zero area are silently discarded before trimming.
    - Deviation from the first valid contour is measured as the area of the
      symmetric difference.
    - At least two valid contours must remain; otherwise a ValueError is raised.
    """
    
    if not isinstance(pts_list, (list, tuple)) or len(pts_list) < 2:
        raise ValueError("pts_list must be a list (or tuple) of at least 2 contours.")
    if not (0 < central_fraction <= 1.0):
        raise ValueError("central_fraction must be in the interval (0, 1].")
    
    # Convert all contours to polygons, skipping degenerate ones
    polys = []
    valid_indices = []
    for i, pts in enumerate(pts_list):
        contour = np.asarray(pts, dtype=float)
    
        if contour.ndim != 2 or contour.shape[1] != 2:
            raise ValueError(f"Contour {i} must have shape (N, 2).")
    
        # Drop rows with NaNs / inf
        good = np.all(np.isfinite(contour), axis=1)
        contour = contour[good]
    
        # Need at least 4 coordinates for a valid LinearRing
        if contour.shape[0] < 4:
            continue
    
        poly = Polygon(contour)
    
        # Skip invalid or zero-area polygons
        if (not poly.is_valid) or (poly.area == 0):
            continue
    
        polys.append(poly)
        valid_indices.append(i)
    
    if len(polys) < 2:
        raise ValueError(
            "After discarding degenerate contours, fewer than 2 valid contours remain."
        )
    
    n = len(polys)
    poly_ref = polys[0]
    
    # Measure deviation from the first valid contour via symmetric difference area
    distances = np.array([
        poly_ref.symmetric_difference(poly).area
        for poly in polys
    ])
    
    # Sort contour indices (in `polys` space) by distance from the reference
    idx_sorted = np.argsort(distances)
    
    # Determine how many contours to keep (central fraction)
    keep_n = max(2, int(round(central_fraction * n)))
    drop_n = n - keep_n
    left_drop = drop_n // 2
    right_drop = drop_n - left_drop
    
    keep_idx = idx_sorted[left_drop : n - right_drop]
    
    # Initialize union and intersection with the first kept polygon
    union_poly = polys[keep_idx[0]]
    intersection_poly = polys[keep_idx[0]]
    
    # Process remaining kept contours
    for idx in keep_idx[1:]:
        poly_i = polys[idx]
        union_poly = union_poly.union(poly_i)
        intersection_poly = intersection_poly.intersection(poly_i)
    
    # Extract outer boundaries (exterior) x/y for union and intersection
    if intersection_poly.is_empty:
        # up to you: raise, or return NaNs / empty arrays; here I raise
        raise ValueError("Intersection of kept contours is empty.")
    
    xu, yu = union_poly.exterior.xy
    xi, yi = intersection_poly.exterior.xy
    
    return np.array(xu), np.array(yu), np.array(xi), np.array(yi)

def find_btst_dataset_within_CI(metric_score, fitEll_btst, CI_percent=0.95,
                                sort_order="descending"):
    """
    Select bootstrap datasets whose metric scores fall within the specified
    confidence interval fraction.

    Parameters
    ----------
    metric_score : array-like of shape (n_btst,)
        Metric describing similarity between each bootstrap dataset and the
        original dataset. Larger or smaller values may be better depending on
        the metric used (see `sort_order`).

    fitEll_btst : array-like
        Fitted ellipses from bootstrap datasets. The first dimension indexes
        bootstrap samples.

    CI_percent : float, optional
        Fraction of bootstrap datasets to keep (default = 0.95).

    sort_order : {"descending", "ascending"}, optional
        Determines whether larger or smaller metric values are better.

        - "descending": larger metric values are better (e.g. NB similarity)
        - "ascending" : smaller metric values are better
                        (e.g. Bures-Wasserstein distance)

    Returns
    -------
    fitEll_btst_keep : ndarray
        Bootstrap ellipse fits retained for confidence interval construction.

    idx_keep_metric_score : ndarray
        Indices of retained bootstrap datasets.

    metric_score_sorted : ndarray
        Metric scores sorted according to `sort_order`.
    """

    metric_score = np.asarray(metric_score)
    fitEll_btst = np.asarray(fitEll_btst)

    if not (0 < CI_percent <= 1):
        raise ValueError("CI_percent must be in the interval (0, 1].")

    if sort_order == "descending":
        idx_sorted = np.argsort(metric_score)[::-1]
    elif sort_order == "ascending":
        idx_sorted = np.argsort(metric_score)
    else:
        raise ValueError('sort_order must be "descending" or "ascending".')

    metric_score_sorted = metric_score[idx_sorted]

    n_btst = metric_score.shape[0]
    n_keep = max(1, int(n_btst * CI_percent))

    idx_keep_metric_score = idx_sorted[:n_keep]
    fitEll_btst_keep = fitEll_btst[idx_keep_metric_score]

    return fitEll_btst_keep, idx_keep_metric_score, metric_score_sorted




