#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 23:54:14 2026

@author: fangfang
"""

import numpy as np

class BinnedPC:
    def __init__(self, xref, x1, y):
        """
        Bin trials by direction (ref -> comp) and compute:
          - pC (mean of y) per bin
          - nTrials per bin
          - trial indices per bin

        Supports:
          - 2D: bin by theta
          - 3D: bin by (theta, elevation-u) or (theta, phi)

        Notes (3D):
          - Computes u = z = sin(phi), but bin_3d currently bins by phi edges
            when edges_phi_deg(...) is used.
          - Theta is always azimuth atan2(y, x) in [-pi, pi).
        """
        self.xref = np.asarray(xref)
        self.x1 = np.asarray(x1)
        self.y = np.asarray(y)

        if self.xref.shape != self.x1.shape:
            raise ValueError("xref and x1 must have the same shape.")
        if self.xref.shape[0] != self.y.shape[0]:
            raise ValueError("xref/x1 and y must have the same number of trials.")
        if self.xref.ndim != 2 or self.xref.shape[1] not in (2, 3):
            raise ValueError("xref/x1 must be shape (N,2) or (N,3).")

        self.ndim = self.xref.shape[1]

    def _delta(self):
        """
        Trial-wise displacement vector from reference to comparison.
        Shape: (N, ndim). For each trial i: d[i] = x1[i] - xref[i].
        """
        return self.x1 - self.xref
    
    def _unit_delta(self):
        """
        Normalize each displacement vector to unit length so we can talk about
        direction independent of step size.
        
        Returns:
          du: (N, ndim) unit direction vectors (0 where invalid)
          valid: (N,) boolean mask for trials with nonzero displacement
        """
        d = self._delta()
        n = np.linalg.norm(d, axis=1)  # (N,) Euclidean length per trial
        valid = n > 0                  # avoid division by zero
    
        if not np.all(valid):
            n_bad = np.sum(~valid)
            print(f"[BinnedPC] Warning: {n_bad} trial(s) have zero displacement (x1 == xref). "
                  "Their direction vectors are set to 0 and marked invalid.")
    
        du = np.zeros_like(d, dtype=float)
        du[valid] = d[valid] / n[valid, None]
        return du, valid
    
    def _theta(self):
        """
        Azimuth angle (in the x-y plane) of each trial's direction vector.
    
        Computation:
          1) d  = x1 - xref
          2) du = d / ||d||   (unit direction; only for trials with ||d|| > 0)
          3) theta_raw = atan2(du_y, du_x) ∈ [-pi, pi)
          4) theta = theta_raw mod (2*pi) ∈ [0, 2*pi)
    
        Returns:
          theta: (N,) azimuth angle in [0, 2*pi) (meaningful only where valid)
          valid: (N,) mask for trials with nonzero displacement (||x1-xref|| > 0)
        """
        du, valid = self._unit_delta()
        theta = np.arctan2(du[:, 1], du[:, 0])      # [-pi, pi)
        theta = np.mod(theta, 2 * np.pi)            # [0, 2*pi)
        return theta, valid
        
    def _u_and_phi(self):
        """
        Elevation information for 3D directions.
        
        u  : equal-area elevation coordinate, defined as z-component of the unit
             direction vector (u = du_z = sin(phi)); u ∈ [-1, 1].
        phi: elevation angle in radians, phi = arcsin(u), so phi ∈ [-pi/2, pi/2].
        
        Returns:
          u: (N,) equal-area elevation coordinate
          phi: (N,) elevation angle in radians
          valid: (N,) mask for nonzero displacement trials
        """
        du, valid = self._unit_delta()
        u = np.clip(du[:, 2], -1.0, 1.0)  # guard against tiny numerical overshoots
        phi = np.arcsin(u)
        return u, phi, valid

    def edges_theta_deg(self, step_deg, start_deg=0, end_deg=360):
        """
        Define azimuth (theta) bin edges and centers over [0°, 360°), and store them.
    
        Theta definition:
          theta_raw = atan2(du_y, du_x) in [-pi, pi)
          theta = mod(theta_raw, 2*pi) in [0, 2*pi)
    
        Parameters:
          step_deg  : bin width in degrees
          start_deg : first edge in degrees (default 0)
          end_deg   : last edge in degrees (default 360)
    
        Saves:
          self.theta_edges_rad    : (n_edges,) theta bin edges in radians
          self.theta_centers_rad  : (n_bins,)  theta bin centers in radians
          self.n_theta_bins       : int        number of theta bins (= n_edges - 1)
        """
        edges_deg = np.arange(start_deg, end_deg + step_deg, step_deg)
        centers_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    
        self.theta_edges_rad = np.deg2rad(edges_deg)
        self.theta_centers_rad = np.deg2rad(centers_deg)
        self.n_theta_bins = self.theta_centers_rad.size
    
    def edges_phi_deg(self, step_deg, start_deg=-90, end_deg=90):
        """
        Define elevation (phi) bin edges and centers, and store them on this object.
    
        Phi definition (3D):
          Let du be the unit direction vector from xref -> x1.
          We define the equal-area elevation coordinate u = du_z = sin(phi),
          and the elevation angle:
            phi = arcsin(du_z)  in [-pi/2, pi/2]  (i.e., [-90°, 90°]).
    
        Parameters:
          step_deg  : bin width in degrees
          start_deg : first edge in degrees (default -90)
          end_deg   : last edge in degrees (default  90)
    
        Saves:
          self.phi_edges_rad    : (n_edges,) phi bin edges in radians
          self.phi_centers_rad  : (n_bins,)  phi bin centers in radians
          self.n_phi_bins       : int        number of phi bins (= n_edges - 1)

        """
        edges_deg = np.arange(start_deg, end_deg + step_deg, step_deg)
        centers_deg = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    
        self.phi_edges_rad = np.deg2rad(edges_deg)
        self.phi_centers_rad = np.deg2rad(centers_deg)
        self.n_phi_bins = self.phi_centers_rad.size

    def bin_2d(self, min_trials_per_bin=1):
        """
        Bin 2D trials by azimuth direction (theta) of the unit displacement vector
        from reference to comparison.
    
        For each trial:
          d  = x1 - xref
          du = d / ||d||                      (only if ||d|| > 0)
          theta = atan2(du_y, du_x) ∈ [-pi, pi)
    
        Trials are assigned to bins using:
          bin_id = digitize(theta, theta_edges_rad) - 1
        Trials with ||d|| == 0 or theta outside the edge range are marked invalid
        and excluded from all bins.
    
        Requires (set by edges_theta_deg):
          self.theta_edges_rad    : (n_edges,) bin edges in radians
          self.theta_centers_rad  : (n_bins,)  bin centers in radians
          self.n_theta_bins       : int        number of bins
    
        Parameters:
          min_trials_per_bin : minimum number of trials required to report pC for a bin;
                               bins with fewer trials return pC = NaN.
    
        Returns:
          out : dict with keys
            pC            : (n_theta_bins,) mean of y within each bin (NaN if too few trials)
            nTrials       : (n_theta_bins,) number of trials in each bin
            idx           : list length n_theta_bins; each entry is an array of trial indices
            theta_edges   : (n_edges,) theta bin edges (radians)
            theta_centers : (n_theta_bins,) theta bin centers (radians)
        """
        if self.ndim != 2:
            raise ValueError("bin_2d requires xref/x1 to be 2D (N,2).")
    
        if not hasattr(self, "theta_edges_rad") or not hasattr(self, "theta_centers_rad"):
            raise ValueError("Theta bins not set. Call edges_theta_deg(...) first.")
    
        theta, valid = self._theta()
        idx_all = np.arange(theta.size)
    
        bin_id = np.digitize(theta, self.theta_edges_rad) - 1
        # mark invalid / out-of-range as -1
        bin_id[~valid] = -1
        bin_id[(bin_id < 0) | (bin_id >= self.n_theta_bins)] = -1
    
        idx_per_bin = [idx_all[bin_id == b] for b in range(self.n_theta_bins)]
        nTrials = np.array([len(ix) for ix in idx_per_bin], dtype=int)
    
        pC = np.full(self.n_theta_bins, np.nan, dtype=float)
        for b, ix in enumerate(idx_per_bin):
            if len(ix) >= min_trials_per_bin:
                pC[b] = float(np.mean(self.y[ix]))
                
        out_dict = dict(
            pC=pC,
            nTrials=nTrials,
            idx=idx_per_bin
        )
    
        self.perBin_data = out_dict
    
    def bin_3d(self, min_trials_per_bin=1):
        """
        Bin 3D trials by direction on the sphere using fixed angular bins in:
          - theta (azimuth): atan2(du_y, du_x) in [-pi, pi)
          - phi (elevation): arcsin(du_z) in [-pi/2, pi/2]
    
        For each trial:
          1) Compute displacement: d = x1 - xref
          2) Normalize: du = d / ||d||   (trials with ||d|| == 0 are invalid)
          3) Compute angles:
               theta = atan2(du_y, du_x)
               phi   = arcsin(du_z)
    
        Trials are assigned to 2D bins using precomputed edges stored on self:
          - self.theta_edges_rad, self.theta_centers_rad, self.n_theta_bins
          - self.phi_edges_rad,   self.phi_centers_rad,   self.n_phi_bins
        (These must be set beforehand via edges_theta_deg(...) and edges_phi_deg(...).)
    
        Within each (phi_bin, theta_bin), this method computes:
          - nTrials: number of trials in the bin
          - pC: mean response y in the bin (set to NaN if too few trials)
          - idx: indices of trials belonging to the bin
    
        Parameters
        ----------
        min_trials_per_bin : int
            Minimum number of trials required to compute pC for a bin.
            Bins with fewer trials return pC = NaN.
    
        Returns
        -------
        out_dict : dict with keys
            pC      : (n_phi_bins, n_theta_bins) mean(y) per bin (NaN if too few trials)
            nTrials : (n_phi_bins, n_theta_bins) number of trials per bin
            idx     : list-of-lists, idx[p][t] gives an array of trial indices for bin (p, t)
        """
        if self.ndim != 3:
            raise ValueError("bin_3d requires xref/x1 to be 3D (N,3).")
    
        if not hasattr(self, "theta_edges_rad") or not hasattr(self, "theta_centers_rad"):
            raise ValueError("Theta bins not set. Call edges_theta_deg(...) first.")
    
        _, phi, valid_phi = self._u_and_phi()   # elevation = phi
    
        # ---- compute theta for each trial ----
        theta, valid_theta = self._theta()
        valid = valid_theta & valid_phi
        idx_all = np.arange(theta.size)
    
        # ---- digitize into 2D bins ----
        theta_id = np.digitize(theta, self.theta_edges_rad) - 1
        elev_id  = np.digitize(phi,  self.phi_edges_rad) - 1
    
        # mark invalid / out-of-range as -1
        theta_id[~valid] = -1
        elev_id[~valid] = -1
        theta_id[(theta_id < 0) | (theta_id >= self.n_theta_bins)] = -1
        elev_id[(elev_id < 0) | (elev_id >= self.n_phi_bins)] = -1
    
        # build idx[p][t]
        idx_per_bin = [
            [idx_all[(elev_id == p) & (theta_id == t)] for t in range(self.n_theta_bins)]
            for p in range(self.n_phi_bins)
        ]
    
        nTrials = np.full((self.n_phi_bins, self.n_theta_bins), np.nan)
        pC = np.full((self.n_phi_bins, self.n_theta_bins), np.nan)
    
        for p in range(self.n_phi_bins):
            for t in range(self.n_theta_bins):
                ix = idx_per_bin[p][t]
                nTrials[p, t] = ix.size
                if ix.size >= min_trials_per_bin:
                    pC[p, t] = float(np.mean(self.y[ix]))
    
        out_dict = dict(
            pC=pC,
            nTrials=nTrials,
            idx=idx_per_bin,
        )
    
        self.perBin_data = out_dict
        return out_dict
