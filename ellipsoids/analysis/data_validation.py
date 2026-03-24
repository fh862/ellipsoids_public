#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:49:27 2025

@author: fangfang
"""
import numpy as np
from tqdm import trange
import pandas as pd

#%%
def shuffle_trials_within_levels(df, xref, x1, seed=None, x2=None):
    """
    Shuffle MOCS trials while preserving trial- and level-based structure.

    Trials are shuffled within each (trial, level) group, and the resulting
    shuffled indices are used to reorder xref, x1 (and optionally x2).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing trial information, with columns:
        - 'trial'           : trial number
        - 'shuffled_level'  : MOCS condition / level
    xref : np.ndarray
        Reference stimulus array, shape (N, ...).
    x1 : np.ndarray
        Comparison stimulus array #1, shape (N, ...).
    seed : int, optional
        Random seed for reproducibility.
    x2 : np.ndarray or None, optional
        Optional comparison stimulus array #2, shape (N, ...).
        If provided, it will be shuffled in the same way.

    Returns
    -------
    tuple
        If x2 is None:
            (xref_shuffled, x1_shuffled, shuffled_idx)
        If x2 is not None:
            (xref_shuffled, x1_shuffled, x2_shuffled, shuffled_idx)
    """
    rng = np.random.default_rng(seed)

    unique_trials = df["trial"].unique()
    unique_levels = df["level"].unique()

    # Initialize shuffled index array with placeholder (-1) of integer type
    shuffled_idx = np.full(len(df), -1, dtype=int)
    current_idx = 0

    # Build shuffled_idx by shuffling within each (trial, level) group
    for trial in unique_trials:
        for level in unique_levels:
            row_indices = df[
                (df["trial"] == trial) & (df["shuffled_level"] == level)
            ].index.to_numpy()

            if row_indices.size == 0:
                continue  # nothing to shuffle for this (trial, level)

            shuffled_rows = rng.permutation(row_indices)

            num_match = len(row_indices)
            shuffled_idx[current_idx : current_idx + num_match] = shuffled_rows
            current_idx += num_match

    # Now apply the shuffled indices once everything is filled
    xref_shuffled = xref[shuffled_idx]
    x1_shuffled = x1[shuffled_idx]

    outputs = [xref_shuffled, x1_shuffled]

    if x2 is not None:
        x2_shuffled = x2[shuffled_idx]
        outputs.append(x2_shuffled)

    outputs.append(shuffled_idx)

    return tuple(outputs)

#%%
class DataExport:
    """Small utilities for exporting grids / covariances / weights to CSV."""
    
    @staticmethod
    def cov_to_str(Sigmas, decimals= 8):
        """
        Convert a collection of N×N covariance matrices into CSV-friendly
        string representations.
    
        Each matrix is formatted as a nested list, e.g.:
            '[[a11,a12,...],[a21,a22,...],...]'
        with fixed decimal precision.
    
        Parameters
        ----------
        Sigmas : array_like, shape (N, N) or (M, N, N)
            One or more square covariance matrices.
        decimals : int, optional
            Number of decimal places to write (default: 8).
    
        Returns
        -------
        list of str
            List of formatted covariance matrices. If a single (N, N) matrix
            is provided, the returned list has length 1.
    
        Examples
        --------
        2×2:
        >>> covNxN_to_str_list([[[1, 0.1],[0.1, 2]]])
        ['[[1.00000000,0.10000000],[0.10000000,2.00000000]]']
    
        3×3:
        >>> covNxN_to_str_list([np.eye(3)])
        ['[[1.00000000,0.00000000,0.00000000],'
         '[0.00000000,1.00000000,0.00000000],'
         '[0.00000000,0.00000000,1.00000000]]']
        """
        Sigmas = np.asarray(Sigmas, dtype=float)
    
        # Promote single matrix to batch
        if Sigmas.ndim == 2:
            Sigmas = Sigmas[None, ...]
    
        if Sigmas.ndim != 3 or Sigmas.shape[1] != Sigmas.shape[2]:
            raise ValueError("Sigmas must have shape (N,N) or (M,N,N)")
    
        fmt = f"{{:.{decimals}f}}"
    
        out = []
        for S in Sigmas:
            rows = [
                "[" + ",".join(fmt.format(v) for v in row) + "]"
                for row in S
            ]
            out.append("[" + ",".join(rows) + "]")
    
        return out

    @staticmethod
    def idx_to_str(index, sep=","):
        """
        Convert an array of indices into CSV-friendly strings.
    
        This is used to export unrolled weight tensors of arbitrary dimension.
        For example:
          - 2D/4D W might be unrolled into (N, 4) indices (i,j,k,l)
          - 3D W might be unrolled into (N, 5) indices (i,j,k,l,m)
          - Other models may produce different numbers of index dimensions
    
        Parameters
        ----------
        index : array_like, shape (N, D) or (D,)
            Integer indices for each tensor entry. D is the number of tensor
            dimensions. If a single index vector (D,) is provided, it is treated
            as one row.
        sep : str, optional
            Separator used to join indices (default: ",").
    
        Returns
        -------
        list of str
            Length-N list of strings like "i,j,k,l" (or more/less fields depending on D).
    
        Examples
        --------
        >>> idx_to_str([[0,1,0,2],[4,4,1,0]])
        ['0,1,0,2', '4,4,1,0']
    
        >>> idx_to_str([3,2,1])
        ['3,2,1']
        """
        idx = np.asarray(index)
    
        # Promote a single (D,) vector to (1, D)
        if idx.ndim == 1:
            idx = idx[None, :]
    
        if idx.ndim != 2:
            raise ValueError("index must have shape (N, D) or (D,)")
    
        # Convert each row to a joined string (works for any D)
        return [sep.join(map(str, row)) for row in idx]

    @staticmethod
    def vec_to_str(X, decimals=8, sep=","):
        """
        Format an (N, D) array into CSV-friendly strings like "v1,v2,...,vD"
        with fixed decimals (works for D=2,3,4,...).
    
        Parameters
        ----------
        X : array_like, shape (N, D) or (D,)
            Vectors to format.
        decimals : int, optional
            Number of decimal places (default: 6).
        sep : str, optional
            Separator between values (default: ",").
    
        Returns
        -------
        list of str
            Length-N list of formatted vectors. If a single (D,) vector is provided,
            returns a list of length 1.
        """
        X = np.asarray(X, dtype=float)
    
        # Promote a single (D,) vector to (1, D)
        if X.ndim == 1:
            X = X[None, :]
    
        if X.ndim != 2:
            raise ValueError(f"X must have shape (N, D) or (D,); got {X.shape}.")
    
        fmt = f"{{:.{decimals}f}}"
        return [sep.join(fmt.format(v) for v in row) for row in X]
    
    @staticmethod
    def flatten_grid_and_sigmas(grid, Sigmas):
        """
        Flatten a grid of reference coordinates and the corresponding covariance
        matrices into row-aligned arrays.
    
        This helper is used when exporting per-grid-point ellipses/noise estimates
        to CSV. It converts inputs like:
            grid   : (n1, n2, D)
            Sigmas : (n1, n2, D, D)
        into:
            grid_flat   : (N, D)
            Sigmas_flat : (N, D, D)
        where N = n1*n2.
    
        Parameters
        ----------
        grid : array_like, shape (..., D) or (D,)
            Grid coordinates. The last dimension is the covariance dimension D.
            A single coordinate (D,) is also accepted.
        Sigmas : array_like, shape (..., D, D) or (D, D)
            Covariance matrices aligned with `grid` grid points. A single (D, D)
            matrix is also accepted (interpreted as one grid point).
    
        Returns
        -------
        grid_flat : np.ndarray, shape (N, D)
            Flattened coordinates; one row per grid point.
        Sigmas_flat : np.ndarray, shape (N, D, D)
            Flattened covariance matrices; one matrix per grid point.
    
        Raises
        ------
        ValueError
            If covariance matrices are not square, if D mismatches between grid and
            Sigmas, or if the number of grid points does not match the number of
            covariance matrices.
        """
        grid = np.asarray(grid)
        Sigmas = np.asarray(Sigmas)
    
        # Allow a single grid point + single covariance matrix
        #   grid  : (D,)   -> (1, D)
        #   Sigmas: (D, D) -> (1, D, D)
        if grid.ndim == 1:
            grid = grid[None, :]
        if Sigmas.ndim == 2:
            Sigmas = Sigmas[None, :, :]
    
        # Basic shape checks
        if Sigmas.shape[-1] != Sigmas.shape[-2]:
            raise ValueError(f"Sigmas must be square on the last two dims; got {Sigmas.shape}.")
    
        D = Sigmas.shape[-1]
        if grid.shape[-1] != D:
            raise ValueError(
                f"grid last dim must match covariance dimension D={D}; got grid shape {grid.shape}."
            )
    
        # Flatten everything into (N, D) and (N, D, D)
        grid_flat = grid.reshape(-1, D)
        Sigmas_flat = Sigmas.reshape(-1, D, D)
    
        # Ensure one covariance per grid point
        if grid_flat.shape[0] != Sigmas_flat.shape[0]:
            raise ValueError(
                f"Number of grid points ({grid_flat.shape[0]}) does not match number of "
                f"covariance matrices ({Sigmas_flat.shape[0]})."
            )
    
        return grid_flat, Sigmas_flat

    @staticmethod
    def export_ellipses_csv(grid, Sigmas, grid_col, sigma_col, out_path, decimals=8):
        """
        Export per-grid-point covariance matrices (ellipses) to a CSV file.
    
        Each row corresponds to one grid point:
          - `grid_col` stores the grid coordinate as a single comma-separated string
            (e.g., "x,y" or "x,y,z" depending on dimensionality).
          - `sigma_col` stores the covariance matrix as a single nested-list string
            (e.g., '[[s11,s12],[s21,s22]]' for 2×2, or the analogous form for N×N).
    
        Returns the DataFrame and the flattened numeric arrays for convenience.
        """
        # 1) Flatten the grid and covariance arrays so that:
        #      grid_flat   : (N, D)
        #      Sigmas_flat : (N, D, D)
        #    where N is the number of grid points.
        grid_flat, Sigmas_flat = DataExport.flatten_grid_and_sigmas(grid, Sigmas)
    
        # 2) Convert numeric arrays into compact, CSV-friendly string columns.
        #    (Storing as strings makes the CSV easy to inspect and parse downstream.)
        grid_str = DataExport.vec_to_str(grid_flat, decimals = decimals)       # e.g., "x,y" (or "x,y,z", ...)
        sigmas_str = DataExport.cov_to_str(Sigmas_flat, decimals = decimals)   # e.g., '[[...],[...]]'
    
        # 3) Assemble and write the table.
        df = pd.DataFrame({grid_col: grid_str, sigma_col: sigmas_str})
        df.to_csv(out_path, index=False)
    
        return df, grid_flat, Sigmas_flat

    @staticmethod
    def export_weights_csv(W, idx_col, val_col, out_path, decimals=8):
        """
        Export a weight tensor W into a long-form CSV (one row per tensor entry).
    
        The CSV contains:
          - `idx_col`: comma-separated index tuple for each entry (e.g., "i,j,k,l")
          - `val_col`: the corresponding weight value, written with fixed decimals
        """
        # Ensure numeric array (so rounding + float_format behave consistently)
        W = np.asarray(W, dtype=float)
    
        # Build an index table for every element in W.
        # Example: W shape (5,5,2,3) -> idx shape (150,4), each row is (i,j,k,l).
        idx = np.indices(W.shape).reshape(W.ndim, -1).T
    
        # Flatten values to align 1-to-1 with rows of `idx`
        vals = W.reshape(-1)
    
        # Pack indices into a single string column and write values with fixed precision
        df = pd.DataFrame({
            idx_col: DataExport.idx_to_str(idx),
            val_col: np.round(vals, decimals),
        })
    
        # float_format ensures CSV always writes exactly `decimals` digits (e.g., 0.20000000)
        df.to_csv(out_path, index=False, float_format=f"%.{decimals}f")
        return df, idx, vals
        
    @staticmethod
    def append_bootstrap_cov_columns(Sigmas_sorted, idx_desc, out_path,
                                     prefix = 'thres', decimals = 8):
        """
        Append bootstrap covariance columns to an existing CSV.
    
        - Reads CSV once
        - Builds all new columns in memory (avoids DataFrame fragmentation)
        - Writes CSV once
        """
        df = pd.read_csv(out_path)
        Sigmas_sorted = np.asarray(Sigmas_sorted)
        idx_desc = np.asarray(idx_desc)
    
        # Expect: (nSets, ...grid..., D, D)
        if Sigmas_sorted.ndim not in (5, 6):
            raise ValueError(
                f"Sigmas_sorted must be 5D or 6D with shape (nSets, ...grid..., D, D); "
                f"got shape {Sigmas_sorted.shape}."
            )
    
        if Sigmas_sorted.shape[-1] != Sigmas_sorted.shape[-2]:
            raise ValueError(f"Cov matrices must be square; got {Sigmas_sorted.shape[-2:]}.")
    
        nSets = Sigmas_sorted.shape[0]
        ndims_cov = Sigmas_sorted.shape[-1]
    
        # Grid size = product of grid dims between nSets and (ndims_cov,ndims_cov)
        grid_shape = Sigmas_sorted.shape[1:-2]
        nG = int(np.prod(grid_shape))
    
        if len(df) != nG:
            raise ValueError(
                f"Row mismatch for {out_path}: df has {len(df)} rows but expected {nG} "
                f"(grid_shape={grid_shape})."
            )
    
        if idx_desc.shape[0] != nSets:
            raise ValueError(f"idx_desc has length {len(idx_desc)} but need at least {nSets}.")
    
        # Build all new columns first (fast; avoids fragmentation)
        new_cols = {}
        existing = set(df.columns)
    
        for rank in trange(nSets, desc=f"Building {prefix} columns"):
            btst_id = int(idx_desc[rank])  # original bootstrap index (AEPsych[k])
    
            Sigmas_flat = Sigmas_sorted[rank].reshape(-1, ndims_cov, ndims_cov)
            col_name = f"Sigmas_{prefix}_grid_btst{btst_id}_rank{rank}"
    
            if col_name in existing or col_name in new_cols:
                raise ValueError(
                    f"Column already exists (or duplicate in build): {col_name}. "
                    f"Did you run append twice on the same file?"
                )
    
            new_cols[col_name] = DataExport.cov_to_str(Sigmas_flat, decimals=decimals)
    
        df_out = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
        df_out.to_csv(out_path, index=False)
        
    @staticmethod
    def append_bootstrap_weights_columns(weights_sorted, idx_desc, csv_path, 
                                         decimals=8):
        """
        Append bootstrap weight columns (numeric) and write with fixed specified decimals,
        e.g. 0.2 -> 0.20000000 in the CSV text.
        """
        df = pd.read_csv(csv_path)
    
        new_cols = {}
        nSets = weights_sorted.shape[0]
        for rank in trange(nSets, desc="Building weight columns"):
            btst_id = int(idx_desc[rank])
            w_flat = np.asarray(weights_sorted[rank]).reshape(-1).astype(float)
    
            col_name = f"W_btst{btst_id}_rank{rank}"
            new_cols[col_name] = w_flat  # keep numeric (no f-strings)
    
        df_out = pd.concat([df, pd.DataFrame(new_cols)], axis=1).copy()
    
        # Ensures CSV text uses exactly 8 decimal places (0.2 -> 0.20000000)
        df_out.to_csv(csv_path, index=False, float_format=f"%.{decimals}f")
        
    @staticmethod
    def build_trial_type_ae(nTrials_strat, nTrials_actual):
        """
        Build TrialType strings for AEPsych trials and optional pregenSobol trials.

        nTrials_strat: e.g. [300,300,300,5100]
            - first sum(nTrials_strat[:-1]) are Sobol-seeded AEPsych
            - last nTrials_strat[-1] are adaptive AEPsych
        nTrials_actual: total trials present in expt_trial (AEPsych + pregenSobol actually run)
        """
        nTrials_AEPsych = int(sum(nTrials_strat))
        nTrials_AEPsych_sobol = int(sum(nTrials_strat[:-1]))
        nTrials_pregenSobol = int(nTrials_actual - nTrials_AEPsych)

        trial_type = (
            [f"AEPsych_{i}_Sobol_{i}" for i in range(nTrials_AEPsych_sobol)] +
            [f"AEPsych_{nTrials_AEPsych_sobol + j}_adaptive_{j}" for j in range(nTrials_strat[-1])]
        )

        if nTrials_pregenSobol > 0:
            trial_type += [f"pregenSobol_{k}" for k in range(nTrials_pregenSobol)]

        return trial_type, nTrials_AEPsych, nTrials_AEPsych_sobol, nTrials_pregenSobol

    @staticmethod
    def build_trial_type_mocs(nRefs_MOCS, nLevels_MOCS, nTrials_perLevel):
        """
        Build TrialType strings for MOCS trials in (cond i, level j, repeat k) order.
        """
        return [
            f"MOCS_{i*(nLevels_MOCS*nTrials_perLevel) + j*nTrials_perLevel + k}"
            f"_cond_{i}_level_{j}_trial_{k}"
            for i in range(nRefs_MOCS)
            for j in range(nLevels_MOCS)
            for k in range(nTrials_perLevel)
        ]