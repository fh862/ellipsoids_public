#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:56:01 2025

@author: fangfang
"""
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import dill as pickled
import re
from analysis.utils_load import select_files_and_get_paths

#%%
class expt_data:
    def __init__(self, xref_all, x1_all, y_all, pseudo_order = None):
        self.xref_all = xref_all
        self.x1_all = x1_all
        self.y_all = y_all
        self.pseudo_order = pseudo_order
        
class CrossValidation:    
    @staticmethod
    def shuffle_data(data, xref_unique = None, tol=5e-2, seed = None, 
                     nRepeats_shuffle = 1, debug_plot = False):
        """
        Shuffle the data separately for each unique reference location.

        Parameters
        ----------
        data : tuple of np.ndarray
            - Contains three arrays: `(y_all, xref_all, x1_all)`, where:
              - `y_all`: Measured responses (dependent variable), shape `(N,)`
              - `xref_all`: Reference locations tested in the experiment (independent variable),
                  shape `(N, M)`
              - `x1_all`: Comparison stimuli (independent variable), shape `(N, M)`

        xref_unique : np.ndarray, shape `(K, M)`
            - `K`: Number of unique reference locations.
            - Stores the unique reference locations used in the experiment.
        
        tol : float, optional (default=5e-2)
            - Tolerance for matching reference locations.
            - A trial is considered to match a reference if the absolute difference is below 
                `tol` in all dimensions.

        seed : int, optional
            - Seed for the random number generator to ensure reproducibility.

        Returns
        -------
        data_shuffled : tuple of np.ndarray
            - Contains three arrays: `(y_shuffled, xref_shuffled, x1_shuffled)`, each shuffled 
                within reference locations.
        """

        # Unpack input data
        y_all, xref_all, x1_all = data    

        # Set random seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)

        # if there is no unique xref, it means the task is 4d, not interleaved 2d
        if xref_unique is None:
            idx_shuffle = np.array(list(range(y_all.shape[0])))
            for _ in range(nRepeats_shuffle):  
                np.random.shuffle(idx_shuffle)
            y_shuffled = y_all[idx_shuffle]
            xref_shuffled = xref_all[idx_shuffle]
            x1_shuffled = x1_all[idx_shuffle]
        else:            
            # Initialize shuffled arrays with the same shape
            y_shuffled = np.empty_like(y_all)
            xref_shuffled = np.empty_like(xref_all)
            x1_shuffled = np.empty_like(x1_all)
        
            # Shuffle data within each unique reference location
            for ref_n in xref_unique:
                # Find the indices of trials matching the current reference location
                idx_match_original = np.where(np.all(np.abs(xref_all - ref_n) < tol, axis=1))[0]
    
                # Shuffle indices in place
                idx_match = np.array(idx_match_original)
                for _ in range(nRepeats_shuffle):
                    np.random.shuffle(idx_match)
    
                lb, ub = np.min(idx_match), np.max(idx_match)+1
                #put them in the appropirate array
                y_shuffled[lb:ub] = y_all[idx_match]
                xref_shuffled[lb:ub] = xref_all[idx_match]
                x1_shuffled[lb:ub] = x1_all[idx_match]
                
                if debug_plot:
                    fig, ax = plt.subplots(1, 2)
                    y_slc = y_all[idx_match_original]
                    x1_slc = x1_all[idx_match_original]
                    ax[0].scatter(x1_slc[y_slc == 1, 0], x1_slc[y_slc == 1, 1], color = 'g', s=1)
                    ax[0].scatter(x1_slc[y_slc == 0, 0], x1_slc[y_slc == 0, 1], color = 'r', marker = 'x',s=1)
                    ax[0].set_title('Before shuffling')
                    
                    yy_slc = y_shuffled[lb:ub]
                    xx1_slc = x1_shuffled[lb:ub]
                    ax[1].scatter(xx1_slc[yy_slc == 1, 0], xx1_slc[yy_slc == 1, 1], color = 'g')
                    ax[1].scatter(xx1_slc[yy_slc == 0, 0], xx1_slc[yy_slc == 0, 1], color = 'r', marker = 'x')
                    ax[1].set_title('After shuffling')

        return (y_shuffled, xref_shuffled, x1_shuffled)
    
    @staticmethod
    def select_NFold_data_noFixedRef(data, total_folds):
        """
        Split data into N folds for cross-validation, assigning any extra trials to the last fold.
        
        Parameters:
        -----------
        data : tuple of (y_all, xref_all, x1_all)
            Each element is an array of shape (nTrials_total, ...) representing the full dataset.
        total_folds : int
            Number of folds to split the data into.
        
        Returns:
        --------
        data_org : dict
            Dictionary with keys 0 to total_folds-1. Each value is a tuple:
            (training_data, validation_data, idx_keep, idx_heldout)
        """
        
        # Unpack input data
        y_all, xref_all, x1_all = data  
    
        # Validate data consistency
        if not (y_all.shape[0] == xref_all.shape[0] == x1_all.shape[0]):
            raise ValueError('Input size mismatch: All data arrays must have the same number')
        nTrials_total = y_all.shape[0]
    
        # Initialize the output dictionary
        data_org = {key: None for key in range(total_folds)}
        
        # Determine base number of trials per fold and remainder
        n_base = nTrials_total // total_folds
        n_remainder = nTrials_total % total_folds  # Extra trials go to the last fold
    
        # Compute lower and upper row indices for each fold
        row_lb_list = [n_base * i for i in range(total_folds)]
        row_ub_list = [n_base * (i + 1) for i in range(total_folds)]
        row_ub_list[-1] += n_remainder  # Add remainder to the last fold
    
        if row_ub_list[-1] != nTrials_total:
            raise ValueError(f'Index computation error: Final upper index {row_ub_list[-1]}'+\
                             f' does not match nTrials_total {nTrials_total}.')
                
        data_org['lb_idx'] = row_lb_list
        data_org['ub_idx'] = row_ub_list
        data_org['heldout_nTrials'] = [ub-lb for ub,lb in zip(row_ub_list, row_lb_list)]
        data_org['keep_nTrials'] = [nTrials_total - m for m in data_org['heldout_nTrials']]
            
        for n in range(total_folds):
            # Determine column indices for the held-out fold
            row_lb = row_lb_list[n]
            row_ub = row_ub_list[n]
            
            # Compute row indices
            idx_all = np.arange(nTrials_total)
            idx_heldout = idx_all[row_lb:row_ub]
            idx_keep = np.delete(idx_all, np.s_[row_lb:row_ub])
            
            # Extract held-out (validation) data
            y_heldout = y_all[idx_heldout]
            xref_heldout = xref_all[idx_heldout]
            x1_heldout = x1_all[idx_heldout]
            data_heldout = (y_heldout, xref_heldout, x1_heldout)
            
            # Extract training data
            y_keep = y_all[idx_keep]
            xref_keep = xref_all[idx_keep]
            x1_keep = x1_all[idx_keep]
            data_keep = (y_keep, xref_keep, x1_keep)
            
            data_org[n] = (data_keep, data_heldout, idx_keep, idx_heldout)
        
        return data_org
    
    
#%%
class TrialDistribution:
    def separate_edge_vs_central_trials_2Dplane(data, dim1_bd, dim2_bd=None, 
                                                ndims=2, tol = 1e-6):
        """
        Separates trials into 'edge' and 'central' based on whether the reference stimulus
        lies on the boundary of the specified dimensions in a 2D color plane.
    
        Parameters
        ----------
        data : tuple
            A tuple (y_all, xref_all, x1_all), where each element is an array of shape (N, ...)
            representing all trials.
        dim1_bd : np.ndarray of shape (2,)
            Boundary values (min and max) for the first stimulus dimension.
        dim2_bd : np.ndarray of shape (2,), optional
            Boundary values for the second dimension. If None, dim1_bd is used for both.
    
        Returns
        -------
        data_edge : tuple
            Subset of data where xref lies on the boundary (edges or corners).
        data_central : tuple
            Subset of data where xref lies in the interior of the plane.
        nTrials_edge: int
            The number of trials that lie on the boundary
        nTrials_central: int
            The number of trials that lie in the interior of the plane
            
        """
        
        # Unpack data
        y_all, xref_all, x1_all = data
        
        # Total number of trials
        nTrials_total = y_all.shape[0]
        
        # If second dimension boundary is not provided, use the first
        if dim2_bd is None:
            dim2_bd = dim1_bd
        bds = [dim1_bd, dim2_bd]
    
        # Collect indices of trials where xref is on the boundary in any dimension
        indices_edge = []
        for dim in range(ndims):
            for boundary_val in bds[dim]:
                indices_ij = np.where(np.abs(xref_all[:, dim] - boundary_val) < tol)[0]
                indices_edge.extend(indices_ij)
    
        # Ensure uniqueness (some points, like corners, may be counted twice)
        indices_edge = np.unique(indices_edge)
    
        # Find the complement: indices of trials with central (non-edge) reference stimuli
        indices_all = np.arange(nTrials_total)
        indices_central = np.setdiff1d(indices_all, indices_edge)
    
        # Sanity check to ensure completeness of split
        if len(indices_all) != len(indices_edge) + len(indices_central):
            raise ValueError("Mismatch in trial count: total trials must equal edge + central trials.")
    
        # Split data into edge and central sets based on indices
        data_edge = (y_all[indices_edge], xref_all[indices_edge], x1_all[indices_edge])
        data_central = (y_all[indices_central], xref_all[indices_central], x1_all[indices_central])
        
        # number of edge and central trials
        nTrials_edge = len(indices_edge)
        nTrials_central = len(indices_central)
        
        return data_edge, data_central, nTrials_edge, nTrials_central
            
    def truncate_edge_vs_central_trials_given_percentage(data, dim1_bd,
                                                         edge_percentage_desired, 
                                                         nLevels=None, nTrials_min=None,
                                                         dim2_bd=None, ndims=2, tol=1e-6):
        """
        Truncates either edge or central trials to achieve a desired edge-to-total trial percentage.
        
        Parameters
        ----------
        data : tuple
            A tuple (y_all, xref_all, x1_all), each of shape (N, ...), representing all trials.
        dim1_bd : np.ndarray of shape (2,)
            Boundary values (min and max) for the first stimulus dimension.
        edge_percentage_desired : float (0 - 1)
            Target fraction of edge trials out of total trials after truncation.
        nLevels : int, optional
            Number of levels to generate evenly spaced trial subsets (for sweep analysis).
        nTrials_min : int, optional
            Minimum number of trials in the lowest sweep level.
        dim2_bd : np.ndarray of shape (2,), optional
            Boundary values for the second dimension. If None, dim1_bd is used for both.
        ndims : int, optional
            Number of stimulus dimensions (default is 2).
        tol : float, optional
            Tolerance for determining whether values lie on the boundary (default is 1e-6).
        
        Returns
        -------
        vars_dict : dict
            Dictionary containing:
                - 'edge_percentage_desired': desired edge-to-total ratio
                - 'data_edge_trunc': truncated edge trial data
                - 'data_central_trunc': truncated central trial data
                - 'data_edge': original edge trial data (before truncation)
                - 'data_central': original central trial data (before truncation)
                - 'nTrials_edge': original number of edge trials
                - 'nTrials_central': original number of central trials
                - 'nTrials_total_keep': total number of trials after truncation
                - 'nTrials_edge_desired': number of edge trials retained
                - 'nTrials_central_desired': number of central trials retained
                - 'nTrials_levels': list of total trial counts for sweep analysis (if nLevels is set)
                - 'nTrials_edge_levels': corresponding number of edge trials for each level
                - 'nTrials_central_levels': corresponding number of central trials for each level
        """
    
        # Separate trials based on whether the reference stimulus is on the boundary
        data_edge, data_central, nTrials_edge, nTrials_central = \
            TrialDistribution.separate_edge_vs_central_trials_2Dplane(
                data, dim1_bd, dim1_bd, ndims=ndims, tol=tol
            )
    
        # Sanity check: total number of trials should match
        if nTrials_edge + nTrials_central != len(data[0]):
            raise ValueError("Mismatch in total trial count: edge + central != total")
    
        # Compute the observed edge trial percentage
        edge_percentage_org = nTrials_edge / (nTrials_edge + nTrials_central)
    
        if edge_percentage_desired > edge_percentage_org:
            # Desired percentage is higher than what we have — need to truncate central trials
            nTrials_edge_desired = nTrials_edge  # retain all edge trials
            nTrials_total_keep = int(nTrials_edge / edge_percentage_desired)
            nTrials_central_desired = nTrials_total_keep - nTrials_edge_desired
            data_edge_trunc = data_edge
            data_central_trunc = tuple(arr[:nTrials_central_desired] for arr in data_central)
        else:
            # Desired percentage is lower — need to truncate edge trials
            nTrials_central_desired = nTrials_central  # retain all central trials
            nTrials_total_keep = int(nTrials_central / (1 - edge_percentage_desired))
            nTrials_edge_desired = nTrials_total_keep - nTrials_central_desired
            data_edge_trunc = tuple(arr[:nTrials_edge_desired] for arr in data_edge)
            data_central_trunc = data_central
    
        # Generate evenly spaced levels of trial counts (optional)
        if nLevels is not None and nTrials_min is not None:
            nTrials_levels = np.linspace(nTrials_min, nTrials_total_keep, nLevels).astype(int)
            nTrials_edge_levels = (nTrials_levels * edge_percentage_desired).astype(int)
            nTrials_central_levels = nTrials_levels - nTrials_edge_levels
    
        vars_dict = {}
        for k in [
            'edge_percentage_org'
            'edge_percentage_desired',
            'data_edge_trunc',
            'data_central_trunc',
            'data_edge',
            'data_central',
            'nTrials_edge',
            'nTrials_central',
            'nTrials_total_keep',
            'nTrials_central_desired',
            'nTrials_edge_desired',
            'nTrials_levels',
            'nTrials_edge_levels',
            'nTrials_central_levels'
        ]:
            vars_dict[k] = locals().get(k, None)  # use .get to avoid KeyError
    
        return vars_dict
            
    def concatenate_data(data_list):
        """
        Concatenates a list of data tuples into a single tuple.
    
        Each element in the input list is a tuple (y_jnp, xref_jnp, x1_jnp), where:
            - y_jnp has shape (N,)
            - xref_jnp and x1_jnp have shape (N, 2)
    
        Parameters
        ----------
        data_list : list of tuples
            Each tuple contains (y_jnp, xref_jnp, x1_jnp) arrays of the same length N.
    
        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            A single tuple (y_jnp_cat, xref_jnp_cat, x1_jnp_cat) containing the concatenated arrays.
        """
        y_cat = jnp.concatenate([d[0] for d in data_list], axis=0)
        xref_cat = jnp.concatenate([d[1] for d in data_list], axis=0)
        x1_cat = jnp.concatenate([d[2] for d in data_list], axis=0)
    
        return y_cat, xref_cat, x1_cat
                
    def select_proportion_data(data_list, nTrials_includsion_list):
        """
        Truncates each dataset in the input list to a specified number of trials 
        and concatenates them.
    
        Parameters
        ----------
        data_list : list of tuples
            Each tuple contains (d_y, d_xref, d_x1), which represent response data,
            reference stimulus data, and comparison stimulus data, respectively.
        
        nTrials_includsion_list : list of int
            List of the number of trials to retain for each dataset in `data_list`.
            Must be the same length as `data_list`.
    
        Returns
        -------
        data_trunc_cat : tuple of arrays
            Concatenated data from all truncated datasets. The structure is the same as
            individual datasets: (d_y_cat, d_xref_cat, d_x1_cat).
        
        data_trunc_list : list of tuples
            List of truncated individual datasets, each with shape determined by its
            corresponding value in `nTrials_includsion_list`.
        
        Raises
        ------
        ValueError
            If the number of datasets does not match the number of trial counts specified.
        """
        
        # Validate input lengths
        if len(data_list) != len(nTrials_includsion_list):
            raise ValueError('The size of the input dataset does not match the '
                             'size of the included trials!')
    
        data_trunc_list = []
    
        # Truncate each dataset according to the number of trials specified
        for d, n in zip(data_list, nTrials_includsion_list):
            d_y, d_xref, d_x1 = d
            data_trunc_list.append((d_y[:n], d_xref[:n], d_x1[:n]))
    
        # Concatenate all truncated datasets
        data_trunc_cat = TrialDistribution.concatenate_data(data_trunc_list)
    
        return data_trunc_cat, data_trunc_list
    
    def extract_params_ell_grid(model):
        """
        Extracts ellipse parameters from a model's params_ell attribute and
        reshapes them into a (grid_h, grid_w, 5) numpy array.
    
        Parameters
        ----------
        model : object
            Model object with a nested list `params_ell` of shape (grid_h, grid_w),
            where each entry is an array of length 5.
    
        Returns
        -------
        np.ndarray
            A numpy array of shape (grid_h, grid_w, 5) containing the ellipse parameters.
        """
        grid_h = len(model.params_ell)
        grid_w = len(model.params_ell[0])
        params_ell_grid = np.zeros((grid_h, grid_w, 5))
        for k in range(grid_h):
            for l in range(grid_w):
                params_ell_grid[k, l] = model.params_ell[k][l]
        return params_ell_grid

    def load_trial_efficiency_results(flag_btst=[False], btst_seed=[None]):
        """
        Loads performance and model prediction results across multiple bootstrap seeds and trial counts.
    
        Parameters
        ----------
        flag_btst : list of bool
            Boolean list indicating whether to include bootstrap seed.
        btst_seed : list of int
            List of bootstrap seeds.
    
        Returns
        -------
        tuple
            (
                BWD_dict : dict with keys
                    - 'nTrials_inclusion' : list of trial counts
                    - 'BWD_sum_all' : (nSets, nSeeds)
                    - 'BWD_all' : (nSets, nSeeds, grid, grid)
                    - 'BWD_sum_all_org' : (nSets,)
                    - 'BWD_sum_all_btst' : (nSets, nSeeds - 1)
                    - 'BWD_CI' : list of lower and upper bounds from bootstrap samples
                params_ell : ndarray
                    Ellipse parameters of shape (nSets, nSeeds, grid, grid, 5)
                sorted_paths : list of str
                    Sorted list of file paths
                model_pred : list
                    Nested list of model predictions per (set, seed)
            )
            or an error message string if loading fails.
        """
        try:
            full_path_set = select_files_and_get_paths()
            if not full_path_set: raise ValueError("No files selected.")
    
            # Extract trial counts from filenames
            nTrials_inclusion = []
            for path in full_path_set:
                match = re.search(r'(\d+)trials', path)
                if match: nTrials_inclusion.append(int(match.group(1)))
            if not nTrials_inclusion: raise ValueError("No trial numbers found in file names.")
            print("Included trial numbers:", nTrials_inclusion)
    
            # Sort paths and trial counts together
            sorted_pairs = sorted(zip(nTrials_inclusion, full_path_set))
            nTrials_inclusion, sorted_paths = zip(*sorted_pairs)
            nSets = len(sorted_paths)
            nSeeds = len(flag_btst)
    
            # Initialize lists
            BWD_sum_all_list, BWD_all_list, params_ell_list, model_pred_list = [], [], [], []
            vars_dict_all_list = []
    
            for idx, path in enumerate(sorted_paths):
                model_pred_idx_list, BWD_sum_seed_list, BWD_seed_list  = [],[],[]
                vars_dict_list = []
    
                for flag_btst_AEPsych, seed in zip(flag_btst, btst_seed):
                    str_ext = f'_btst_AEPsych[{seed}]' if flag_btst_AEPsych else ''
                    full_path = f'{path[:-4]}{str_ext}.pkl'
    
                    with open(full_path, 'rb') as f:
                        vars_dict = pickled.load(f)
                    vars_dict_list.append(vars_dict)
    
                    # Load one seed
                    BWD_sum_seed_list.append(vars_dict['BWD_sum'])
                    BWD_seed_list.append(vars_dict['BWD'])
    
                    model_pred_idx = vars_dict['model_pred_Wishart']
                    model_pred_idx_list.append(model_pred_idx)
    
                    # Infer grid size and extract ellipse parameters                            
                    params_ell_grid = TrialDistribution.extract_params_ell_grid(model_pred_idx)
    
                BWD_sum_all_list.append(BWD_sum_seed_list)
                BWD_all_list.append(BWD_seed_list)
                model_pred_list.append(model_pred_idx_list)
                vars_dict_all_list.append(vars_dict_list)
    
            # Convert to arrays
            BWD_sum_all = np.array(BWD_sum_all_list)  # (nSets, nSeeds)
            BWD_all = np.array(BWD_all_list)          # (nSets, nSeeds, grid, grid)
            params_ell = np.array(params_ell_list)    # (nSets, nSeeds, grid, grid, 5)
    
            # Post-processing
            BWD_sum_all_org = BWD_sum_all[:, 0]
            BWD_sum_all_btst = np.sort(BWD_sum_all[:, 1:], axis=1) if nSeeds > 1 else np.empty((nSets, 0))
            BWD_CI = [BWD_sum_all_btst[:, 0], BWD_sum_all_btst[:, -1]] if BWD_sum_all_btst.shape[1] > 0 else None
    
            BWD_dict = {
                'nTrials_inclusion': list(nTrials_inclusion),
                'BWD_sum_all': BWD_sum_all,
                'BWD_all': BWD_all,
                'BWD_sum_all_org': BWD_sum_all_org,
                'BWD_sum_all_btst': BWD_sum_all_btst,
                'BWD_CI': BWD_CI
            }
    
            return BWD_dict, list(sorted_paths), vars_dict_all_list
    
        except Exception as e:
            return f"[ERROR] Failed to load trial efficiency results: {e}"