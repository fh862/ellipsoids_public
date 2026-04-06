#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:33:29 2024

@author: fangfang
"""

import os
import dill as pickled
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List
import re
import warnings

#%%
@dataclass(frozen=True)
class ColorTransFilenames:
    """
    date : str
        String specifying which calibration to use. This determines which
        set of transformation matrices are loaded via `self._find_exact_path`.
        Currently supported values:

        - "02242025"
            Calibration used to compute transformation matrices for the 4D
            color discrimination experiment on the isoluminant plane
            (Hong et al., 2025, eLife).

        - "07292025"
            Calibration used for the pilot 6D color discrimination experiment
            spanning the full 3D RGB cube.

        - "10062025"
            Calibration used for the adaptation experiment (gray vs. blue)
            on the isoluminant plane.

        - "11172025"
            Same underlying calibration as "10062025", but with transformation
            matrices corresponding to the L–S–isolating plane.
    
    """
    date: str = "02242025"   # default MMDDYYYY
    prefix: str = "DELL"
    suffix: str = "copy"

    def __post_init__(self):
        if not re.fullmatch(r"\d{8}", self.date[:8]):
            raise ValueError("date must be MMDDYYYY, e.g., '02242025'.")

    @property
    def cal_tag(self) -> str:
        return f"{self.prefix}_{self.date}_{self.suffix}"

    @property
    def M_2DWToRGB_filename(self) -> str:
        return f"M_2DWToRGB_{self.cal_tag}.csv"

    @property
    def M_RGBTo2DW_filename(self) -> str:
        return f"M_RGBTo2DW_{self.cal_tag}.csv"

    @property
    def M_2DWToDKL_filename(self) -> str:
        return f"M_2DWToDKLPlane_{self.cal_tag}.csv"

    @property
    def M_DKLTo2DW_filename(self) -> str:
        return f"M_DKLPlaneTo2DW_{self.cal_tag}.csv"
    
    @property
    def M_LMSToRGB_filename(self) -> str:
        return f"M_LMSToRGB_{self.cal_tag}.csv"
    
    @property
    def M_RGBToLMS_filename(self) -> str:
        return f"M_RGBToLMS_{self.cal_tag}.csv"
    
@dataclass(frozen=True)
class ModelFitFilenames:
    base_2d_filename: str = 'Fitted_isothreshold_{plane}_sim240perCond_samplingNearContour_jitter0.1_seed0_bandwidth0.005_oddity.pkl'
    isoluminant_filename: str = 'Fitted_isothreshold_Isoluminant plane{cie}_sim18000total_samplingNearContour_jitter0.3_seed0_bandwidth0.005_decay0.4_oddity.pkl'
    ellipsoid_3d_filename: str = 'Fitted_isothreshold_ellipsoids_sim240perCond_samplingNearContour_jitter0.3_seed0{cie}_bandwidth0.005_oddity.pkl'

    @staticmethod
    def get_filename(plane_2D: str, cie_version: str = '', is_3d: bool = False, manual_input: bool = False) -> str:
        cie_suffix = '' if cie_version in {'', 'CIE1976'} else f'_{cie_version}'
        if is_3d:
            return ModelFitFilenames().ellipsoid_3d_filename.format(cie=cie_suffix)
        if manual_input:
            return None
        if plane_2D == 'Isoluminant plane':
            return ModelFitFilenames().isoluminant_filename.format(cie=cie_suffix)
        else:
            return ModelFitFilenames().base_2d_filename.format(plane=plane_2D)

@dataclass(frozen=True)
class CIEDataFilenames:
    isoluminant_2d: str = 'Isothreshold_ellipses_isoluminant{cie}.pkl'
    general_2d: str = 'Isothreshold_ellipses_3slices{cie}.pkl'
    ellipsoid_3d: str = 'Isothreshold_ellipsoid_CIELABderived{cie}.pkl'

    @staticmethod
    def get_filename(color_dimension: int, plane_2D: str = '', cie_version: str = '') -> str:
        cie_suffix = f'_{cie_version}'
        if color_dimension == 2:
            if plane_2D == 'Isoluminant plane':
                return CIEDataFilenames().isoluminant_2d.format(cie=cie_suffix)
            else:
                return CIEDataFilenames().general_2d.format(cie=cie_suffix)
        else:
            return CIEDataFilenames().ellipsoid_3d.format(cie=cie_suffix)

@dataclass(frozen=True)
class ManualFitFilenames:
    file_name_part1: str = field(default='Fitted_isothreshold_isoluminant_plane_360trialsPerRef_9refs_AEPsychSampling_bandwidth0.005_sub', init=False)

    @property
    def full_names(self) -> List[str]:
        return [f"{self.file_name_part1}{i}.pkl" for i in range(1, 6)]

    @property
    def simplified_names(self) -> List[str]:
        return [f"Fitted_isothreshold_isoluminant_plane_sub{i}.pkl" for i in range(1, 6)]
    
#%%
class color_thresholds():
    def __init__(self, color_dimension, data_base_dir, plane_2D = "Isoluminant plane", 
                 fixed_value = None, manual_input = False):
        """
        Initializes the color_thresholds class with various configuration settings
        for handling color dimensionality, either in 2D or 3D analysis.

        Args:
            color_dimension (int): 
                2 or 3, indicating the dimensionality of the color analysis.
            data_base_dir (str): 
                Base directory path where the data files are stored.
            fixed_value (float, optional, N unit): 
                Fixed parameter value used in file naming, relevant for 2D analysis.
            plane_2D (str, optional): 
                Specific plane ('GB plane', 'RB plane', 'RG plane', or 'Isoluminant plane') if in 2D.
            fixed_color_dim (int, optional): 
                Index of the fixed color dimension in 2D analyses.
        """
        if color_dimension not in (2, 3):
            raise ValueError("color_dimension must be either 2 or 3.")
        
        self.color_dimension = color_dimension
        self.fixed_value = fixed_value
        self.plane_2D = plane_2D
        self.manual_input = manual_input
        self.base_path = data_base_dir
        self.CIE_data = dict()
        self.Wishart_data = dict()

        if self.color_dimension == 2:
            self._set_plane_config()
                    
    def _set_plane_config(self):
        """Set fixed and varying dimensions for different 2D planes."""
        plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
        if self.plane_2D in plane_2D_dict:
            self.fixed_color_dim = plane_2D_dict[self.plane_2D]
            self.varying_color_dim = [i for i in range(3) if i != self.fixed_color_dim]
        elif self.plane_2D == 'Isoluminant plane':
            self.fixed_color_dim = 2  # that dimension will have 1's as filler
            self.varying_color_dim = [0, 1]
        elif self.plane_2D == 'LSisolating plane':
            self.fixed_color_dim = 1  
            self.varying_color_dim = [0, 2]            
        else:
            raise ValueError("Unsupported plane_2D value for 2D analysis.")
        
    def _find_exact_path(self, file_name):
        """
        Searches for the exact path of a specified file within the base directory.
        """
        for root, dirs, files in os.walk(self.base_path):
            if file_name in files:
                return os.path.join(root, file_name)
        raise FileNotFoundError(f"Data files directory not found for file {file_name}.")
        
    def _file_selection_popup(self):
        """
        Displays a list of file options in the console, assigns an ID to each, 
        and allows the user to select a file by entering its ID. 
        Returns the selected file name.
        """
        files = ManualFitFilenames()
        print("Please select a file by entering its corresponding ID:")
        for idx, name in enumerate(files.simplified_names, 1):
            print(f"{idx}: {name}")
    
        while True:
            try:
                selected_id = int(input("Enter the ID of your selection: "))
                if 1 <= selected_id <= len(files.full_names):
                    return files.full_names[selected_id - 1]
                else:
                    print(f"Invalid ID. Please enter a number between 1 and {len(files.full_names)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    #%%
    def load_CIE_data(self, CIE_version = '', num_grid_pts = 5):
        """
        Loads CIE data for color discrimination analysis, supporting both 2D and 3D data.
        """
        if CIE_version not in {'', 'CIE1994', 'CIE1976', 'CIE2000'}:
            raise ValueError('Only CIE versions 1976, 1994, and 2000 are supported.')
    
        file_name = CIEDataFilenames.get_filename(
            color_dimension=self.color_dimension,
            plane_2D=self.plane_2D if self.color_dimension == 2 else '',
            cie_version=CIE_version
        )
    
        file_path = self._find_exact_path(file_name)
        self.file_path_CIE_data = file_path
        with open(file_path, 'rb') as f:
            data = pickled.load(f)
    
        stim_key = f'stim{self.color_dimension}D'
        results_key = f'results{self.color_dimension}D'
        
        print(f"Loaded CIELab simulated data for {self.color_dimension}D case. \n"+\
              f"Number of grid points = {num_grid_pts}")
        if self.color_dimension == 2 and self.plane_2D != 'Isoluminant plane':
            self.CIE_data[stim_key] = data[f'stim_grid{num_grid_pts}_fixedVal{self.fixed_value}']
            self.CIE_data[results_key] = data[f'results_grid{num_grid_pts}_fixedVal{self.fixed_value}']                
        else:
            self.CIE_data[stim_key] = data[f'stim_grid{num_grid_pts}']
            self.CIE_data[results_key] = data[f'results_grid{num_grid_pts}']
            
    def load_model_fits(self, CIE_version='CIE1994'):
        """
        Loads model fitting data based on the specified color plane and CIE version.
        """
        if CIE_version not in {'', 'CIE1994', 'CIE1976', 'CIE2000'}:
            raise ValueError('Only CIE versions 1976, 1994, and 2000 are supported.')
    
        if self.color_dimension == 2:
            if self.manual_input:
                file_name = self._file_selection_popup()
            else:
                file_name = ModelFitFilenames.get_filename(
                    plane_2D=self.plane_2D,
                    cie_version=CIE_version,
                    is_3d=False,
                    manual_input=False
                )
        else:
            file_name = ModelFitFilenames.get_filename(
                plane_2D=None,
                cie_version=CIE_version,
                is_3d=True
            )
    
        file_path = self._find_exact_path(file_name)
        self.file_path_model_fits = file_path
        with open(file_path, 'rb') as f:
            self.Wishart_data = pickled.load(f)
    
        print("Loaded Wishart model fits.")
        
    def load_transformation_matrix(self, file_date = "02242025"):
        """
        Load color transformation matrices for a specified calibration date.
    
        Parameters
        ----------
        file_date : str, optional
            String specifying which calibration to use. This determines which
            set of transformation matrices are loaded via `self._find_exact_path`.
            Currently supported values:
    
            - "02242025"
                Calibration used to compute transformation matrices for the 4D
                color discrimination experiment on the isoluminant plane
                (Hong et al., 2025, eLife).
    
            - "07292025"
                Calibration used for the pilot 6D color discrimination experiment
                spanning the full 3D RGB cube.
    
            - "10062025"
                Calibration used for the adaptation experiment (gray vs. blue)
                on the isoluminant plane.
    
            - "11172025"
                Same underlying calibration as "10062025", but with transformation
                matrices corresponding to the L–S–isolating plane.
    
        Notes
        -----
        This method:
        - Instantiates a `ColorTransFilenames` object with the given `file_date`.
        - Locates and loads the RGB ↔ 2DW transformation matrices:
            * M_2DWToRGB
            * M_RGBTo2DW
        - Locates and loads the DKL ↔ 2DW transformation matrices:
            * M_2DWToDKL
            * M_DKLTo2DW
        - Locates and loads the LMS ↔ 2DW transformation matrices:
            * M_2DWToLMS
            * M_LMSTo2DW

        """
        
        #save the file paths
        filenames = ColorTransFilenames(file_date)  # instantiate with default cal_date or provide one if needed
        print(f'The transformation matrices are based on the calibration done on {file_date}.')
        
        try: 
            self.file_path_M_2DWToRGB = self._find_exact_path(filenames.M_2DWToRGB_filename)
            self.file_path_M_RGBTo2DW = self._find_exact_path(filenames.M_RGBTo2DW_filename)
            self.M_2DWToRGB = np.array(pd.read_csv(self.file_path_M_2DWToRGB, header=None))
            self.M_RGBTo2DW = np.array(pd.read_csv(self.file_path_M_RGBTo2DW, header=None))
            print('Loaded transformation matrices between RGB and 2DW.')
        except Exception:
            # If files/paths aren't available for this calibration, keep as None
            print("Transformation matrices between RGB and 2DW are not found for this calibration date.")
            
        # Try to load DKL matrices (optional)
        try:
            self.file_path_M_2DWToDKL = self._find_exact_path(filenames.M_2DWToDKL_filename)
            self.file_path_M_DKLTo2DW = self._find_exact_path(filenames.M_DKLTo2DW_filename)
    
            self.M_2DWToDKL = np.array(pd.read_csv(self.file_path_M_2DWToDKL, header=None))
            self.M_DKLTo2DW = np.array(pd.read_csv(self.file_path_M_DKLTo2DW, header=None))
            print("Loaded transformation matrices between 2DW and DKL.")
        except Exception:
            # If files/paths aren't available for this calibration, keep as None
            print("Transformation matrices between DKL and 2DW are not found for this calibration date.")
            
        # Try to load LMS transformation matrices (optional)
        try:
            self.file_path_M_LMSToRGB = self._find_exact_path(filenames.M_LMSToRGB_filename)
            self.file_path_M_RGBToLMS = self._find_exact_path(filenames.M_RGBToLMS_filename)
    
            self.M_LMSToRGB = np.array(pd.read_csv(self.file_path_M_LMSToRGB, header=None))
            self.M_RGBToLMS = np.array(pd.read_csv(self.file_path_M_RGBToLMS, header=None))
            print("Loaded transformation matrices between RGB and LMS.")
        except Exception:
            # If files/paths aren't available for this calibration, keep as None
            print("Transformation matrices between RGB and LMS are not found for this calibration date.")
                    
    def get_data(self, key, dataset='CIE_data'):
        """
        Retrieves specific data stored within the class, either from CIE_data or Wishart_data.

        Args:
            key (str): Key for the data to retrieve.
            dataset (str): Specifies which dataset to query, default is 'CIE_data'.

        Returns:
            The data associated with the provided key.

        Raises:
            ValueError: If the specified dataset does not exist.
            KeyError: If the key does not exist in the specified dataset.
        """
        if dataset == 'Wishart_data':
            if key not in self.Wishart_data:
                print(self.Wishart_data.keys())
                raise KeyError(f"Key '{key}' does not exist in Wishart_data.")
            return self.Wishart_data[key]
        elif dataset == 'CIE_data':
            if key not in self.CIE_data:
                print(self.CIE_data.keys())
                raise KeyError(f"Key '{key}' does not exist in CIE_data.")
            return self.CIE_data[key]
        else:
            raise ValueError(f"Dataset '{dataset}' does not exist.")

    def rgb_to_2DW(self, rgb, tol=1e-2, return_2DW = True):
        """
        Convert RGB values in [0, 1] to W-space using the homogeneous transform
        `M_RGBTo2DW`.
    
        The transform is expected to output a 3-vector per input:
            [W1, W2, 1]
        where the last component is a homogeneous coordinate that should be ~1.
        This method checks that assumption within `tol` and warns if violated.
    
        Parameters
        ----------
        rgb : array_like
            RGB values with shape (3,), (N, 3), or (3, N). Values are typically in [0, 1].
        tol : float, optional
            Tolerance for validating the homogeneous coordinate (last component) is ~1.
        return_2DW : bool, optional
            If True (default), return only the 2D W coordinates with shape (N, 2).
            If False, return the full homogeneous W coordinates with shape (N, 3).
    
        Returns
        -------
        W_out : ndarray
            If return_2DW is True:  array of shape (N, 2) containing (W1, W2).
            If return_2DW is False: array of shape (N, 3) containing (W1, W2, 1).
        """
        rgb = np.asarray(rgb, dtype=float)
    
        # Normalize input to shape (N, 3)
        if rgb.shape == (3,):
            rgb2 = rgb[None, :]
        elif rgb.ndim == 2 and rgb.shape[1] == 3:
            rgb2 = rgb
        elif rgb.ndim == 2 and rgb.shape[0] == 3:
            rgb2 = rgb.T
        else:
            raise ValueError(f"`rgb` must have shape (3,), (N,3), or (3,N); got {rgb.shape}")
    
        # Apply transform. Expected: M_RGBTo2DW shape (3, 3), rgb2.T shape (3, N)
        W = (self.M_RGBTo2DW @ rgb2.T).T  # (N, 3)
    
        # Homogeneous coordinate check: last column should be ~1
        w_last = W[:, -1]
        max_dev = np.max(np.abs(w_last - 1)) if w_last.size else 0.0
        if max_dev > tol:
            bad = np.where(np.abs(w_last - 1) > tol)[0]
            msg = (
                f"rgb_to_2DW: homogeneous coordinate deviates from 1 by up to {max_dev:.3g} "
                f"(tol={tol}). Example indices: {bad[:5].tolist()}."
            )
            warnings.warn(msg, RuntimeWarning)
    
        return W[:, :2] if return_2DW else W
    
    def W2D_to_rgb(self, W2D, tol=1e-6):
        """
        Convert 2D W-space coordinates in [-1, 1] to RGB via the homogeneous transform
        `M_2DWToRGB`.
    
        Parameters
        ----------
        W2D : array_like
            W-space coordinates with shape (2,), (N, 2), or (2, N).
        tol : float, optional
            Tolerance for gamut checking. Values slightly outside [0, 1] within `tol`
            are treated as numerical noise.
    
        Returns
        -------
        rgb : ndarray, shape (N, 3)
            RGB values. If W2D is out of gamut, values may fall outside [0, 1].
        """
        W2D = np.asarray(W2D, dtype=float)
        
        if W2D.size == 0:
            # Preserve a sensible empty shape: (0, 3)
            return np.empty((0, 3), dtype=float)
    
        # Normalize input to a consistent (N, 2) representation:
        #   - (2,)   -> (1, 2)   (single point)
        #   - (N, 2) -> (N, 2)   (N points)
        #   - (2, N) -> (N, 2)   (transpose convention)
        # Note: for shape (2, 2) we assume rows index points (N=2).
        if W2D.shape == (2,):
            W = W2D[None, :]
        elif W2D.ndim == 2 and W2D.shape[1] == 2:
            W = W2D
        elif W2D.ndim == 2 and W2D.shape[0] == 2:
            W = W2D.T
        else:
            raise ValueError(f"`W2D` must have shape (2,), (N,2), or (2,N); got {W2D.shape}")
    
        # Convert to homogeneous coordinates by appending a column of ones:
        #   [w1, w2] -> [w1, w2, 1]
        # Then apply the 3x3 affine transform to obtain RGB.
        ones = np.ones((W.shape[0], 1), dtype=W.dtype)
        W_h = np.hstack([W, ones])              # (N, 3)
        rgb = (self.M_2DWToRGB @ W_h.T).T       # (N, 3)
    
        # Gamut check + clipping
        rgb_min = float(np.min(rgb))
        rgb_max = float(np.max(rgb))
        
        # How far outside [0, 1] are we?
        under = max(0.0, 0.0 - rgb_min)        # amount below 0
        over  = max(0.0, rgb_max - 1.0)        # amount above 1
        max_violation = max(under, over)
        
        if max_violation > tol:
            bad = np.where((rgb < -tol) | (rgb > 1.0 + tol))
            nbad = bad[0].size
            warnings.warn(
                f"W2D_to_rgb: out-of-gamut RGB values detected (min={rgb_min:.4g}, max={rgb_max:.4g}, tol={tol}). "
                f"{nbad} entries exceed [0,1] by more than tol. Clipping to [0,1]. "
                f"Example indices (row, channel): {list(zip(bad[0][:5].tolist(), bad[1][:5].tolist()))}.",
                RuntimeWarning
            )
        
        # Always clip (either tiny numerical drift or true out-of-gamut)    
        return np.clip(rgb, 0.0, 1.0)
    
    def W3D_to_cc(self, W3D, ambient_lms, bg_lms):
        """
        Convert 3D model-space coordinates to cone-contrast coordinates.

        Parameters
        ----------
        W3D : array-like, shape (3,) or (N, 3)
            Input point(s) in 3D W-space.
        ambient_lms : array-like, shape (3,) or (N, 3)
            Ambient LMS offset. A single `(3,)` vector is broadcast to all
            input points; an `(N, 3)` array specifies a different ambient LMS
            for each input point.
        bg_lms : array-like, shape (3,) or (N, 3)
            Background LMS used to compute cone contrast. A single `(3,)`
            vector is broadcast to all input points; an `(N, 3)` array
            specifies a different background LMS for each input point.

        Returns
        -------
        cc : np.ndarray
            Cone-contrast coordinates with the same shape as `W3D`:
            `(3,)` for a single point or `(N, 3)` for a batch.
        """
        W3D = np.asarray(W3D, dtype=float)
        was_1d = W3D.ndim == 1

        if was_1d:
            if W3D.shape[0] != 3:
                raise ValueError("W3D must have shape (3,) or (N, 3).")
            W3D_eval = W3D[None, :]
        elif W3D.ndim == 2 and W3D.shape[1] == 3:
            W3D_eval = W3D
        else:
            raise ValueError("W3D must have shape (3,) or (N, 3).")

        ambient_lms = np.asarray(ambient_lms, dtype=float)
        bg_lms = np.asarray(bg_lms, dtype=float)

        if ambient_lms.shape == (3,):
            ambient_lms_eval = np.broadcast_to(ambient_lms, W3D_eval.shape)
        elif ambient_lms.shape == W3D_eval.shape:
            ambient_lms_eval = ambient_lms
        else:
            raise ValueError("ambient_lms must have shape (3,) or match W3D.")

        if bg_lms.shape == (3,):
            bg_lms_eval = np.broadcast_to(bg_lms, W3D_eval.shape)
        elif bg_lms.shape == W3D_eval.shape:
            bg_lms_eval = bg_lms
        else:
            raise ValueError("bg_lms must have shape (3,) or match W3D.")

        # Map model-space coordinates from [-1, 1] to normalized RGB [0, 1],
        # then convert RGB to LMS and finally express the result as cone contrast.
        rgb = self.W_unit_to_N_unit(W3D_eval)
        lms = (self.M_RGBToLMS @ rgb.T).T + ambient_lms_eval
        cc = (lms - bg_lms_eval) / bg_lms_eval

        return cc[0] if was_1d else cc
    
    def cc_to_W3D(self, cc, ambient_lms, bg_lms):
        """
        Convert cone-contrast coordinates to 3D model-space coordinates.

        Parameters
        ----------
        cc : array-like, shape (3,) or (N, 3)
            Input point(s) in cone-contrast space.
        ambient_lms : array-like, shape (3,) or (N, 3)
            Ambient LMS offset. A single `(3,)` vector is broadcast to all
            input points; an `(N, 3)` array specifies a different ambient LMS
            for each input point.
        bg_lms : array-like, shape (3,) or (N, 3)
            Background LMS used to define cone contrast. A single `(3,)`
            vector is broadcast to all input points; an `(N, 3)` array
            specifies a different background LMS for each input point.

        Returns
        -------
        W3D : np.ndarray
            Model-space coordinates with the same shape as `cc`:
            `(3,)` for a single point or `(N, 3)` for a batch.
        """
        cc = np.asarray(cc, dtype=float)
        was_1d = cc.ndim == 1

        if was_1d:
            if cc.shape[0] != 3:
                raise ValueError("cc must have shape (3,) or (N, 3).")
            cc_eval = cc[None, :]
        elif cc.ndim == 2 and cc.shape[1] == 3:
            cc_eval = cc
        else:
            raise ValueError("cc must have shape (3,) or (N, 3).")

        ambient_lms = np.asarray(ambient_lms, dtype=float)
        bg_lms = np.asarray(bg_lms, dtype=float)

        if ambient_lms.shape == (3,):
            ambient_lms_eval = np.broadcast_to(ambient_lms, cc_eval.shape)
        elif ambient_lms.shape == cc_eval.shape:
            ambient_lms_eval = ambient_lms
        else:
            raise ValueError("ambient_lms must have shape (3,) or match cc.")

        if bg_lms.shape == (3,):
            bg_lms_eval = np.broadcast_to(bg_lms, cc_eval.shape)
        elif bg_lms.shape == cc_eval.shape:
            bg_lms_eval = bg_lms
        else:
            raise ValueError("bg_lms must have shape (3,) or match cc.")

        # Convert cone contrast back to LMS, remove the ambient offset after
        # mapping through RGB space, and finally map normalized RGB [0, 1]
        # back to model-space coordinates in [-1, 1].
        lms = cc_eval * bg_lms_eval + bg_lms_eval
        rgb = (self.M_LMSToRGB @ (lms - ambient_lms_eval).T).T
        W3D = self.N_unit_to_W_unit(rgb)

        return W3D[0] if was_1d else W3D
    
    #%% other useful methods                
    @staticmethod
    def N_unit_to_W_unit(N_unit):
        """
        Convert normalized RGB values from a 0-1 range to a -1 to 1 range.

        Parameters:
        N_unit : array_like
            value in the normalized 0-1 range.

        Returns:
        W_unit : array_like
            value in the normalized -1 to 1 range.
        """
        
        return N_unit * 2 - 1
        
    @staticmethod
    def W_unit_to_N_unit(W_unit):
        """
        Convert normalized RGB values from a -1 to 1 range to a 0-1 range.
        If transformation_matrix is not None, then the shape of W_unit should be 3 x N

        Parameters:
        W_unit : array_like
            value in the normalized -1 to 1 range.

        Returns:
        N_unit : array_like
            value in the normalized 0-1 range.
        """
        
        return (W_unit + 1) / 2
    
