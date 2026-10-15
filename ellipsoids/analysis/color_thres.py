#!/usr/bin/env python3
"""Color-threshold data loading and color-space transformations.

@author: fangfang
"""

import os
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache

import dill
import numpy as np
import pandas as pd


@lru_cache(maxsize=None)
def _index_files(base_path):
    """Walk `base_path` once and map every file name to all of its paths."""
    index = {}
    for root, _, files in os.walk(base_path):
        for file_name in files:
            index.setdefault(file_name, []).append(os.path.join(root, file_name))
    return index


#%%
@dataclass(frozen=True)
class ColorTransFilenames:
    """File names of the calibration matrices for one calibration date.

    Parameters
    ----------
    date : str
        Calibration date as MMDDYYYY. See
        `color_thresholds.load_transformation_matrix` for supported values.
    prefix, suffix : str
        Parts of the shared calibration tag, `{prefix}_{date}_{suffix}`.
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

    @property
    def M_DKLToConeInc_filename(self) -> str:
        return f"M_DKLToConeInc_{self.cal_tag}.csv"

    @property
    def M_ConeIncToDKL_filename(self) -> str:
        return f"M_ConeIncToDKL_{self.cal_tag}.csv"
    
@dataclass(frozen=True)
class ModelFitFilenames:
    base_2d_filename: str = 'Fitted_isothreshold_{plane}_sim240perCond_samplingNearContour_jitter0.1_seed0_bandwidth0.005_oddity.pkl'
    isoluminant_filename: str = 'Fitted_isothreshold_Isoluminant plane{cie}_sim18000total_samplingNearContour_jitter0.3_seed0_bandwidth0.005_decay0.4_oddity.pkl'
    ellipsoid_3d_filename: str = 'Fitted_isothreshold_ellipsoids_sim240perCond_samplingNearContour_jitter0.3_seed0{cie}_bandwidth0.005_oddity.pkl'

    @staticmethod
    def get_filename(plane_2D: str, cie_version: str = '', is_3d: bool = False) -> str:
        cie_suffix = '' if cie_version in {'', 'CIE1976'} else f'_{cie_version}'
        if is_3d:
            return ModelFitFilenames().ellipsoid_3d_filename.format(cie=cie_suffix)
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


#%%
class color_thresholds():
    def __init__(self, color_dimension, data_base_dir, plane_2D="Isoluminant plane",
                 fixed_value=None):
        """
        Store the data location and, for 2D analyses, the color plane.

        Parameters
        ----------
        color_dimension : int
            2 or 3, the dimensionality of the color analysis.
        data_base_dir : str
            Base directory searched (recursively) for data and calibration files.
        plane_2D : str, optional
            For 2D analyses: 'GB plane', 'RB plane', 'RG plane',
            'Isoluminant plane', or 'LSisolating plane'. Sets
            `fixed_color_dim` and `varying_color_dim`.
        fixed_value : float, optional
            Value (N unit) of the fixed RGB channel, used in 2D file names.
        """
        if color_dimension not in (2, 3):
            raise ValueError("color_dimension must be either 2 or 3.")

        self.color_dimension = color_dimension
        self.fixed_value = fixed_value
        self.plane_2D = plane_2D
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
        Return the path of `file_name` anywhere under the base directory.

        The directory tree is indexed once per base directory and reused. If
        the file is missing from the index (or was deleted since), the tree is
        re-indexed once in case it changed. If several files share the name,
        the first one found is returned with a warning.
        """
        paths = _index_files(self.base_path).get(file_name)
        if not paths or not all(os.path.exists(path) for path in paths):
            _index_files.cache_clear()
            paths = _index_files(self.base_path).get(file_name)
        if not paths:
            raise FileNotFoundError(f"Data files directory not found for file {file_name}.")
        if len(paths) > 1:
            warnings.warn(
                f"Found {len(paths)} files named {file_name}; using {paths[0]}.",
                RuntimeWarning,
            )
        return paths[0]

    #%%
    def load_CIE_data(self, CIE_version='', num_grid_pts=5):
        """
        Load CIELab-simulated thresholds for the 2D or 3D case into `CIE_data`.

        Parameters
        ----------
        CIE_version : {'', 'CIE1976', 'CIE1994', 'CIE2000'}, optional
            Color-difference formula used to simulate the thresholds.
        num_grid_pts : int, optional
            Number of reference grid points per dimension.
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
            data = dill.load(f)
    
        stim_key = f'stim{self.color_dimension}D'
        results_key = f'results{self.color_dimension}D'
        
        print(f"Loaded CIELab simulated data for {self.color_dimension}D case.\n"
              f"Number of grid points = {num_grid_pts}")
        if self.color_dimension == 2 and self.plane_2D != 'Isoluminant plane':
            self.CIE_data[stim_key] = data[f'stim_grid{num_grid_pts}_fixedVal{self.fixed_value}']
            self.CIE_data[results_key] = data[f'results_grid{num_grid_pts}_fixedVal{self.fixed_value}']                
        else:
            self.CIE_data[stim_key] = data[f'stim_grid{num_grid_pts}']
            self.CIE_data[results_key] = data[f'results_grid{num_grid_pts}']
            
    def load_model_fits(self, CIE_version='CIE1994'):
        """
        Load the Wishart model fits for the current plane into `Wishart_data`.

        Parameters
        ----------
        CIE_version : {'', 'CIE1976', 'CIE1994', 'CIE2000'}, optional
            Color-difference formula of the simulated data that was fitted.
        """
        if CIE_version not in {'', 'CIE1994', 'CIE1976', 'CIE2000'}:
            raise ValueError('Only CIE versions 1976, 1994, and 2000 are supported.')

        file_name = ModelFitFilenames.get_filename(
            plane_2D=self.plane_2D if self.color_dimension == 2 else None,
            cie_version=CIE_version,
            is_3d=self.color_dimension == 3,
        )

        file_path = self._find_exact_path(file_name)
        self.file_path_model_fits = file_path
        with open(file_path, 'rb') as f:
            self.Wishart_data = dill.load(f)

        print("Loaded Wishart model fits.")

    def load_transformation_matrix(self, file_date="02242025"):
        """
        Load color transformation matrices for a specified calibration date.

        Parameters
        ----------
        file_date : str, optional
            Calibration date (MMDDYYYY) selecting which set of matrices is
            loaded. Currently supported values:

            - "02242025"
                4D color discrimination experiment on the isoluminant plane
                (Hong et al., 2025, eLife).
            - "07292025"
                Pilot 6D color discrimination experiment spanning the full
                3D RGB cube.
            - "10062025"
                Adaptation experiment (gray vs. blue) on the isoluminant plane.
            - "11172025"
                Same underlying calibration as "10062025", but with matrices
                for the L–S–isolating plane.
            - "02012026"
                Formal 3D experiment.
            - "08272026"
                Formal 2D interleaved suprathreshold experiment.

        Notes
        -----
        Loads four forward/inverse pairs and their file paths:
        M_2DWToRGB / M_RGBTo2DW, M_2DWToDKL / M_DKLTo2DW,
        M_LMSToRGB / M_RGBToLMS, and M_ConeIncToDKL / M_DKLToConeInc.
        A pair that cannot be loaded is set to None with a warning; not every
        calibration date provides every pair.
        """
        filenames = ColorTransFilenames(file_date)
        print(f'Loading transformation matrices for the {file_date} calibration.')

        (self.file_path_M_2DWToRGB, self.file_path_M_RGBTo2DW,
         self.M_2DWToRGB, self.M_RGBTo2DW) = self._load_transformation_pair(
            filenames.M_2DWToRGB_filename, filenames.M_RGBTo2DW_filename,
            "RGB and 2DW"
        )

        (self.file_path_M_2DWToDKL, self.file_path_M_DKLTo2DW,
         self.M_2DWToDKL, self.M_DKLTo2DW) = self._load_transformation_pair(
            filenames.M_2DWToDKL_filename, filenames.M_DKLTo2DW_filename,
            "2DW and DKL"
        )

        (self.file_path_M_LMSToRGB, self.file_path_M_RGBToLMS,
         self.M_LMSToRGB, self.M_RGBToLMS) = self._load_transformation_pair(
            filenames.M_LMSToRGB_filename, filenames.M_RGBToLMS_filename,
            "RGB and LMS"
        )

        (self.file_path_M_ConeIncToDKL, self.file_path_M_DKLToConeInc,
         self.M_ConeIncToDKL, self.M_DKLToConeInc) = self._load_transformation_pair(
            filenames.M_ConeIncToDKL_filename, filenames.M_DKLToConeInc_filename,
            "cone increments and DKL", expected_shape=(3, 3)
        )

    def _load_transformation_pair(self, forward_filename, inverse_filename,
                                  description, expected_shape=None):
        """Return both paths and matrices, or four Nones if either load fails.

        Load into local variables so callers replace the entire pair together.
        Optional shape/finite validation preserves the checks for DKL cone-
        increment matrices without changing legacy matrix formats.
        """
        try:
            forward_path = self._find_exact_path(forward_filename)
            inverse_path = self._find_exact_path(inverse_filename)
            forward_matrix = np.array(pd.read_csv(forward_path, header=None))
            inverse_matrix = np.array(pd.read_csv(inverse_path, header=None))
            if expected_shape is not None:
                for path, matrix in ((forward_path, forward_matrix),
                                     (inverse_path, inverse_matrix)):
                    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
                        raise ValueError(f"{path} must be a finite {expected_shape} matrix.")
        except Exception as error:
            warnings.warn(
                f"Could not load transformation matrices between {description}: {error}",
                RuntimeWarning,
            )
            return None, None, None, None

        return forward_path, inverse_path, forward_matrix, inverse_matrix

    def get_data(self, key, dataset='CIE_data'):
        """
        Return one entry of `CIE_data` or `Wishart_data`.

        Parameters
        ----------
        key : str
            Key of the entry to retrieve.
        dataset : {'CIE_data', 'Wishart_data'}, optional
            Which stored dictionary to query.

        Raises
        ------
        ValueError
            If `dataset` is not one of the two dictionaries.
        KeyError
            If `key` is not in the selected dictionary; the message lists the
            available keys.
        """
        if dataset not in ('CIE_data', 'Wishart_data'):
            raise ValueError(f"Dataset '{dataset}' does not exist.")
        data = getattr(self, dataset)
        if key not in data:
            raise KeyError(
                f"Key '{key}' does not exist in {dataset}. "
                f"Available keys: {list(data.keys())}"
            )
        return data[key]

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
    
    #%% 3D point transformations
    @staticmethod
    def _as_points(points, name):
        """Return `points` as an (N, 3) float array and whether it was 1D."""
        points = np.asarray(points, dtype=float)
        if points.shape == (3,):
            return points[None, :], True
        if points.ndim == 2 and points.shape[1] == 3:
            return points, False
        raise ValueError(f"{name} must have shape (3,) or (N, 3).")

    @staticmethod
    def _match_points(values, points, name):
        """Broadcast a (3,) vector to `points`, or check an (N, 3) array matches.

        `values=None` is treated as zeros (e.g. no ambient light).
        """
        if values is None:
            return np.zeros_like(points)
        values = np.asarray(values, dtype=float)
        if values.shape != (3,) and values.shape != points.shape:
            raise ValueError(f"{name} must have shape (3,) or match the input points.")
        return np.broadcast_to(values, points.shape)

    def W3D_to_cc(self, W3D, bg_lms, ambient_lms=None):
        """
        Convert 3D model-space coordinates to cone-contrast coordinates.

        Parameters
        ----------
        W3D : array-like, shape (3,) or (N, 3)
            Input point(s) in 3D W-space.
        bg_lms : array-like, shape (3,) or (N, 3)
            Background LMS used to compute cone contrast. A single `(3,)`
            vector is broadcast to all input points; an `(N, 3)` array
            specifies a different background LMS for each input point.
        ambient_lms : array-like, shape (3,) or (N, 3), optional
            Ambient LMS offset, broadcast like `bg_lms`. Defaults to zero.

        Returns
        -------
        cc : np.ndarray
            Cone-contrast coordinates with the same shape as `W3D`:
            `(3,)` for a single point or `(N, 3)` for a batch.
        """
        W3D, was_1d = self._as_points(W3D, "W3D")
        bg_lms = self._match_points(bg_lms, W3D, "bg_lms")

        # Map model-space coordinates from [-1, 1] to normalized RGB [0, 1],
        # then convert RGB to LMS and finally express the result as cone contrast.
        lms = self.W3D_to_lms(W3D, ambient_lms)
        cc = (lms - bg_lms) / bg_lms

        return cc[0] if was_1d else cc

    def cc_to_W3D(self, cc, bg_lms, ambient_lms=None):
        """
        Convert cone-contrast coordinates to 3D model-space coordinates.

        Parameters
        ----------
        cc : array-like, shape (3,) or (N, 3)
            Input point(s) in cone-contrast space.
        bg_lms : array-like, shape (3,) or (N, 3)
            Background LMS used to define cone contrast. A single `(3,)`
            vector is broadcast to all input points; an `(N, 3)` array
            specifies a different background LMS for each input point.
        ambient_lms : array-like, shape (3,) or (N, 3), optional
            Ambient LMS offset, broadcast like `bg_lms`. Defaults to zero.

        Returns
        -------
        W3D : np.ndarray
            Model-space coordinates with the same shape as `cc`:
            `(3,)` for a single point or `(N, 3)` for a batch.
        """
        cc, was_1d = self._as_points(cc, "cc")
        bg_lms = self._match_points(bg_lms, cc, "bg_lms")

        # Convert cone contrast back to LMS, then remove the ambient offset and
        # map back through RGB to model-space coordinates in [-1, 1].
        W3D = self.lms_to_W3D(cc * bg_lms + bg_lms, ambient_lms)

        return W3D[0] if was_1d else W3D
    
    def recenter_via_cc(self, w_before, bgLMS_before, bgLMS_after):
        """Recenter 2D W points by preserving cone contrast across backgrounds.

        The transform is:
            2D W -> RGB -> LMS -> cone contrast relative to bgLMS_before
            -> LMS relative to bgLMS_after -> RGB -> 2D W.
        """
        w_before = np.asarray(w_before, dtype=float)
        was_1d = w_before.shape == (2,)

        rgb_before = self.W2D_to_rgb(w_before)
        lms_before = (self.M_RGBToLMS @ rgb_before.T).T

        bgLMS_before = np.asarray(bgLMS_before, dtype=float)
        if bgLMS_before.shape == (3,):
            bgLMS_before = np.broadcast_to(bgLMS_before, lms_before.shape)
        elif bgLMS_before.shape != lms_before.shape:
            raise ValueError(
                f"bgLMS_before must have shape (3,) or {lms_before.shape}; "
                f"got {bgLMS_before.shape}."
            )

        bgLMS_after = np.asarray(bgLMS_after, dtype=float)
        if bgLMS_after.shape == (3,):
            bgLMS_after = np.broadcast_to(bgLMS_after, lms_before.shape)
        elif bgLMS_after.shape != lms_before.shape:
            raise ValueError(
                f"bgLMS_after must have shape (3,) or {lms_before.shape}; "
                f"got {bgLMS_after.shape}."
            )

        cc = (lms_before - bgLMS_before) / bgLMS_before
        lms_after = cc * bgLMS_after + bgLMS_after
        rgb_after = (self.M_LMSToRGB @ lms_after.T).T

        w_after = self.rgb_to_2DW(rgb_after)
        return w_after[0] if was_1d else w_after

    def W3D_to_dkl(self, s, bg):
        """Convert 3D W-space stimuli to DKL relative to a W-space background.

        Parameters
        ----------
        s : array-like, shape (3,) or (N, 3)
            Stimulus coordinates, with rows indexing points in a batch.
        bg : array-like, shape (3,) or matching s
            Background in the same W-space units as s. A single vector is
            broadcast across all stimuli.

        Returns
        -------
        dkl : ndarray
            DKL coordinates with the same shape as s. Axis order and scaling
            are defined by the loaded M_ConeIncToDKL calibration matrix.

        Notes
        -----
        Uses RGB = (W + 1) / 2, hence delta_RGB = (s - bg) / 2.
        M_RGBToLMS maps these increments to cone increments, which are then
        mapped by M_ConeIncToDKL (without dividing by background LMS).
        The DKL matrix must be calibrated for the intended adapting background;
        changing bg only changes the subtracted origin, not that calibration.
        Values are not clipped. A shared ambient LMS offset cancels out.
        """
        s, was_1d = self._as_points(s, "s")
        bg = self._match_points(bg, s, "bg")
        if not np.isfinite(s).all() or not np.isfinite(bg).all():
            raise ValueError("s and bg must contain only finite values.")

        diff_rgb = self.W_unit_to_N_unit(s) - self.W_unit_to_N_unit(bg)
        dkl = (self.M_ConeIncToDKL @ self.M_RGBToLMS @ diff_rgb.T).T
        return dkl[0] if was_1d else dkl

    # Former name, kept so older scripts and notebooks keep working.
    W3D_to_DKL = W3D_to_dkl

    def dkl_to_W3D(self, dkl, bg):
        """Convert DKL coordinates back to 3D W-space; inverse of `W3D_to_dkl`.

        Parameters
        ----------
        dkl : array-like, shape (3,) or (N, 3)
            DKL coordinates, with rows indexing points in a batch.
        bg : array-like, shape (3,) or matching dkl
            Background in W-space that defines the DKL origin. A single vector
            is broadcast across all points.

        Returns
        -------
        W3D : ndarray
            W-space coordinates with the same shape as dkl.

        Notes
        -----
        Uses the loaded inverse matrices M_DKLToConeInc and M_LMSToRGB.
        Because W = 2 * RGB - 1, an RGB increment maps to twice that W
        increment. The same background caveat as `W3D_to_dkl` applies.
        """
        dkl, was_1d = self._as_points(dkl, "dkl")
        bg = self._match_points(bg, dkl, "bg")
        if not np.isfinite(dkl).all() or not np.isfinite(bg).all():
            raise ValueError("dkl and bg must contain only finite values.")

        diff_rgb = (self.M_LMSToRGB @ self.M_DKLToConeInc @ dkl.T).T
        W3D = bg + 2 * diff_rgb
        return W3D[0] if was_1d else W3D

    def W3D_to_lms(self, W3D, ambient_lms=None):
        """Convert 3D W-space coordinates to absolute LMS cone excitations.

        Parameters
        ----------
        W3D : array-like, shape (3,) or (N, 3)
            Input point(s) in 3D W-space.
        ambient_lms : array-like, shape (3,) or (N, 3), optional
            Ambient LMS offset added after the monitor's contribution.
            Defaults to zero.

        Returns
        -------
        lms : np.ndarray
            LMS values with the same shape as `W3D`.
        """
        W3D, was_1d = self._as_points(W3D, "W3D")
        ambient_lms = self._match_points(ambient_lms, W3D, "ambient_lms")

        rgb = self.W_unit_to_N_unit(W3D)
        lms = (self.M_RGBToLMS @ rgb.T).T + ambient_lms
        return lms[0] if was_1d else lms

    def lms_to_W3D(self, lms, ambient_lms=None):
        """Convert absolute LMS cone excitations back to 3D W-space.

        Parameters
        ----------
        lms : array-like, shape (3,) or (N, 3)
            Input point(s) in LMS space.
        ambient_lms : array-like, shape (3,) or (N, 3), optional
            Ambient LMS offset removed before mapping LMS back to RGB.
            Defaults to zero.

        Returns
        -------
        W3D : np.ndarray
            W-space coordinates with the same shape as `lms`.
        """
        lms, was_1d = self._as_points(lms, "lms")
        ambient_lms = self._match_points(ambient_lms, lms, "ambient_lms")

        rgb = (self.M_LMSToRGB @ (lms - ambient_lms).T).T
        W3D = self.N_unit_to_W_unit(rgb)
        return W3D[0] if was_1d else W3D

    #%% covariance transformations
    # Every transformation below is affine in the point coordinates, so a
    # covariance Sigma maps to J @ Sigma @ J.T, where J is the Jacobian of the
    # matching point transformation. Offsets (background, ambient light) do
    # not affect covariance. All methods accept a single (3, 3) matrix or a
    # batch of shape (N, 3, 3).
    @staticmethod
    def _check_covM(covM):
        covM = np.asarray(covM, dtype=float)
        if covM.shape[-2:] != (3, 3) or covM.ndim not in (2, 3):
            raise ValueError("covM must have shape (3, 3) or (N, 3, 3).")
        return covM

    @staticmethod
    def _propagate_covM(jacobian, covM):
        return jacobian @ covM @ np.swapaxes(jacobian, -1, -2)

    def W3D_to_rgb_covM(self, covM):
        """Convert W-space covariance(s) to normalized RGB (Jacobian 0.5 * I)."""
        return self._check_covM(covM) / 4

    def rgb_to_W3D_covM(self, covM):
        """Convert normalized-RGB covariance(s) to W-space (Jacobian 2 * I)."""
        return self._check_covM(covM) * 4

    def W3D_to_lms_covM(self, covM):
        """Convert W-space covariance(s) to LMS (Jacobian 0.5 * M_RGBToLMS)."""
        return self._propagate_covM(0.5 * self.M_RGBToLMS, self._check_covM(covM))

    def lms_to_W3D_covM(self, covM):
        """Convert LMS covariance(s) to W-space (Jacobian 2 * M_LMSToRGB)."""
        return self._propagate_covM(2 * self.M_LMSToRGB, self._check_covM(covM))

    def _cc_scaling(self, covM, bg_lms):
        """Validate bg_lms against covM and return it with a trailing axis."""
        bg_lms = np.asarray(bg_lms, dtype=float)
        if bg_lms.shape != (3,) and bg_lms.shape != covM.shape[:-1]:
            raise ValueError("bg_lms must have shape (3,) or (N, 3) matching covM.")
        return bg_lms[..., :, None]

    def W3D_to_cc_covM(self, covM, bg_lms):
        """Convert W-space covariance(s) to cone contrast.

        Parameters
        ----------
        covM : array-like, shape (3, 3) or (N, 3, 3)
            Covariance matrices in W-space.
        bg_lms : array-like, shape (3,) or (N, 3)
            Background LMS (including any ambient light), as in `W3D_to_cc`.

        Notes
        -----
        The Jacobian is diag(1 / bg_lms) @ M_RGBToLMS * 0.5.
        """
        covM = self._check_covM(covM)
        jacobian = 0.5 * self.M_RGBToLMS / self._cc_scaling(covM, bg_lms)
        return self._propagate_covM(jacobian, covM)

    def cc_to_W3D_covM(self, covM, bg_lms):
        """Convert cone-contrast covariance(s) back to W-space.

        Parameters
        ----------
        covM : array-like, shape (3, 3) or (N, 3, 3)
            Covariance matrices in cone-contrast space.
        bg_lms : array-like, shape (3,) or (N, 3)
            Background LMS (including any ambient light), as in `cc_to_W3D`.

        Notes
        -----
        The Jacobian is 2 * M_LMSToRGB @ diag(bg_lms).
        """
        covM = self._check_covM(covM)
        bg_lms = self._cc_scaling(covM, bg_lms)
        jacobian = 2 * self.M_LMSToRGB * np.swapaxes(bg_lms, -1, -2)
        return self._propagate_covM(jacobian, covM)

    def W3D_to_dkl_covM(self, covM):
        """Convert W-space covariance(s) to DKL.

        The Jacobian is 0.5 * M_ConeIncToDKL @ M_RGBToLMS. The same
        background caveat as `W3D_to_dkl` applies.
        """
        jacobian = 0.5 * self.M_ConeIncToDKL @ self.M_RGBToLMS
        return self._propagate_covM(jacobian, self._check_covM(covM))

    def dkl_to_W3D_covM(self, covM):
        """Convert DKL covariance(s) to W-space.

        The Jacobian is 2 * M_LMSToRGB @ M_DKLToConeInc.
        """
        jacobian = 2 * self.M_LMSToRGB @ self.M_DKLToConeInc
        return self._propagate_covM(jacobian, self._check_covM(covM))

    #%% other useful methods
    @staticmethod
    def N_unit_to_W_unit(N_unit):
        """
        Convert normalized RGB values from a 0-1 range to a -1 to 1 range.

        Parameters
        ----------
        N_unit : array_like
            Values in the normalized 0-1 range, any shape.

        Returns
        -------
        W_unit : array_like
            Values in the -1 to 1 range, same shape as the input.
        """
        return N_unit * 2 - 1

    @staticmethod
    def W_unit_to_N_unit(W_unit):
        """
        Convert normalized RGB values from a -1 to 1 range to a 0-1 range.

        Parameters
        ----------
        W_unit : array_like
            Values in the -1 to 1 range, any shape.

        Returns
        -------
        N_unit : array_like
            Values in the normalized 0-1 range, same shape as the input.
        """
        return (W_unit + 1) / 2
