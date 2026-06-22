#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:14:11 2024

@author: fangfang
"""

from __future__ import annotations
from scipy.io import loadmat
import colour
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import sys
import os
import warnings
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Sequence, Dict, List
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000, delta_e_cie1994, delta_e_cie1976
from analysis.ellipses_tools import UnitCircleGenerate, ellParamsQ_to_covMat, PointsOnEllipseQ
#in order for delta_e_cie2000 to work, we need to do the following adjustment
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

required_file_dir = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox/"
    
#%%
class SimThresCIELab:
    def __init__(self, background, plane_2D_list = ['GB plane', 'RB plane', 'RG plane'], 
                 file_date = "02242025", flag_force_Lstar50 = False,
                 background_space = "rgb"):
        """
        Parameters:

        """
        self.file_data = file_date
        self.background_space = background_space.lower()
        if self.background_space not in {"rgb", "xyy"}:
            raise ValueError(
                f"background_space must currently be 'rgb' or 'xyY'; got {background_space!r}."
            )
        # Copy the caller-provided background so any later in-place updates to
        # the class-owned white point do not mutate external variables.
        self.background = np.asarray(background, dtype=float).copy()
        self.background_rgb = self.background if self.background_space == "rgb" else None
        self.background_xyY = self.background if self.background_space == "xyy" else None
        
        #load T_cones, B_monitor, M_LMSToXYZ
        self._load_required_files(file_date)
                
        # compute transformation matrices from XYZ -> RGB and vice versa
        self._compute_transformation_matrices()
        
        self.flag_force_Lstar50 = flag_force_Lstar50
        self.compute_bg_xyY()
        
        #number of selected planes
        # Validate plane_2D_list
        self._validate_plane_list(plane_2D_list)
        self.plane_2D_list  = plane_2D_list
        self.nPlanes        = len(self.plane_2D_list)
        
        #Note that if plane_2D_list is 'Isoluminant plane', some of the following methods are not applicable
        #come back to this class in the future and refine the code to be more generalizable
        self.plane_2D_dict  = dict(zip(self.plane_2D_list, list(range(self.nPlanes))))
        if self.nPlanes == 3:
            self.varying_dims = [[1,2],[0,2],[0,1]]
        elif self.nPlanes == 1:
            self.varying_dims = [[0,1]] #treat it as RG plane with the third dimension fixed at 1
    
    def _load_required_files(self, file_date):
        """
        Load the calibration tables needed for RGB/SPD/XYZ conversions.

        The loaded arrays define the display pipeline at the spectral level:
            RGB --(B_MONITOR)--> spectral power distribution (SPD)
            SPD --(T_xyz)--> CIE 1931 XYZ tristimulus values

        Parameters
        ----------
        file_date : str
            Calibration tag used to select the measured monitor-primary file.
        """
        sys.path.append(required_file_dir)
        os.chdir(required_file_dir)

        # Measured monitor spectral primaries for the selected calibration date.
        # Shape: (201, 3) = (wavelength samples, R/G/B primaries).
        self.B_MONITOR = loadmat(f'B_monitor_dell_{file_date}.mat')['B_monitor']

        # CIE 1931 XYZ color matching functions on the same wavelength grid.
        # Shape: (3, 201) = (Xbar/Ybar/Zbar, wavelength samples).
        self.T_xyz = loadmat('T_xyz_CIE1931.mat')['T_xyz']
        
    def _compute_transformation_matrices(self):
        """ Compute transformation matrice between RGB and XYZ """
        # Collapse the spectral pipeline RGB -> SPD -> XYZ into one 3x3 matrix.
        # T_xyz:     (3, 201)
        # B_MONITOR: (201, 3)
        # Result:    (3, 3), mapping linear RGB directly to XYZ.
        self.M_RGBToXYZ = self.T_xyz @ self.B_MONITOR

        # Inverse transform used when mapping XYZ coordinates back to display RGB.
        self.M_XYZToRGB = np.linalg.inv(self.M_RGBToXYZ)
            
    def _validate_plane_list(self, plane_2D_list):
        """Internal method to validate plane_2D_list."""
        valid_plane_options = [['GB plane', 'RB plane', 'RG plane'], ['Isoluminant plane']]
        if plane_2D_list not in valid_plane_options:
            raise ValueError(f"Invalid plane_2D_list: {plane_2D_list}. "+\
                             "Must be one of {valid_plane_options}.")
        
        if self.background_space == "rgb":
            if self.background_rgb.shape != (3,):
                raise ValueError(
                    f"background_rgb must have shape (3,); got {self.background_rgb.shape}."
                )
        else:
            if self.background_xyY.shape != (3,):
                raise ValueError(
                    f"background_xyY must have shape (3,); got {self.background_xyY.shape}."
                )
            
    def compute_bg_xyY(self):
        """
        Set the background white point used for Lab conversions.

        Two input modes are supported:
            1. `background_space == "rgb"`:
               Convert the provided display RGB background to XYZ and then xyY.
            2. `background_space == "xyy"`:
               Use the provided xyY background directly.

        If `flag_force_Lstar50` is enabled, the background luminance is
        rescaled so that the corresponding background color would map to
        L* = 50 under the resulting Lab reference white.
        """
        if self.background_space == "rgb":
            # Convert the display RGB background into CIE XYZ, then express it
            # as xyY so it can serve as the Lab reference white.
            background_XYZ = self.M_RGBToXYZ @ self.background_rgb
            self.background_xyY = np.asarray(colour.XYZ_to_xyY(background_XYZ))
        else:
            # The caller already provided a measured xyY white point.
            background_XYZ = colour.xyY_to_XYZ(self.background_xyY)

        self.background_xy = self.background_xyY[:2].copy()

        if not self.flag_force_Lstar50:
            # Default behavior: use the background as-is, with no additional
            # luminance renormalization.
            self.background_Yn = 1.0
        else:
            # Solve for the white-point luminance that would make this
            # background land at L* = 50.
            Y_target = background_XYZ[1]
            f_target = (50 + 16) / 116
            self.background_Yn = Y_target / (f_target ** 3)
            self.background_xyY[2] = self.background_Yn
            
    def get_plane_1slice(self, grid_lb, grid_ub, num_grid_pts,fixed_val, plane_2D):
        """
        Generates a 2D slice of a 3D plane with one fixed dimension.
        
        Parameters:
        grid_lb (float): Lower bound of the grid.
        grid_ub (float): Upper bound of the grid.
        num_grid_pts (int): Number of grid points in each dimension.
        fixed_val (float): Fixed value for the plane's constant dimension.
        plane_2D (str): Identifier for the 2D plane being generated (e.g., 'XY', 'XZ').
        
        Returns:
        tuple: A tuple containing:
            - plane_1slice (numpy.ndarray, shape: (num_grid_pts x num_grid_pts x 3)): 
                3D array representing the slice with varying values in two dimensions and 
                a fixed value in the third.
            - grid_1d (numpy.ndarray, shape: (num_grid_pts,)): 1D array of grid points 
                used for the X and Y dimensions.
            - X (numpy.ndarray, shape: (num_grid_pts x num_grid_pts)): 2D array of X coordinates.
            - Y (numpy.ndarray, shape: (num_grid_pts x num_grid_pts)): 2D array of Y coordinates.
        """
        
        # Generate a 1D grid and mesh grids for the X and Y dimensions
        grid_1d = np.linspace(grid_lb, grid_ub, num_grid_pts)
        X, Y = np.meshgrid(grid_1d, grid_1d)
        
        # Initialize a 3D array to store the slice with NaN values
        plane_1slice = np.full((self.nPlanes, num_grid_pts, num_grid_pts), np.nan)
        
        # Identify which dimension remains constant and which two vary
        plane_2D_idx = self.plane_2D_dict[plane_2D]
        varying_dim = self.varying_dims[plane_2D_idx]
        
        # Assign the meshgrid values to the varying dimensions in the 3D slice
        plane_1slice[varying_dim[0]] = X
        plane_1slice[varying_dim[1]] = Y
        
        # Assign the fixed value to the fixed dimension
        plane_1slice[plane_2D_idx] = np.ones((num_grid_pts, num_grid_pts))*fixed_val
            
        return np.moveaxis(plane_1slice, 0, -1), grid_1d, X, Y
    
    def get_planes(self, grid_lb, grid_ub, num_grid_pts = 5, fixed_val = 0.5):
        """
        Generates multiple 2D slices of 3D planes for different 2D planes within a 3D space.
        
        Parameters:
        grid_lb (float): Lower bound of the grid.
        grid_ub (float): Upper bound of the grid.
        num_grid_pts (int, optional): Number of grid points in each dimension. Default is 5.
        fixed_val (float, optional): Fixed value for the plane's constant dimension. Default is 0.5.
        
        Returns:
        tuple: A tuple containing:
            - plane_3slices (numpy.ndarray, shape: (3 x 3 x num_grid_pts x num_grid_pts)): 
                    4D array where each slice corresponds to a 2D plane within the 3D space.
            - grid_1d (numpy.ndarray, shape: (num_grid_pts,)): 1D array of grid points used 
                for the X and Y dimensions.
            - X (numpy.ndarray, shape: num_grid_pts x num_grid_pts): 2D array of X coordinates 
                (from the last processed plane).
            - Y (numpy.ndarray, shape: num_grid_pts x num_grid_pts): 2D array of Y coordinates 
                (from the last processed plane).
        """
        
        # Initialize a 4D array to store slices for each 2D plane
        plane_3slices = np.full((self.nPlanes, num_grid_pts, num_grid_pts, self.nPlanes), np.nan)
        # Iterate over each 2D plane identifier and generate its corresponding slice
        for i, plane_str in enumerate(self.plane_2D_list):
            plane_3slices[i], grid_1d, X, Y = self.get_plane_1slice(grid_lb,
                                                                    grid_ub, 
                                                                    num_grid_pts, 
                                                                    fixed_val, 
                                                                    plane_2D = plane_str
                                                                    )
        return plane_3slices, grid_1d, X, Y
    
    def convert_lab_rgb(self, color_Lab):
        """
        Convert a CIELab color value back to RGB, using the inverse of the
        display pipeline: Lab → XYZ → RGB.
        
        Parameters:
        - color_Lab (array; 3, or N,3): Lab color to convert (1D array)
        
        Returns:
        - color_RGB.T (array; 3, or N,3): RGB values (may need gamma correction or clipping)
        - color_XYZ   (array; 3, or N,3): CIEXYZ intermediate
        """

        color_Lab = np.asarray(color_Lab)
        if color_Lab.shape[-1] != 3:
            raise ValueError(
                f"color_Lab must have shape (..., 3) before colour.Lab_to_XYZ; "
                f"got {color_Lab.shape}."
            )
        color_XYZ = colour.Lab_to_XYZ(color_Lab, self.background_xyY)
        
        # Map XYZ back to linear display RGB using the inverse calibration matrix.
        color_RGB = self.M_XYZToRGB @ color_XYZ.T
    
        return color_RGB.T, color_XYZ 

    def convert_rgb_lab(self, color_RGB):
        """
        Convert an RGB color value into the CIELab color space.
    
        Parameters:
        - color_RGB (array; 3, or N,3): RGB color value(s) to be converted.
            where N is the number of selected wavelengths
        
        Returns:
        - color_Lab (array; 3, or N,3): The converted color(s) in CIELab color space, a 1D array.
        - color_XYZ (array; 3, or N,3): The intermediate CIEXYZ color space representation, a 1D array.
    
        """
        
        # Convert the input display RGB values directly to XYZ using the
        # monitor-specific RGB->XYZ calibration matrix.
        color_XYZ = self.M_RGBToXYZ @ color_RGB.T

        color_XYZ_input = color_XYZ.T
        if color_XYZ_input.shape[-1] != 3:
            raise ValueError(
                f"color_XYZ must have shape (..., 3) before colour.XYZ_to_Lab; "
                f"got {color_XYZ_input.shape}."
            )
        color_Lab = colour.XYZ_to_Lab(color_XYZ_input, self.background_xyY)
        
        return color_Lab, color_XYZ.T
    
    def compute_deltaE(self, ref_RGB, vecDir, vecLen, comp_RGB=None, method='CIE1994'):
        """
        Computes the perceptual difference (deltaE) between a reference stimulus
        and a comparison stimulus in the CIELab color space. The comparison stimulus
        can either be specified directly or calculated based on a chromatic direction
        and distance from the reference stimulus.
    
        Parameters:
        - ref_RGB (array; 3,): RGB values of the reference stimulus (source color).
        - vecDir (array; 1 x 3): Chromatic direction vector defining how the comparison 
          stimulus varies from the reference stimulus in RGB space.
        - vecLen (float): Magnitude of the variation along the chromatic direction `vecDir`.
        - comp_RGB (array; 3, optional): RGB values of the comparison stimulus. If not provided,
          it will be calculated by applying the `vecDir` and `vecLen` to the `ref_RGB`.
        - method (str): The method for calculating deltaE. Options are:
            - 'CIE1976': DeltaE using the CIE1976 method (Euclidean distance in CIELab).
            - 'CIE1994': DeltaE using the CIE1994 method (accounts for perceptual non-uniformity).
            - 'CIE2000': DeltaE using the CIE2000 method (more advanced perceptual uniformity).
    
        Returns:
        - deltaE (float): The computed perceptual difference between the reference and comparison stimuli.
    
        Notes:
        - If an invalid method is specified, the function will issue a warning and default to
          the Euclidean distance in CIELab ('Euclidean').
    
        """
    
        # Convert reference RGB to CIELab values.
        ref_Lab, *_ = self.convert_rgb_lab(ref_RGB)
    
        # If comparison RGB is not provided, calculate it by moving ref_RGB along vecDir by vecLen.
        if comp_RGB is None:
            comp_RGB = ref_RGB + vecDir * vecLen
    
        # Convert comparison RGB to CIELab values.
        comp_Lab, *_ = self.convert_rgb_lab(comp_RGB)
        
        # Define reference and comparison colors in LabColor format.
        color1 = LabColor(lab_l=ref_Lab[0], lab_a=ref_Lab[1], lab_b=ref_Lab[2])
        color2 = LabColor(lab_l=comp_Lab[0], lab_a=comp_Lab[1], lab_b=comp_Lab[2])
            
        # Compute deltaE using the specified method.
        if method == 'CIE2000':
            # CIE2000 method (more accurate perceptual uniformity).
            deltaE = delta_e_cie2000(color1, color2)
        elif method == 'CIE1994':
            # CIE1994 method (intermediate between CIE1976 and CIE2000).
            deltaE = delta_e_cie1994(color1, color2)
        else:
            # Simple Euclidean distance in CIELab.
            deltaE = np.linalg.norm(comp_Lab - ref_Lab)
            # THIS IS EQUIVALENT AS CIE1976
            
            # CIE1976 method (Euclidean distance in CIELab).
            #deltaE = delta_e_cie1976(color1, color2)
    
        return deltaE    
    
    def find_vecLen(self, ref_RGB_test, vecDir_test, deltaE, lb_opt = 0,
                    ub_opt = 0.1, N_opt = 3, coloralg = 'CIE1994'):
        """
        This function finds the optimal vector length for a chromatic direction
        that achieves a target perceptual difference in the CIELab color space.
    
        Parameters:
        - ref_RGB_test (array): The RGB values of the reference stimulus
        - vecDir_test (array): The chromatic direction vector for comparison stimulus variation
        - deltaE (float): The target deltaE value (e.g., 1 JND)
        - lb_gridsearch (float): the lower bounds for the search of the vector length
        - ub_gridsearch (float): the upper bounds for the search of the vector length
        - N_gridsearch (int): the number of runs for optimization to ensure we don't get stuck 
            at local minima
        
        Returns:
        - opt_vecLen (float): The optimal vector length that achieves the target deltaE value
        """
        
        if coloralg not in ['CIE1976', 'CIE1994', 'CIE2000']:
            raise ValueError("The method can only be 'CIE1976' or 'CIE1994' or 'CIE2000'.")
                
        #The lambda function computes the absolute difference between the
        #deltaE obtained from compute_deltaE function and the target deltaE.
        deltaE_func = lambda d: abs(self.compute_deltaE(ref_RGB_test,
                                                        vecDir_test, 
                                                        d, method=coloralg) - deltaE)
            
        # Generate initial points for the optimization algorithm within the bounds.
        init = np.random.rand(N_opt) * (ub_opt - lb_opt) + lb_opt
        # Set the options for the optimization algorithm.
        options = {'maxiter': 1e5, 'disp': False}
        # Initialize arrays to store the vector lengths and corresponding deltaE 
        #values for each run.
        vecLen_n = np.empty(N_opt)
        deltaE_n = np.empty(N_opt)
        
        # Loop over the number of runs to perform the optimizations.
        for n in range(N_opt):
            # Use scipy's minimize function to find the vector length that minimizes
            # the difference to the target deltaE. SLSQP method is used for 
            #constrained optimization.
            res = minimize(deltaE_func, init[n],method='SLSQP',
                           bounds=[(lb_opt, ub_opt)], options=options)
            # Store the result of each optimization run.
            vecLen_n[n] = res.x
            deltaE_n[n] = res.fun
            
        # Identify the index of the run that resulted in the minimum deltaE value.
        idx_min = np.argmin(deltaE_n)
        # Choose the optimal vector length from the run with the minimum deltaE value.
        opt_vecLen = vecLen_n[idx_min]
        
        return opt_vecLen
    
    def find_threshold_point_on_isoluminant_plane(self, W_ref, chrom_dir, M_RGBTo2DW, 
                                                  M_2DWToRGB, deltaE, coloralg = 'CIE1994'):
        """
        Compute the threshold point along a given chromatic direction on the isoluminant plane.
    
        This function simulates a just-noticeable color difference (deltaE) starting from 
        a reference color in Wishart space, along a specified chromatic direction, and finds 
        the corresponding comparison stimulus in both RGB and W spaces.
    
        Parameters
        ----------
        W_ref : np.array of shape (2,)
            Reference color in 2D Wishart space.
        chrom_dir : np.array of shape (2,)
            Normalized chromatic direction in Wishart space.
        M_RGBTo2DW : np.array of shape (3, 3)
            Transformation matrix from RGB to Wishart space.
        M_2DWToRGB : np.array of shape (3, 3)
            Transformation matrix from Wishart space to RGB.
        deltaE : float
            Desired perceptual color difference in CIELab space.
        coloralg : str, default='CIE2000'
            Algorithm used to compute perceptual color difference. Must be one of:
            'CIE1976', 'CIE1994', or 'CIE2000'.
    
        Returns
        -------
        rgb_vecDir : np.array of shape (3,)
            Chromatic direction vector in RGB space.
        opt_vecLen : float
            Length of the RGB vector that yields the desired deltaE.
        rgb_comp_threshold : np.array of shape (3,)
            Comparison color in RGB space at threshold.
        W_comp_threshold : np.array of shape (2,)
            Corresponding point in 2D Wishart space.
    
        """
        if coloralg not in ['CIE1976', 'CIE1994', 'CIE2000']:
            raise ValueError("The method can only be 'CIE1976' or 'CIE1994' or 'CIE2000'.")
            
        # Step 1: Define a chromatic direction in W-space and convert to RGB
        chrom_dir_W = chrom_dir + W_ref  # Shifted chromatic direction
        chrom_dir_rgb = M_2DWToRGB @ np.append(chrom_dir_W, 1)  # Convert back to RGB

        # Step 2: Compute normalized direction vector in RGB space
        rgb_ref = M_2DWToRGB @ np.append(W_ref, 1)
        rgb_vecDir_temp = chrom_dir_rgb - rgb_ref  # Vector difference
        rgb_vecDir = rgb_vecDir_temp / np.linalg.norm(rgb_vecDir_temp)  # Normalize

        # Step 3: Find vector length that produces ΔE = deltaE in CIELab space
        opt_vecLen = self.find_vecLen(
            rgb_ref, rgb_vecDir, deltaE = deltaE, coloralg = coloralg
        )

        # Step 4: Compute threshold point in RGB space
        rgb_comp_threshold = opt_vecLen * rgb_vecDir + rgb_ref
        
        # Step 5: Transform Threshold Points from RGB → W 
        W_comp_temp = M_RGBTo2DW @ rgb_comp_threshold  # Convert to W-space
        W_comp = W_comp_temp / W_comp_temp[-1]  # Normalize last row to 1
        W_comp_threshold = W_comp[:2]
        
        return rgb_ref, rgb_vecDir, opt_vecLen, rgb_comp_threshold, W_comp_threshold
        
    def find_boundary_LabPts_onSlice(self, fixedVal, fixedDim, 
                                     rgb_bds=[0, 1],
                                     num_dir_pts=360,
                                     bruteforce_scaler_bds=(1, 200), 
                                     vectors_centroid = np.array([0,0]),
                                     bruteforce_num_scalers=1000):
        """
        Find boundary points on a CIELab 2D slice (fixing one Lab dimension) that remain
        within the display gamut.
    
        This generalizes the “fixed L*” case to any Lab slice by fixing exactly one
        dimension (L*, a*, or b*) and sampling radial directions in the remaining
        2D subspace. Along each in-plane direction, we increase a scalar radius,
        convert the resulting Lab points to RGB, and detect the first out-of-gamut
        sample. We then return the Lab / RGB / radius for the *last in-gamut*
        sample just before that exit.
    
        Parameters
        ----------
        fixedVal : float
            The fixed value for the chosen Lab dimension (e.g., L* = 60, or a* = 10, …).
        fixedDim : int
            Which Lab dimension to fix:
                0 → fix L* (vary a*, b*)
                1 → fix a* (vary L*, b*)
                2 → fix b* (vary L*, a*)
        rgb_bds : list[float, float], optional
            Inclusive lower/upper bounds for valid RGB (default [0, 1]).
        num_dir_pts : int, optional
            Number of evenly spaced directions in the slice plane (default 360).
        bruteforce_scaler_bds : tuple[float, float], optional
            Start/end of the radial scalar grid along each direction (default (1, 200)).
        bruteforce_num_scalers : int, optional
            Number of scalar samples per direction (default 1000).
    
        Returns
        -------
        Lab_at_boundary : (num_dir_pts, 3) ndarray
            Lab coordinates at the last in-gamut sample for each direction on the slice.
        rgb_at_boundary : (num_dir_pts, 3) ndarray
            Corresponding RGB values at the last in-gamut sample.
        scaler_at_boundary : (num_dir_pts,) ndarray
            The scalar radius used to generate the last in-gamut sample.
    
        Notes
        -----
        - “Brute force” sampling: a fixed 1D scalar grid per direction.
        - Gamut exit is detected per RGB channel; the earliest channel exit defines
          the direction’s “first exit” index. We report the previous index (last in-gamut).
        - Assumes `self.convert_lab_rgb` accepts a batched (N,3) Lab array and returns
          the first output as an (N,3) RGB array in the same value domain as `rgb_bds`.
        """
    
        # --- 1) In-plane directions: always a 2D plane orthogonal to `fixedDim`.
        # For fixedDim=0 (fix L*), directions are in the (a*, b*) plane.
        # For fixedDim=1 (fix a*), directions are in the (L*, b*) plane.
        # For fixedDim=2 (fix b*), directions are in the (L*, a*) plane.
        grid_theta_xy = UnitCircleGenerate(num_dir_pts)  # (2, N), unit directions in the slice plane
    
        # --- 2) Radial scalar grid along each direction.
        bruteforce_scaler = np.linspace(*bruteforce_scaler_bds, bruteforce_num_scalers)  # (M,)
    
        # --- 3) Broadcast directions & scalars to form all candidate points in the slice plane.
        # grid_theta_xy_rep: (M, N, 2) — repeat each 2D direction across M scalars
        grid_theta_xy_rep = np.tile(grid_theta_xy.T[None], (bruteforce_num_scalers, 1,1)) #(M, N, 2)
        
        # bruteforce_scaler_rep: (M, N, 2) — match scalars across both in-plane coordinates
        bruteforce_scaler_rep = np.tile(bruteforce_scaler[:, None, None], (1, *grid_theta_xy_rep.shape[1:]))
    
        # valV_batch: (M, N, 2) — the two *varying* coordinates in the slice plane
        valV_batch = grid_theta_xy_rep * bruteforce_scaler_rep + vectors_centroid[None, None]
    
        # valF_batch: (M, N, 1) — the *fixed* coordinate (the chosen Lab dimension)
        valF_batch = np.full(valV_batch.shape[:2] + (1,), fixedVal)
    
        # --- 4) Assemble Lab triplets as (M, N, 3) in [L*, a*, b*] order.
        # Insert the fixed coordinate according to `fixedDim`, and the two varying ones
        # in their natural order for the slice plane.
        Lab_batch = self.fill_in_vals(fixedDim, valV_batch, valF_batch)
    
        # --- 5) Lab → RGB conversion in batch: reshape to (N*M, 3), then back to (M, N, 3).
        Lab_batch_reshape = np.reshape(Lab_batch, (-1, 3))
        rgb_batch_flat, *_ = self.convert_lab_rgb(Lab_batch_reshape)  # (N*M, 3) expected
        rgb_batch = np.reshape(rgb_batch_flat, (bruteforce_num_scalers, num_dir_pts, -1)) 
    
        # --- 6) Detect out-of-gamut per channel; find earliest exit index along the scalar axis.
        out_of_gamut = (rgb_batch < rgb_bds[0]) | (rgb_batch > rgb_bds[-1])   # (M, N, 3)
        first_idx_allcols = np.argmax(out_of_gamut, axis=0).astype(float)     # (N, 3)
        first_idx_allcols[~out_of_gamut.any(axis=0)] = np.nan                 # NaN if a channel never exits
    
        # --- 7) Collapse channels → first exit per direction, then step back to last in-gamut.
        first_idx = (np.nanmin(first_idx_allcols, axis=1) - 1).astype(int)     # (N,)
    
        # --- 8) Gather the boundary scalar, Lab, and RGB using per-row indices.
        scaler_at_boundary = bruteforce_scaler[first_idx]                      # (N,)
    
        # Build indices shaped (1, N, 1) to gather along axis=0 for both Lab and RGB
        first_idx_rep_Lab = first_idx[None, :, None]
        Lab_at_boundary = np.take_along_axis(Lab_batch, first_idx_rep_Lab, axis=0).squeeze(0)   # (N, 3)
        
        first_idx_rep_rgb = first_idx[None, :, None]
        rgb_at_boundary = np.take_along_axis(rgb_batch, first_idx_rep_rgb, axis=0).squeeze(0)   # (N, 3)
    
        return Lab_at_boundary, rgb_at_boundary, scaler_at_boundary
    
    @staticmethod
    def fill_in_vals(fixed_dim, existing, filling, C=-1):
        """
        Insert the fixed Lab channel at position `fixed_dim` along axis `C`,
        producing channels ordered as [L*, a*, b*] on that axis.
    
        Parameters
        ----------
        fixed_dim : {0,1,2}
            0=L*, 1=a*, 2=b* (insertion index along axis `C`)
        existing : ndarray
            Shape like (..., 2, ...) on axis `C` — the two varying channels
            in Lab order with the fixed one removed.
        filling : ndarray
            Same shape as `existing` except axis `C` has size 1 — the fixed channel.
        C : int, optional
            Channel axis (default -1). Works with channels-last or channels-middle.
    
        Returns
        -------
        out : ndarray
            Same shape as `existing` but with axis `C` size 3 (L*, a*, b*).
        """
        ex = np.asarray(existing)
        fi = np.asarray(filling)
    
        # normalize negative axis
        C = ex.ndim + C if C < 0 else C
    
        # basic checks
        if ex.shape[C] != 2:
            raise ValueError(f"`existing` channel axis C={C} must be size 2, got {ex.shape[C]}")
        if fi.shape[C] != 1:
            raise ValueError(f"`filling` channel axis C={C} must be size 1, got {fi.shape[C]}")
    
        # split without manual slices; left has channels [:fixed_dim], right has [fixed_dim:]
        left, right = np.split(ex, [int(fixed_dim)], axis=C)
    
        # concatenate left + fixed + right along channel axis
        return np.concatenate([left, fi, right], axis=C)

#%%
def fit_best_scaler_for_covariance_match(
    cov_ref,
    cov_match,
    bounds=(0.1, 5),
    tol=1e-5,
    maxiter=500,
):
    """
    Fit one scalar that best matches two batches of covariance matrices.

    The fitted scalar `d` rescales `cov_match` as `(d**2) * cov_match`, which
    corresponds to scaling ellipse radii by `d`. Both inputs are reshaped to
    `(-1, 2, 2)` before optimization.

    Parameters
    ----------
    cov_ref : array_like
        Reference covariance matrices with shape (..., 2, 2).
    cov_match : array_like
        Covariance matrices to be scaled, with the same shape as `cov_ref`.
    bounds : tuple, optional
        Lower and upper bounds for the fitted scaler.
    tol : float, optional
        Tolerance for warning when the optimum lands near a bound.
    maxiter : int, optional
        Maximum number of iterations for `minimize_scalar`.

    Returns
    -------
    bestfit_scaler : float
        Best-fitting scalar applied to ellipse radii.
    """
    cov_ref = np.asarray(cov_ref, dtype=float)
    cov_match = np.asarray(cov_match, dtype=float)

    if cov_ref.shape != cov_match.shape:
        raise ValueError(
            f"`cov_ref` and `cov_match` must have the same shape, got "
            f"{cov_ref.shape} and {cov_match.shape}."
        )
    if cov_ref.ndim < 2 or cov_ref.shape[-2:] != (2, 2):
        raise ValueError(
            f"Expected covariance inputs with shape (..., 2, 2), got {cov_ref.shape}."
        )

    cov_ref_flat = cov_ref.reshape(-1, 2, 2)
    cov_match_flat = cov_match.reshape(-1, 2, 2)

    #the squared Frobenius norm
    sum_sqerr = lambda d: np.sum(
        (cov_ref_flat - (d ** 2) * cov_match_flat) ** 2
    )

    res = minimize_scalar(
        sum_sqerr,
        bounds=bounds,
        method="bounded",
        options={"maxiter": maxiter},
    )

    bestfit_scaler = res.x

    if abs(bestfit_scaler - bounds[0]) < tol or abs(bestfit_scaler - bounds[1]) < tol:
        warnings.warn(
            f"bestfit_scaler={bestfit_scaler:.4f} is at or very near the bounds "
            f"[{bounds[0]}, {bounds[1]}]",
            RuntimeWarning,
        )

    return bestfit_scaler


def fit_best_scaler_for_ellparams_match(
    ellParams_ref,
    ellParams_match,
    bounds=(0.1, 5),
    tol=1e-5,
    maxiter=500,
):
    """
    Fit one scalar that best matches two batches of 2D ellipse parameters.

    The fitted scalar `d` rescales the semi-axis lengths in `ellParams_match`
    by `d`. Ellipse parameters are first converted into covariance matrices,
    then matched by calling `fit_best_scaler_for_covariance_match`.

    Parameters
    ----------
    ellParams_ref : array_like
        Reference ellipse parameters with shape (..., 5) for
        `[xc, yc, a, b, theta_deg]` or (..., 3) for `[a, b, theta_deg]`.
        Any center coordinates are ignored because covariance depends only on
        `(a, b, theta_deg)`.
    ellParams_match : array_like
        Ellipse parameters to be scaled, with the same shape as
        `ellParams_ref`.
    bounds : tuple, optional
        Lower and upper bounds for the fitted scaler.
    tol : float, optional
        Tolerance for warning when the optimum lands near a bound.
    maxiter : int, optional
        Maximum number of iterations for `minimize_scalar`.

    Returns
    -------
    bestfit_scaler : float
        Best-fitting scalar applied to ellipse radii.
    """
    ellParams_ref = np.asarray(ellParams_ref, dtype=float)
    ellParams_match = np.asarray(ellParams_match, dtype=float)

    if ellParams_ref.shape != ellParams_match.shape:
        raise ValueError(
            f"`ellParams_ref` and `ellParams_match` must have the same shape, got "
            f"{ellParams_ref.shape} and {ellParams_match.shape}."
        )
    if ellParams_ref.ndim < 1 or ellParams_ref.shape[-1] not in (3, 5):
        raise ValueError(
            "Expected ellipse-parameter inputs with shape (..., 5) for "
            "[xc, yc, a, b, theta_deg] or (..., 3) for [a, b, theta_deg], "
            f"got {ellParams_ref.shape}."
        )

    n_params = ellParams_ref.shape[-1]
    ellParams_ref_flat = ellParams_ref.reshape(-1, n_params)
    ellParams_match_flat = ellParams_match.reshape(-1, n_params)

    if n_params == 5:
        ellParams_ref_flat = ellParams_ref_flat[:, 2:]
        ellParams_match_flat = ellParams_match_flat[:, 2:]

    cov_ref = np.stack(
        [ellParamsQ_to_covMat(a, b, theta) for a, b, theta in ellParams_ref_flat],
        axis=0,
    ).reshape(ellParams_ref.shape[:-1] + (2, 2))
    cov_match = np.stack(
        [ellParamsQ_to_covMat(a, b, theta) for a, b, theta in ellParams_match_flat],
        axis=0,
    ).reshape(ellParams_match.shape[:-1] + (2, 2))

    return fit_best_scaler_for_covariance_match(
        cov_ref,
        cov_match,
        bounds=bounds,
        tol=tol,
        maxiter=maxiter,
    )


def scale_2d_ellipse_contour_from_params(
    ellParams,
    radii_scaler=1.0,
    ellipse_vis_scaler=1.0,
    nTheta=200,
):
    """
    Scale 2D ellipse radii from fitted ellipse parameters and render contours.

    Parameters
    ----------
    ellParams : array_like
        Ellipse parameters with shape (..., 5) encoded as
        `[xc, yc, a, b, theta_deg]`.
    radii_scaler : float, optional
        Scale factor applied to the semi-axis lengths `(a, b)`.
    ellipse_vis_scaler : float, optional
        Additional visualization-only scaling applied to rendered contour
        points about the ellipse center.
    nTheta : int, optional
        Number of contour samples used to render each ellipse.

    Returns
    -------
    fitEll_scaled : ndarray
        Visualization-scaled contour points with shape (..., 2, nTheta).
    fitEll_unscaled : ndarray
        Contour points after applying `radii_scaler` only, with shape
        (..., 2, nTheta).
    ellParams_scaled : ndarray
        Scaled ellipse parameters with shape (..., 5).
    """
    ellParams = np.asarray(ellParams, dtype=float)

    if ellParams.ndim < 1 or ellParams.shape[-1] != 5:
        raise ValueError(
            "Expected ellipse parameters with shape (..., 5) encoded as "
            f"[xc, yc, a, b, theta_deg], got {ellParams.shape}."
        )

    ellParams_scaled = ellParams.copy()
    ellParams_scaled[..., 2:4] *= radii_scaler

    ellParams_scaled_flat = ellParams_scaled.reshape(-1, 5)
    fitEll_unscaled_flat = np.empty((ellParams_scaled_flat.shape[0], 2, nTheta), dtype=float)

    for idx, (xc, yc, a, b, theta_deg) in enumerate(ellParams_scaled_flat):
        fit_x, fit_y = PointsOnEllipseQ(a, b, theta_deg, xc, yc, nTheta=nTheta)
        fitEll_unscaled_flat[idx] = np.stack((fit_x, fit_y), axis=0)

    fitEll_unscaled = fitEll_unscaled_flat.reshape(ellParams.shape[:-1] + (2, nTheta))
    centers = ellParams_scaled[..., :2][..., :, None]
    fitEll_scaled = (fitEll_unscaled - centers) * ellipse_vis_scaler + centers

    return fitEll_scaled, fitEll_unscaled, ellParams_scaled

#%%
def strip_trailing_zeros(s):
    return s.rstrip('0').rstrip('.')

#the following classes and methods are used to visualize boundary points and slices
#of L*a*b* that are within monitor's gamut
class FixedDim(IntEnum):
    """
    Indicates which CIELab dimension is held fixed when taking a 2D slice.

    L → Fix L*, vary (a*, b*)
    a → Fix a*, vary (L*, b*)
    b → Fix b*, vary (L*, a*)
    """
    L = 0   # Fix L*, look at slices in a*–b*
    a = 1   # Fix a*, look at slices in L*–b*
    b = 2   # Fix b*, look at slices in L*–a*

@dataclass(frozen=True)
class SlicePreset:
    fixed_dim: FixedDim                        # which dimension is fixed
    vis_range: Tuple[float, float, int]        # (start, stop, num) for np.linspace
    vectors_centroid: Tuple[float, float]      # 2D vector for your arrows/centroid
    varied_slc: Sequence[Tuple[float, float]]  # (N,2) array-like of sample points
    fixed_slc_idx: Sequence[int]               # indices into the visible range

    # Convenience accessors that return ready-to-use numpy arrays
    @property
    def val_fixed_vis(self) -> np.ndarray:
        lo, hi, num = self.vis_range
        return np.linspace(lo, hi, num)

    @property
    def val_varied_slc(self) -> np.ndarray:
        return np.asarray(self.varied_slc, dtype=float)

    @property
    def val_fixed_slc_idx(self) -> np.ndarray:
        return np.asarray(self.fixed_slc_idx, dtype=int)

    @property
    def vectors_centroids(self) -> np.ndarray:
        return np.asarray(self.vectors_centroid, dtype=float)
    
    @property
    def varied_dims(self) -> List[int]:
        """Indices of the two varying Lab dimensions for this slice."""
        return [i for i in (0, 1, 2) if i != int(self.fixed_dim)]

# ---- Define your three presets once ----
PRESETS: Dict[FixedDim, SlicePreset] = {
    FixedDim.L: SlicePreset(
        fixed_dim=FixedDim.L,
        vis_range=(10, 120, 23),
        vectors_centroid=(0, 0),
        varied_slc=[(0, 0), (-30, 30), (-15, -15), (15, -15), (5, 30)],
        fixed_slc_idx=list(range(12, 22, 2)),
    ),
    FixedDim.a: SlicePreset(
        fixed_dim=FixedDim.a,
        vis_range=(-60, 60, 23),
        vectors_centroid=(80, 50),
        varied_slc=[(90, 0), (83, 40), (83, -35), (100, -15), (93, 50)],
        fixed_slc_idx=list(range(8, 18, 2)),
    ),
    FixedDim.b: SlicePreset(
        fixed_dim=FixedDim.b,
        vis_range=(-70, 75, 23),
        vectors_centroid=(80, 0),
        varied_slc=[(95, 0), (85, 20), (82, -10), (100, -25), (75, 50)],
        fixed_slc_idx=list(range(8, 18, 2)),
    ),
}


def pretty_print_preset(preset):
    dim_names = {0: "L*", 1: "a*", 2: "b*"}
    fixed_name = dim_names[int(preset.fixed_dim)]
    varied = [d for d in (0,1,2) if d != int(preset.fixed_dim)]
    varied_names = [dim_names[d] for d in varied]

    vals = np.linspace(*preset.vis_range)                 # full fixed-dim sweep
    idx  = np.asarray(preset.fixed_slc_idx, dtype=int)    # chosen indices
    sel_vals = vals[idx]                                  # chosen fixed-dim values

    print("=== Stimulus Configuration ===")
    print(f"Fixed dimension       : {fixed_name} (code {int(preset.fixed_dim)})")
    print(f"Varied dimensions     : {varied_names} (codes {varied})")
    print(f"Fixed range (linspace): start={preset.vis_range[0]}, "
          f"stop={preset.vis_range[1]}, num={preset.vis_range[2]}")
    print(f"Selected fixed idx    : {idx.tolist()}")
    print(f"Selected fixed vals   : {np.round(sel_vals, 1).tolist()}")
    print(f"Vectors centroid      : {tuple(preset.vectors_centroids)}")
    print(f"Varied slice points   : {np.asarray(preset.val_varied_slc).tolist()}")


    
