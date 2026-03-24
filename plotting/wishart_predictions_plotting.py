#!/usr/bin/env python3
"""
Created on Mon Jul 29 11:15:34 2024

@author: fangfang
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import plotly.graph_objects as go
from matplotlib.patches import Polygon

script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from analysis.color_thres import color_thresholds
from analysis.ellipses_tools import covMat_to_ellParamsQ
from analysis.ellipsoids_tools import EllipsoidSurfaceMesh
from core import model_predictions, viz
from plotting.wishart_plotting import PlotSettingsBase, PlottingTools


# %%
@dataclass
class Plot2DPredSettings(PlotSettingsBase):
    fig_size: tuple[float, float] = field(default=(3, 3.5))
    flag_rescale_axes_label: bool = False
    visualize_gt: bool = False
    visualize_gt_only_wSamples: bool = False
    visualize_model_pred: bool = True
    visualize_samples: bool = True
    visualize_model_estimatedCov: bool = False
    samples_colorcoded_resp: bool = False
    samples_s: float = 10
    samples_alpha: float = 0.5
    samples_label: str = "Simulated CIELab data"
    samples_c_no: str | np.ndarray | list[float] = field(default_factory=lambda: np.array([107, 142, 35]) / 255)
    samples_c_yes: str | np.ndarray | list[float] = field(default_factory=lambda: np.array([128, 0, 0]) / 255)
    gt_lc: str | np.ndarray | list[float] = "r"
    gt_alpha: float = 1.0
    gt_ls: str = "--"
    gt_label: str = "Ground truths"
    gt_lw: float = 2
    ticks: np.ndarray | None = None
    plane_2D: str | None = None  # You can set this in code after instantiating with color_thres.plane_2D
    modelpred_lc: str | None = None
    modelpred_ls: str = "-"
    modelpred_lw: float = 1
    modelpred_label: str | None = None
    modelpred_alpha: float = 0.5
    sigma_lc: str | np.ndarray | list[float] = "k"
    sigma_ls: str = "-"
    sigma_lw: float = 1
    sigma_alpha: float = 0.5
    sigma_label: str = "Model-estimated cov matrix"
    legend_off: bool = False
    anchor_legend_box: tuple[float, float] = field(default=(0.5, -0.47))
    title: str | None = None
    fig_name: str = field(
        default_factory=lambda: f"Fitted_isothreshold_ellipses_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


@dataclass
class Plot3DPredSettings(PlotSettingsBase):
    fig_size: tuple[float, float] = field(default=(8, 5.5))
    flag_rescale_axes_label: bool = False
    visualize_model_pred: bool = True
    visualize_samples: bool = True
    visualize_model_estimatedCov: bool = False
    visualize_gt: bool = True
    visualize_modelpred_CI: bool = False
    visualize_gt_proj_slice_one_plot: bool = False
    gt_3Dproj_lw: float = 2
    modelpred_polygon_alpha: float = 0.3
    gt_3Dproj_lc: str | np.ndarray | list[float] | None = None
    gt_3Dproj_ls: str = "-"
    gt_lc: str | np.ndarray | list[float] = "r"
    gt_alpha: float = 0.5
    gt_ls: str = "--"
    gt_lw: float = 2
    samples_s: float = 10
    samples_alpha: float = 0.5
    samples_label: str = "Simulated CIELab data"
    modelpred_lc: str | np.ndarray | list[float] = "g"
    modelpred_ls: str = "-"
    modelpred_lw: float = 3
    modelpred_alpha: float = 0.5
    modelpred_projection_CI: np.ndarray | None = None
    modelpred_slice_CI: np.ndarray | None = None
    modelpred_CI_alpha: float = 0.7
    sigma_lc: str | np.ndarray | list[float] = "k"
    sigma_lw: float = 1
    sigma_alpha: float = 0.5
    sigma_label: str = "Model-estimated cov matrix"
    axes_samples: list[int] = field(default_factory=lambda: [0])
    fixedRGB_val_scaled: list[float] = field(default_factory=lambda: [-0.6, -0.3, 0, 0.3, 0.6])
    scatter_label: str = "Simulated CIELab data"
    contour_3D_label: str = "3D ground truths"
    contour_2D_label: str = "2D ground truths"
    fits_label: str = "Model-estimated cov matrix"
    pred3D_label: str = "Model-predicted ellipsoid projections"
    pred2D_label: str = "Model-predicted ellipsoid slices"
    CI_3D_label: str = "Range of model-predicted projections"
    CI_2D_label: str = "Range of model-predicted slices"
    # These values are fixed and not meant to be changed after creation
    dim_indices: list[int] = field(default_factory=lambda: [0, 1, 2], init=False, repr=False)
    dim_labels: list[str] = field(default_factory=lambda: ["R", "G", "B"], init=False, repr=False)
    orthogonal_pairs: list[list[int]] = field(default_factory=lambda: [[1, 2], [0, 2], [0, 1]], init=False, repr=False)
    plane_labels: list[str] = field(default_factory=lambda: ["GB", "RB", "RG"], init=False, repr=False)
    fig_name: str = field(
        default_factory=lambda: f"Fitted_isothreshold_ellipsoids_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


@dataclass
class Plot3DPredHTMLSettings:
    disable_hover: bool = True  # universal disabling
    # Fonts / labels
    font_family: str = "Arial"
    font_size: float = 10
    xlabel: str = "Model space dimension 1"
    ylabel: str = "Model space dimension 2"
    zlabel: str = "Model space dimension 3"
    # Axes
    ticks: np.ndarray = field(default_factory=lambda: np.linspace(-0.7, 0.7, 5))
    lim: list[float] = field(default_factory=lambda: [-1, 1])
    hide_spikes: bool = True
    # Camera
    camera_eye: tuple[float, float, float] = (-1.8, -1.8, 1.2)
    camera_center: tuple[float, float, float] = (0, 0, 0)
    camera_up: tuple[float, float, float] = (0, 0, 1)
    # Plane style
    isoluminant_plane_color: np.ndarray | list[float] = field(default_factory=lambda: np.array([0.75, 0.75, 0.75]))
    isoluminant_edge_color: np.ndarray | list[float] = field(default_factory=lambda: np.array([0, 0, 0]))
    isoluminant_plane_alpha: float = 0.30
    isoluminant_plane_lw: float = 1.0
    plane_hover_info: str = "skip"
    # Ref points
    ref_hover_info: str = "skip"
    ref_size: float = 5
    ref_marker: str = "circle"
    # Ellipsoids
    ell_mesh_nu: int = 120
    ell_mesh_nv: int = 240
    ell_alpha: float = 0.5
    ell_scaler: float = 1
    ell_hover_info: str = "skip"  # 3D ellipsoids
    # sliced ellipses
    sEll_hover_info: str = "skip"  # sliced ellipses
    sEll_line_color: np.ndarray | list[float] = field(default_factory=lambda: np.array([0, 0, 0]))
    sEll_line_width: float = 5.0
    # Lighting
    flag_lighting: bool = True
    light_ambient: float = 0.25
    light_diffuse: float = 0.80
    light_specular: float = 0.10
    light_roughness: float = 0.15
    light_fresnel: float = 0.05
    light_position: tuple[float, float, float] = (10.0, 10.0, 10.0)
    # Lines
    geo_path_lw: float = 2.0
    geo_path_color: np.ndarray | list[float] = field(default_factory=lambda: np.array([0, 0, 0]))
    geo_hover_info: str = "skip"


@dataclass
class PlotMahaSettings(PlotSettingsBase):
    fig_size: tuple[float, float] | None = None
    visualize_68CI_95CI: list[bool] = field(default_factory=lambda: [True, False])
    visualize_SE: bool = True
    visualize_hist: bool = False
    visualize_timecourse: bool = True
    p_SE: list[float] = field(default_factory=lambda: [0.667])
    markersize: float = 50
    markercolor: tuple[float, float, float] | str = (0.3, 0.3, 0.3)
    edgecolor: tuple[float, float, float] | str = (1.0, 1.0, 1.0)
    linecolor: np.ndarray = field(default_factory=lambda: np.array([70, 130, 180]) / 255)
    CI_color: np.ndarray = field(default_factory=lambda: np.array([167, 199, 231]) / 255)
    xlabel: str = "Binned Mahalanobis distance"
    ylabel: str = "Binned percent correct"
    yticks: np.ndarray = field(default_factory=lambda: np.round(np.linspace(0.333, 1, 3), 3))
    xticks: np.ndarray = field(default_factory=lambda: np.linspace(0, 20, 6))
    ylim: list[float] = field(default_factory=lambda: [0.2, 1.05])
    xlim: list[float] | None = None
    marker: list[str] = field(default_factory=lambda: ["s", "o"])
    label: list[str] = field(default_factory=lambda: ["Strategy: Sobol", "Strategy: EAVC"])
    # Error bar styling
    err_capsize: float = 3
    err_capthick: float = 1
    err_lw: float = 1  # Line width for error bars
    PMF_lw: float = 2
    # Custom labels
    PMF_label: str = "Simulated psychometric function"
    SE_label: str = "SE for a Bernoulli random variable"
    fig_name: str = field(default_factory=lambda: f"Model_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


@dataclass
class PlotMCGeoSettings(PlotSettingsBase):
    fig_size: tuple[float, float] = (4, 4)
    xlabel: str | None = None
    ylabel: str | None = None
    path_lc: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.6, 0.6], dtype=float))
    path_lw: float = 0.2
    path_alpha: float = 0.5
    samples_size: float = 2.0
    samples_alpha: float = 0.4
    samples_labels: tuple[str, str, str] = ("z0", "z1", "z2")
    surf_alpha: float = 0.5
    diff_nBins: int = 100
    legend_off: bool = False
    anchor_legend_box: list | None = None
    legend_loc: str = "lower left"
    flag_show_cond_title: bool = True
    fig_name: str = field(default_factory=lambda: f"Geopath_varyingComp2_{datetime.now():%Y%m%d_%H%M%S}")


# %%
class WishartPredictionsVisualization(PlottingTools):
    def __init__(
        self,
        trial_data,
        model,
        model_pred,
        color_thres,
        settings: PlotSettingsBase,
        save_fig=False,
        save_format="pdf",
    ):
        """
        Initialize an instance of sampling_ref_comp_pair_visualization, a subclass of
        wishart_model_basics_visualization, which extends its functionality for specific
        visualization tasks related to sampling reference and comparison stimulus pairs.

        """
        super().__init__(settings, save_fig, save_format)
        self.trial_data = trial_data
        self.model = model
        self.model_pred = model_pred
        self.color_thres = color_thres

    def _find_idx_corresponding_xref(self, ref_all, x1_all, ref_slc, tol=5e-2, y_all=None, further_sort_by_resp=False):
        """
        Finds the indices of rows in 'ref_all' that match the reference slice 'ref_slc'
        within a specified tolerance. Extracts the corresponding rows from 'x1_all'
        and returns them along with the matched rows and indices.

        Parameters and Returns are the same as described above.

        Parameters:
        -----------
        ref_all : numpy.ndarray
            The array of reference data with shape (N, 2) from which we want to find the matching rows.

        x1_all : numpy.ndarray
            The array of data corresponding to 'ref_all' with the same shape (N, 2).
            The rows corresponding to the matched indices will be returned.

        ref_slc : numpy.ndarray
            A single row array with shape (1, 2) that we want to match against the rows in 'ref_all'.

        tol : float, optional
            The tolerance value used to determine a match between 'ref_slc' and rows in 'ref_all'.
            Default is 1e-4.

        """

        # Determine matching indices based on tolerance
        match_indices = np.where(np.all(np.abs(ref_all - ref_slc) < tol, axis=1))[0]

        if further_sort_by_resp and y_all is not None:
            # Separate matches based on response values
            yes_indices = match_indices[np.abs(y_all[match_indices] - 1) < tol]
            no_indices = match_indices[np.abs(y_all[match_indices]) < tol]

            return (
                [x1_all[yes_indices], x1_all[no_indices]],
                [ref_all[yes_indices], ref_all[no_indices]],
                [yes_indices, no_indices],
            )
        else:
            # Return matched rows and corresponding indices
            return x1_all[match_indices], ref_all[match_indices], match_indices

    def _org_Mahalanobis_distance_trial_ID(self, trial_ID, resp, nTrials_bin_list):
        """
        Organizes and bins Mahalanobis distances and response data by trial type.

        This method processes Mahalanobis distance data and corresponding responses
        for each unique trial type in the provided `trial_ID` list. For each trial type,
        the data is sorted by Mahalanobis distance and grouped into bins of size
        specified in `nTrials_bin_list`. If the number of trials for a given trial type
        is not divisible by the bin size, the method pads the data with np.nan to allow
        reshaping and binning.

        Parameters:
        ----------
        trial_ID : list or array-like
            A list of trial identifiers, where each element corresponds to a unique trial type.
            The length of `trial_ID` must match the number of entries in `resp`.
        resp : ndarray of shape (N,)
            An array of binary or continuous responses corresponding to each trial.
        nTrials_bin_list : list of int
            A list specifying the number of trials per bin for each unique trial type.
            The list should have the same length as the number of unique trial types.
            If the number of trials for a trial type is not divisible by the corresponding
            bin size, np.nan values are appended to pad the array before binning.

        Returns:
        -------
        Maha_data : dict
            A dictionary containing the organized and binned data for each trial type:
            - 'trial_type': ndarray of shape (num_trial_types,)
                Sorted array of unique trial types.
            - 'M_D_sorted': list of ndarrays
                Mahalanobis distances sorted for each trial type.
            - 'resp_sorted': list of ndarrays
                Responses sorted to align with sorted Mahalanobis distances for each trial type.
            - 'M_D_binned': list of ndarrays
                Binned mean Mahalanobis distances for each trial type. Uses np.nanmean if padded.
            - 'resp_binned': list of ndarrays
                Binned mean responses for each trial type. Uses np.nanmean if padded.
            - 'nTrials': list of ints
                Total number of trials for each trial type (before padding).
        """
        # Unique trial types
        trial_type_unsorted = np.unique(trial_ID)
        trial_type = np.sort(trial_type_unsorted)

        # Prepare lists to store results
        (
            M_D_unsorted,
            M_D_sorted,
            resp_unsorted,
            resp_sorted,
            M_D_binned,
            resp_binned,
            nTrials,
        ) = [], [], [], [], [], [], []

        for i in range(len(trial_type)):
            nTrials_bin_i = nTrials_bin_list[i]
            """
            Process and bin data for each trial type.
    
            Steps:
            1. Select data indices corresponding to the current trial type.
            2. Extract Mahalanobis distances and response values for these indices.
            3. Sort the extracted data by Mahalanobis distances.
            4. Trim data to ensure divisibility by the bin size (`nTrials_bin`).
            5. Reshape and compute the mean for each bin.
            6. Store the results for plotting.
            """
            # Select indices for the current trial type
            idx_slc = np.where(trial_ID == trial_type[i])[0]
            nTrials.append(len(idx_slc))

            # Extract Mahalanobis distances and responses for the current trial type
            M_D_selected_i = self.model_pred.mahalanobis_distances[idx_slc]
            resp_selected_i = resp[idx_slc]

            # Sort by Mahalanobis distances
            sorted_indices = np.argsort(M_D_selected_i)  # Indices to sort Mahalanobis distances
            M_D_sorted_i = M_D_selected_i[sorted_indices]  # Sorted Mahalanobis distances
            resp_sorted_i = resp_selected_i[sorted_indices]  # Responses sorted accordingly

            # Ideally, we want to ensure data size is divisible by the bin size,
            # but if the total trial number is a prime number, then there is no
            # way to do divide them evenly, so for the last bin we can just pad nan
            if len(M_D_sorted_i) % nTrials_bin_i != 0:
                print(f"Data size {len(M_D_sorted_i)} is not divisible by the bin size {nTrials_bin_i}.")
                # Pad with NaNs to make the length divisible by the bin size
                num_pad_nan = nTrials_bin_i - (len(M_D_sorted_i) % nTrials_bin_i)
                M_D_sorted_i = np.pad(M_D_sorted_i, (0, num_pad_nan), constant_values=np.nan)
                resp_sorted_i = np.pad(
                    resp_sorted_i.astype(float),
                    (0, num_pad_nan),
                    constant_values=np.nan,
                )

            D_M_reshaped = M_D_sorted_i.reshape(-1, nTrials_bin_i)
            resp_reshaped = resp_sorted_i.reshape(-1, nTrials_bin_i)

            M_D_binned_i = np.nanmean(D_M_reshaped, axis=1)  # Compute mean Mahalanobis distance for each bin
            resp_binned_i = np.nanmean(resp_reshaped, axis=1)  # Compute mean response for each bin

            # Store binned data for later use in plotting
            M_D_unsorted.append(M_D_selected_i)
            resp_unsorted.append(resp_selected_i)
            M_D_sorted.append(M_D_sorted_i)  # Original selected Mahalanobis distances
            resp_sorted.append(resp_sorted_i)  # Original selected responses
            M_D_binned.append(M_D_binned_i)  # Binned Mahalanobis distances
            resp_binned.append(resp_binned_i)  # Binned responses

        # Save all data into a dictionary after processing
        Maha_data = {
            "trial_type": trial_type,
            "M_D_unsorted": M_D_unsorted,
            "M_D_sorted": M_D_sorted,
            "resp_unsorted": resp_unsorted,
            "resp_sorted": resp_sorted,
            "M_D_binned": M_D_binned,
            "resp_binned": resp_binned,
            "nTrials": nTrials,
        }

        return Maha_data

    def _check_nTrials_bin(self, nTrials_bin, num_trial_types):
        """
        Validates and processes the nTrials_bin input.

        Parameters:
        ----------
        nTrials_bin : int or list of int
            Either a single integer applied to all trial types, or a list of integers specifying
            bin sizes for each trial type.
        trial_type : array-like
            Sorted array of unique trial types.

        Returns:
        -------
        nTrials_bin_list : list of int
            A list of bin sizes corresponding to each trial type.
        """
        if isinstance(nTrials_bin, int):
            # Use the same bin size for all trial types
            nTrials_bin_list = [nTrials_bin] * num_trial_types
        elif isinstance(nTrials_bin, list):
            if not all(isinstance(x, int) for x in nTrials_bin):
                raise ValueError("All elements in nTrials_bin list must be integers.")
            if len(nTrials_bin) < num_trial_types:
                # Pad with the last value
                nTrials_bin_list = nTrials_bin + [nTrials_bin[-1]] * (num_trial_types - len(nTrials_bin))
            else:
                # Truncate to match number of trial types
                nTrials_bin_list = nTrials_bin[:num_trial_types]
        else:
            raise TypeError("nTrials_bin must be either an int or a list of ints.")

        return nTrials_bin_list

    def _get_colormap(self, grid_point):
        """
        Convert a grid point in W space to an RGB color suitable for plotting.

        Parameters
        ----------
        grid_point : np.ndarray or list
            A 2-element vector representing the stimulus location in W space.

        Returns
        -------
        cm : np.ndarray
            A 3-element vector (R, G, B) representing the RGB color.
        """
        # Handle Isoluminant plane case (append fixed luminance of 1 and transform)
        if self.color_thres.plane_2D in ["Isoluminant plane", "LSisolating plane"]:
            cm = self.color_thres.W2D_to_rgb(grid_point)
        else:
            cm = self.color_thres.W_unit_to_N_unit(grid_point)
            # Insert or append the fixed value depending on index
            if self.color_thres.fixed_color_dim != len(cm):
                cm = np.insert(cm, self.color_thres.fixed_color_dim, self.color_thres.fixed_value)
            else:
                cm = np.append(cm, self.color_thres.fixed_value)

        return cm

    def _get_legend_labels_2D(self, i, j, settings):
        """
        Determines legend labels for different plot elements to avoid duplicates.

        Parameters
        ----------
        i, j : int
            Indices for the current subplot location.
        settings : Plot2DPredSettings
            The settings object containing plot label configurations.

        Returns
        -------
        scatter_label : str or None
        ellipse_label : str or None
        prediction_label : str or None
        gt_label : str or None
        """
        if i == 0 and j == 0:
            scatter_label = settings.samples_label
            ellipse_label = settings.sigma_label
            gt_label = settings.gt_label
            if settings.modelpred_label is not None:
                prediction_label = settings.modelpred_label
            else:
                prediction_label = "Model predictions"
                if self.model_pred and hasattr(self.model_pred, "target_pC"):
                    prediction_label += f" (pCorrect = {self.model_pred.target_pC:.3f})"

        else:
            scatter_label = None
            ellipse_label = None
            prediction_label = None
            gt_label = None

        return scatter_label, ellipse_label, prediction_label, gt_label

    def _get_legend_labels_3D(self, i, j, fixedRGB_val_scaled_k, fixedPlane, settings):
        """
        Determines legend labels for 3D visualization to avoid duplicates and include
        axis-specific labeling.

        Parameters
        ----------
        i, j : int
            Indices of the current subplot location.
        fixedRGB_val_scaled_k : float
            The value of the fixed axis (e.g., R, G, or B).
        fixedPlane : str
            The name of the fixed dimension ('R', 'G', or 'B').
        settings : Plot3DPredSettings
            Settings containing labels and formatting preferences.

        Returns
        -------
        scatter_label : str or None
        contour_3D_label : str or None
        contour_2D_label : str or None
        fits_label : str or None
        pred3D_label : str or None
        pred2D_label : str or None
        CI_3D_label : str or None
        CI_2D_label : str or None
        fixedRGB_val_temp : float or None
        """
        if i == 0 and j == 0:
            fixedRGB_val_temp = self.color_thres.W_unit_to_N_unit(fixedRGB_val_scaled_k)

            contour_2D_label = (
                f"{settings.contour_2D_label}\nformed at the "
                f"intersection\nof 3D ellipsoids with {fixedPlane}={fixedRGB_val_temp}"
            )

            return (
                settings.scatter_label,
                settings.contour_3D_label,
                contour_2D_label,
                settings.fits_label,
                settings.pred3D_label,
                settings.pred2D_label,
                settings.CI_3D_label,
                settings.CI_2D_label,
                fixedRGB_val_temp,
            )
        else:
            return (None, None, None, None, None, None, None, None, None)

    def _initialize_axes(self, ax, Maha_data, settings):
        """Initialize figure and axes based on trial types."""
        n_trials = len(Maha_data["trial_type"])
        if ax is None:
            if n_trials == 1:
                fsize = settings.fig_size or (3.2, 3.8)  # (4.8, 5.5)
                fig, ax = plt.subplots(1, 1, figsize=fsize, dpi=settings.dpi)
                ax = [ax]  # Wrap in a list
            else:
                fsize = settings.fig_size or (3.2, 3.8 + (n_trials - 1))  # (4.8, 6.5)
                fig, ax = plt.subplots(n_trials, 1, figsize=fsize, dpi=settings.dpi)
                ax = ax.ravel()
        else:
            fig = ax.figure
            if not isinstance(ax, (list, np.ndarray)):
                ax = [ax]
        return ax, fig

    # %%
    def plot_2D(self, grid_est, settings: Plot2DPredSettings, gt_ellipses=None, ax=None):
        """
        Visualizes the Wishart model predictions, ground truth ellipses, and optionally
        simulated/experimentally tested trials.

        Parameters
        ----------
        grid_est : numpy.ndarray, shape: (N x N x 2)
            A grid of reference stimuli where the model predictions/estimates will be plotted.

        grid_samples : numpy.ndarray, shape: (M x M x 2)
            A grid of reference stimuli used for experiments or simulations to examine the
            model predictions. The shape does not need to match `grid_est` since predictions
            can be made on stimuli that were not experimentally tested.

        gt_ellipses : numpy.ndarray, optional, shape: (N x N x 2 x nTheta)
            The ground truth ellipses for comparison.

        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. If None, a new figure and axes will be created. Default is None.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.

        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """

        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=settings.fig_size, dpi=settings.dpi)
        else:
            fig = ax.figure

        # Iterate over each point in the estimated grid to plot the corresponding data.
        for ij in np.ndindex(*grid_est.shape[0:2]):
            # define the color map, which is the RGB value of the reference stimulus
            cm = self._get_colormap(grid_est[ij])
            # Set labels for different plot elements, ensuring they only appear once in the legend.
            scatter_label, ellipse_label, prediction_label, gt_label = self._get_legend_labels_2D(*ij, settings)

            # If visualizing sample data, scatter the points from the simulation trials.
            if settings.visualize_samples and (self.trial_data.x1_all.shape[0] != 0):
                if settings.samples_colorcoded_resp:
                    x1_slc, *_ = self._find_idx_corresponding_xref(
                        self.trial_data.xref_all,
                        self.trial_data.x1_all,
                        np.reshape(grid_est[ij], (1, -1)),
                        y_all=self.trial_data.y_all,
                        further_sort_by_resp=True,
                    )
                    for kk in range(2):
                        if ij == (0, 0) and kk == 0:
                            lbl_kk = scatter_label + " (correct resp)"
                        else:
                            lbl_kk = None
                        c_kk = settings.samples_c_yes if kk == 1 else settings.samples_c_no
                        ax.scatter(
                            x1_slc[kk][:, 0],
                            x1_slc[kk][:, 1],
                            color=c_kk,
                            s=settings.samples_s,
                            linewidth=0,
                            alpha=0.5,
                            label=lbl_kk,
                        )
                else:
                    x1_slc, *_ = self._find_idx_corresponding_xref(
                        self.trial_data.xref_all,
                        self.trial_data.x1_all,
                        np.reshape(grid_est[ij], (1, -1)),
                    )

                    ax.scatter(
                        x1_slc[:, 0],
                        x1_slc[:, 1],
                        color=tuple(cm),
                        s=settings.samples_s,
                        linewidth=0,
                        alpha=settings.samples_alpha,
                        label=scatter_label,
                    )
                if settings.visualize_gt_only_wSamples and (gt_ellipses is not None):
                    ax.plot(
                        *gt_ellipses[ij],
                        c=settings.gt_lc,
                        alpha=settings.gt_alpha,
                        ls=settings.gt_ls,
                        lw=settings.gt_lw,
                        label=gt_label,
                    )
            # Plot the model-estimated covariance ellipses.
            if settings.visualize_model_estimatedCov:
                viz.plot_ellipse(
                    ax,
                    grid_est[ij],
                    self.model_pred.Sigmas_noise_grid[ij],
                    c=settings.sigma_lc,
                    alpha=settings.sigma_alpha,
                    ls=settings.sigma_ls,
                    lw=settings.sigma_lw,
                    label=ellipse_label,
                )
            if settings.visualize_model_pred:
                if settings.modelpred_lc is None:
                    lineC = cm
                else:
                    lineC = settings.modelpred_lc
                # Plot the model predictions as lines.
                ax.plot(
                    *self.model_pred.fitEll_unscaled[ij],
                    c=lineC,
                    lw=settings.modelpred_lw,
                    alpha=settings.modelpred_alpha,
                    ls=settings.modelpred_ls,
                    label=prediction_label,
                )
            # Optionally plot the ground truth ellipses.
            if settings.visualize_gt and (gt_ellipses is not None):
                ax.plot(
                    *gt_ellipses[ij],
                    c=settings.gt_lc,
                    alpha=settings.gt_alpha,
                    ls=settings.gt_ls,
                    lw=settings.gt_lw,
                    label=gt_label,
                )
        tickmarks = settings.ticks if settings.ticks is not None else np.unique(grid_est)
        if settings.flag_rescale_axes_label:
            tickmarks_show = self.color_thres.W_unit_to_N_unit(tickmarks)
        else:
            tickmarks_show = tickmarks
        self._update_axes_limits(ax)
        self._update_axes_labels(ax, tickmarks, tickmarks_show)
        self._configure_labels_and_title(ax, title=settings.title)
        ax.set_aspect("equal", adjustable="box")
        plt.grid(alpha=0.2)
        if not settings.legend_off:
            plt.legend(
                loc="lower center",
                bbox_to_anchor=settings.anchor_legend_box,
                fontsize=settings.fontsize,
            )
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = pltP['fontsize'])
        fig.tight_layout()
        # Save the figure if the directory is set and saving is enabled.
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
        return fig, ax

    # %%
    def plot_3D(
        self,
        grid_est,
        settings: Plot3DPredSettings,
        gt_covMat=None,
        gt_slice_2d_ellipse=None,
    ):
        self.ndims = 2  # plotting projections

        # compute the model predicted covariance matrix sliced by a 2D plane.
        # in older version of the code, this was not computed and saved in model_pred
        if settings.visualize_model_estimatedCov:
            if hasattr(self.model_pred, "Sigmas_noise_grid_slice_2d"):
                model_pred_cov_2d_slice = self.model_pred.Sigmas_noise_grid_slice_2d
            else:
                temp = model_predictions.covMat3D_to_2DsurfaceSlice(self.model_pred.Sigmas_noise_grid)
                model_pred_cov_2d_slice = np.transpose(temp, (1, 0, 2, 3, 4, 5))

        NUM_GRID_PTS = grid_est.shape[0]
        for k, fixedRGB_val_scaled_k in zip(list(range(NUM_GRID_PTS)), settings.fixedRGB_val_scaled):  # noqa: B905
            for _idx, fixedPlane, _idx_varying, varyingPlanes in zip(  # noqa: B905
                settings.dim_indices,
                settings.dim_labels,
                settings.orthogonal_pairs,
                settings.plane_labels,
            ):
                fig, axes = plt.subplots(1, 2, dpi=settings.dpi, figsize=settings.fig_size)
                idx = jnp.array(_idx_varying)

                # plot model prediction 95% confidence interval
                if settings.visualize_modelpred_CI:
                    if settings.modelpred_projection_CI is not None:
                        fitEll_min, fitEll_max = settings.modelpred_projection_CI
                    if settings.modelpred_slice_CI is not None:
                        fitEll_s_min, fitEll_s_max = settings.modelpred_slice_CI

                for i in range(NUM_GRID_PTS):
                    for j in range(NUM_GRID_PTS):
                        if _idx == 0:
                            ii, jj, kk = k, i, j
                        elif _idx == 1:
                            ii, jj, kk = i, k, j
                        elif _idx == 2:
                            ii, jj, kk = i, j, k
                        # lables
                        (
                            scatter_label,
                            contour_3D_label,
                            contour_2D_label,
                            fits_label,
                            pred3D_label,
                            pred2D_label,
                            CI_3D_label,
                            CI_2D_label,
                            fixedRGB_val_temp,
                        ) = self._get_legend_labels_3D(i, j, fixedRGB_val_scaled_k, fixedPlane, settings)
                        # colormap
                        cm = self.color_thres.W_unit_to_N_unit(grid_est[ii, jj, kk])

                        # model-predicted ellipses (projections)
                        if settings.visualize_model_pred:
                            polygon = Polygon(
                                self.model_pred.fitEll_unscaled[ii, jj, kk][idx].T,
                                closed=True,
                                fill=True,
                                color=np.array(cm),
                                alpha=settings.modelpred_polygon_alpha,
                                label=pred3D_label,
                            )
                            axes[0].add_patch(polygon)
                        else:
                            if settings.visualize_modelpred_CI and settings.modelpred_projection_CI is not None:
                                idx_max_nonan = ~np.isnan(fitEll_max[ii, jj, kk, _idx, 0, :])
                                axes[0].fill(
                                    fitEll_max[ii, jj, kk, _idx, 0, idx_max_nonan],
                                    fitEll_max[ii, jj, kk, _idx, 1, idx_max_nonan],
                                    color=np.array(cm),
                                    label=CI_3D_label,
                                    alpha=settings.modelpred_CI_alpha,
                                    lw=0,
                                )
                                idx_min_nonan = ~np.isnan(fitEll_min[ii, jj, kk, _idx, 0, :])
                                axes[0].fill(
                                    fitEll_min[ii, jj, kk, _idx, 0, idx_min_nonan],
                                    fitEll_min[ii, jj, kk, _idx, 1, idx_min_nonan],
                                    color="white",
                                    lw=0,
                                )

                        # visualize the ground truth 3D projections on 2D
                        if settings.visualize_gt:
                            if settings.gt_3Dproj_lc is None:
                                c = np.array(cm)
                            else:
                                c = settings.gt_3Dproj_lc
                            viz.plot_ellipse(
                                axes[0],
                                grid_est[ii, jj, kk, idx],
                                gt_covMat[ii, jj, kk][idx][:, idx],
                                ls=settings.gt_3Dproj_ls,
                                color=c,
                                lw=settings.gt_3Dproj_lw,
                                label=contour_3D_label,
                            )  # np.array(cm)

                        # visualize the fits
                        if settings.visualize_model_estimatedCov:
                            sig_rec = self.model_pred.Sigmas_noise_grid[jj, ii, kk]
                            viz.plot_ellipse(
                                axes[0],
                                grid_est[ii, jj, kk, idx],
                                sig_rec[idx][:, idx],
                                color=settings.sigma_lc,
                                lw=settings.sigma_lw,
                                linestyle=settings.sigma_ls,
                                alpha=settings.sigma_alpha,
                                label=fits_label,
                            )

                            viz.plot_ellipse(
                                axes[1],
                                grid_est[ii, jj, kk, idx],
                                model_pred_cov_2d_slice[ii, jj, kk, _idx],
                                color=settings.sigma_lc,
                                lw=settings.sigma_lw,
                                linestyle=settings.sigma_ls,
                                alpha=settings.sigma_alpha,
                                label=fits_label,
                            )

                        # 2d ground truth slices
                        if settings.visualize_gt:
                            viz.plot_ellipse(
                                axes[1],
                                grid_est[ii, jj, kk, idx],
                                gt_slice_2d_ellipse[ii, jj, kk, _idx],
                                color=settings.gt_lc,
                                linestyle=settings.gt_ls,
                                lw=settings.gt_lw,
                                alpha=settings.gt_alpha,
                                label=contour_2D_label,
                            )
                            if settings.visualize_gt_proj_slice_one_plot:
                                viz.plot_ellipse(
                                    axes[1],
                                    grid_est[ii, jj, kk, idx],
                                    gt_covMat[ii, jj, kk][idx][:, idx],
                                    ls="-",
                                    color=c,
                                    lw=settings.gt_3Dproj_lw,
                                    label=contour_3D_label,
                                )  # np.array(cm)

                        # 2d model predicted slices
                        if settings.visualize_model_pred:
                            if settings.modelpred_lc is None:
                                cmap = np.array(cm)
                            else:
                                cmap = settings.modelpred_lc
                            viz.plot_ellipse(
                                axes[1],
                                grid_est[ii, jj, kk, idx],
                                self.model_pred.pred_slice_2d_ellipse[ii, jj, kk, _idx],
                                ls=settings.modelpred_ls,
                                lw=settings.modelpred_lw,
                                color=cmap,
                                alpha=settings.modelpred_alpha,
                                label=pred2D_label,
                            )

                        else:
                            if settings.visualize_modelpred_CI and settings.modelpred_slice_CI is not None:
                                idx_max_nonan = ~np.isnan(fitEll_s_max[ii, jj, kk, _idx, 0, :])
                                axes[1].fill(
                                    fitEll_s_max[ii, jj, kk, _idx, 0, idx_max_nonan],
                                    fitEll_s_max[ii, jj, kk, _idx, 1, idx_max_nonan],
                                    color=np.array(cm),
                                    label=CI_2D_label,
                                    alpha=settings.modelpred_CI_alpha,
                                    lw=0,
                                )
                                idx_min_nonan = ~np.isnan(fitEll_s_min[ii, jj, kk, _idx, 0, :])
                                axes[1].fill(
                                    fitEll_s_min[ii, jj, kk, _idx, 0, idx_min_nonan],
                                    fitEll_s_min[ii, jj, kk, _idx, 1, idx_min_nonan],
                                    color="white",
                                    lw=0,
                                )

                # plot xref
                if settings.visualize_samples:
                    slc_idx_samples = np.where(
                        np.abs(self.trial_data.xref_all[:, _idx] - (fixedRGB_val_scaled_k)) < 1e-4
                    )
                    xref_jnp_slc_temp = self.trial_data.xref_all[slc_idx_samples]
                    x1_jnp_slc_temp = self.trial_data.x1_all[slc_idx_samples]
                    x1_jnp_slc = x1_jnp_slc_temp[:, idx]
                    for a in settings.axes_samples:
                        axes[a].scatter(
                            x1_jnp_slc[:, 0],
                            x1_jnp_slc[:, 1],
                            s=settings.samples_s,
                            c=self.color_thres.W_unit_to_N_unit(xref_jnp_slc_temp),
                            alpha=settings.samples_alpha,
                        )
                axes[0].set_aspect("equal")
                axes[1].set_aspect("equal")
                axes[0].grid(True, alpha=0.3)
                axes[1].grid(True, alpha=0.3)
                ticks = np.unique(grid_est)
                if settings.flag_rescale_axes_label:
                    ticks_show = self.color_thres.W_unit_to_N_unit(ticks)
                else:
                    ticks_show = ticks
                for n in range(2):  # 2 axes
                    self._update_axes_limits(axes[n])
                    self._update_axes_labels(axes[n], ticks, ticks_show)
                    ttl_part2 = (
                        f"{varyingPlanes} plane ({fixedPlane} = "
                        + f"{self.color_thres.W_unit_to_N_unit(fixedRGB_val_scaled_k)})"
                    )
                    if n == 0:
                        ttl_part1 = "Projections onto "
                    else:
                        ttl_part1 = "Slices by "
                    self._configure_labels_and_title(axes[n], title=ttl_part1 + ttl_part2)
                    axes[n].legend(
                        loc="lower center",
                        bbox_to_anchor=(0.5, -0.45),
                        fontsize=settings.fontsize,
                    )
                fig.tight_layout()
                plt.show()
                # Save the figure if the directory is set and saving is enabled.
                if settings.fig_dir and self.save_fig:
                    self._save_figure(
                        fig,
                        f"{settings.fig_name}_slice_{varyingPlanes}plane_"
                        + f"fixedVal{self.color_thres.W_unit_to_N_unit(fixedRGB_val_scaled_k)}",
                    )
        return fig, axes

    # %% Visualize mahalanobis distance
    def plot_Mahalanobis_distance(
        self,
        resp,
        trial_ID,
        settings: PlotMahaSettings,
        sim_PMF=None,
        nTrials_bin=30,
        ax=None,
    ):
        """
        Plot binned Mahalanobis distances and responses by trial type.

        This method creates a visualization of psychometric data, including binned Mahalanobis
        distances and corresponding response rates. It also optionally overlays confidence
        intervals (68% and/or 95%), standard error bars, and simulated psychometric functions.

        Parameters:
        ----------
        resp : ndarray of shape (N,)
            Array of response values (e.g., binary or continuous) corresponding to each trial.
            The number of elements matches the length of `trial_ID`.
        trial_ID : list or array-like
            List of trial identifiers, where each entry corresponds to a trial type.
        nTrials_bin : int, optional, default=30
            Number of trials per bin for aggregating Mahalanobis distances and responses.
            If the total number of trials for any trial type is not divisible by `nTrials_bin`,
            a ValueError is raised during data preparation.
        ax : matplotlib.axes.Axes, optional
            Pre-existing matplotlib axes for plotting. If `None`, new axes are created.
        kwargs : dict
            Additional plotting parameters to customize the visualization.

        Plotting Parameters (kwargs):
        -----------------------------
        - 'visualize_68CI_95CI' : list of bool, default=[True, False]
            Whether to visualize 68% and/or 95% confidence intervals.
        - 'visualize_SE' : bool, default=True
            Whether to visualize standard error bars for a Bernoulli random variable.
        - 'sim_PMF' : dict, default=None
            Simulated psychometric function with keys 'x' (Mahalanobis distances) and 'y' (percent correct).

        Returns:
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes or list of Axes
            The axes objects used for the plot, either a single axis or a list of axes
            depending on the number of trial types.
        Maha_data : dict
            A dictionary containing the organized and binned data for each trial type:
            - 'trial_type': ndarray of shape (unique_trials,)
                Sorted array of unique trial types.
            - 'M_D_sorted': list of ndarrays
                Mahalanobis distances sorted for each trial type.
            - 'resp_sorted': list of ndarrays
                Responses sorted to align with sorted Mahalanobis distances for each trial type.
            - 'M_D_binned': list of ndarrays
                Binned mean Mahalanobis distances for each trial type.
            - 'resp_binned': list of ndarrays
                Binned mean responses for each trial type.
            - 'nTrials': list of ints
                Total number of trials for each trial type.

        """
        # Unique trial types
        trial_type_unsorted = np.unique(trial_ID)
        trial_type = np.sort(trial_type_unsorted)

        # Determine bin sizes for each trial type
        nTrials_bin_list = self._check_nTrials_bin(nTrials_bin, len(trial_type))
        print(nTrials_bin_list)

        Maha_data = self._org_Mahalanobis_distance_trial_ID(trial_ID, resp, nTrials_bin_list)

        ax, fig = self._initialize_axes(ax, Maha_data, settings)

        # Plot for each trial type
        CI_bounds_all = []
        for i in range(len(Maha_data["trial_type"])):
            nTrials_bin_i = nTrials_bin_list[i]
            # Define confidence interval bounds
            CI_bounds = {
                "68%": [
                    int(np.round(Maha_data["nTrials"][i] * 0.16)),
                    int(np.round(Maha_data["nTrials"][i] * 0.84)),
                ],
                "95%": [
                    int(np.round(Maha_data["nTrials"][i] * 0.025)),
                    int(np.round(Maha_data["nTrials"][i] * 0.975)),
                ],
            }
            key_list = list(CI_bounds.keys())
            CI_bounds_values = list(CI_bounds.values())
            # append
            CI_bounds_all.append(CI_bounds)
            for c, alpha, k, (lb_idx, ub_idx) in zip(list(range(2)), [0.4, 0.2], key_list, CI_bounds_values):  # noqa: B905
                if settings.visualize_68CI_95CI[c]:
                    x_CI_lb = (Maha_data["M_D_sorted"][i][lb_idx],)
                    x_CI_ub = Maha_data["M_D_sorted"][i][ub_idx]
                    ax[i].fill_betweenx(
                        [settings.ylim[0], settings.ylim[1]],
                        x_CI_lb,
                        x_CI_ub,
                        color=settings.CI_color,
                        alpha=alpha,
                        label=k + " confidence interval",
                        zorder=1,
                    )

            if sim_PMF is not None:
                ax[i].plot(
                    sim_PMF["x"],
                    sim_PMF["y"],
                    color=settings.linecolor,
                    lw=settings.PMF_lw,
                    label=settings.PMF_label,
                    zorder=3,
                )

            if settings.visualize_SE:
                for p in settings.p_SE:
                    ax[i].errorbar(
                        0,
                        p,
                        yerr=np.sqrt(p * (1 - p) / nTrials_bin_i),
                        capsize=settings.err_capsize,
                        capthick=settings.err_capthick,
                        lw=settings.err_lw,
                        color=settings.markercolor,
                        label=settings.SE_label,
                    )

            ax[i].scatter(
                Maha_data["M_D_binned"][i],
                Maha_data["resp_binned"][i],
                marker=settings.marker[i],
                color=settings.markercolor,
                edgecolor=settings.edgecolor,
                s=settings.markersize,
                label=settings.label[i],
                zorder=2,
            )

            ax[i].grid(True, color=[0.9, 0.9, 0.9], zorder=0)
            ax[i].set_xlabel(settings.xlabel)
            ax[i].set_ylabel(settings.ylabel)
            ax[i].tick_params(axis="both")  # Update tick size
            ax[i].set_yticks(settings.yticks)
            ax[i].set_xticks(settings.xticks)
            if settings.xlim is None:
                ax[i].set_xlim([-0.5, np.round(np.max(Maha_data["M_D_binned"][i])) + 1])
            else:
                ax[i].set_xlim(settings.xlim)
            ax[i].set_ylim(settings.ylim)
        if i == (len(Maha_data["trial_type"]) - 1):
            ax[i].legend(
                loc="lower center",
                bbox_to_anchor=(0.5, -0.85),  # Center it below the panel
                ncol=1,
            )  # Adjust the number of columns if needed
        fig.subplots_adjust(bottom=0.25, top=0.95, left=0.1, right=0.9)

        plt.tight_layout()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)

        # histogram
        if settings.visualize_hist:
            ax1, fig1 = self._initialize_axes(None, Maha_data, settings)
            # Plot for each trial type
            for i in range(len(Maha_data["trial_type"])):
                normalized_D_M = np.full(Maha_data["M_D_binned"][i].shape, np.nan)
                for n in range(len(Maha_data["resp_binned"][i])):
                    idx_match_D_M = np.argmin(np.abs(sim_PMF["x"] - Maha_data["M_D_binned"][i][n]))
                    match_p = sim_PMF["y"][idx_match_D_M]
                    normalized_D_M[n] = Maha_data["resp_binned"][i][n] - match_p
                # Plot data in the inset
                ax1[i].hist(
                    normalized_D_M,
                    bins=np.linspace(-0.2, 0.2, 30),
                    color=settings.markercolor,
                    alpha=0.3,
                    edgecolor=settings.markercolor,
                )
                ax1[i].set_yticks([])
                ax1[i].set_xticks(np.linspace(-0.2, 0.2, 5))
                ax1[i].tick_params(labelsize=settings.font * 2)  # this figure will be reduced
                # Remove specific spines
                # ax1[i].spines['top'].set_visible(False)
                # ax1[i].spines['left'].set_visible(False)
                # ax1[i].spines['right'].set_visible(False)
            plt.tight_layout()
            # Save the figure if `fig_name` is provided
            if settings.fig_dir and self.save_fig:
                self._save_figure(fig1, f"{settings.fig_name}_hist")

        print(CI_bounds)
        if settings.visualize_timecourse:
            ax2, fig2 = self._initialize_axes(None, Maha_data, settings)
            # Plot for each trial type
            for i in range(len(Maha_data["trial_type"])):
                # Plot data in the inset
                xx = np.arange(1, Maha_data["M_D_unsorted"][i].shape[0] + 1)
                xx_range = xx[-1] - xx[0]
                ax2[i].plot(xx, Maha_data["M_D_unsorted"][i], color=settings.markercolor, lw=0.2)
                key_list = list(CI_bounds_all[i].keys())
                CI_bounds_values = list(CI_bounds_all[i].values())

                for c, alpha, k, (lb_idx, ub_idx) in zip(list(range(2)), [0.9, 0.9], key_list, CI_bounds_values):  # noqa: B007, B905
                    if settings.visualize_68CI_95CI[c]:
                        x_CI_lb = Maha_data["M_D_sorted"][i][lb_idx]
                        x_CI_ub = Maha_data["M_D_sorted"][i][ub_idx]
                        ax2[i].fill_betweenx(
                            [x_CI_lb, x_CI_ub],
                            xx[0] - xx_range * 0.05,
                            xx[-1] + xx_range * 0.05,
                            color=settings.CI_color,
                            alpha=alpha,
                            zorder=0,
                        )
                ax2[i].set_yticks(np.linspace(0, 30, 4))
                ax2[i].set_xticks(np.linspace(0, xx[-1], 4))
                ax2[i].tick_params(labelsize=settings.fontsize * 2)
                ax2[i].set_xlim([xx[0] - xx_range * 0.05, xx[-1] + xx_range * 0.05])
                # Remove specific spines
                # ax1[i].spines['top'].set_visible(False)
                # ax1[i].spines['left'].set_visible(False)
                # ax1[i].spines['right'].set_visible(False)
            plt.tight_layout()
            # Save the figure if `fig_name` is provided
            if settings.fig_dir and self.save_fig:
                self._save_figure(fig2, f"{settings.fig_name}_timecourse")

        return fig, ax, Maha_data


# %%
def add_CI_ellipses(ell_min, ell_max, ax=None, cm="k", alpha=0.9, label=None, lw_outer=0, lw_inner=0):
    """
    Plots the confidence interval (CI) region of an ellipse on the given axis.

    The CI region is defined by:
    - ell_max: the outer contour (union of all bootstrapped ellipses)
    - ell_min: the inner contour (intersection of all bootstrapped ellipses)

    This method fills the area between the outer and inner contours, visually
    representing uncertainty in the fitted ellipse.

    Args:
        ell_min (ndarray): Inner contour of the CI (2 x N array of x and y coordinates).
        ell_max (ndarray): Outer contour of the CI (2 x N array of x and y coordinates).
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, assumes one is already in use.
        cm (str or color, optional): Color for the CI fill (default: 'k' for black).
        lbl (str, optional): Label for the outer region (useful for legend).
    """
    # Identify valid (non-NaN) points in the outer contour
    idx_max_nonan = ~np.isnan(ell_max[0])
    # Fill the outer contour region with solid color to indicate the CI boundary
    ax.fill(
        ell_max[0, idx_max_nonan],
        ell_max[1, idx_max_nonan],
        color=cm,
        alpha=alpha,
        lw=lw_outer,
        label=label,
    )

    # Identify valid (non-NaN) points in the inner contour
    idx_min_nonan = ~np.isnan(ell_min[0])
    # Fill the inner contour with white to "punch out" the inside, leaving a ring
    ax.fill(
        ell_min[0, idx_min_nonan],
        ell_min[1, idx_min_nonan],
        color="white",
        lw=lw_inner,
    )


def sort_plane_corners_ccw(corners_3x4):
    """
    Order 4 coplanar corners counter-clockwise in their plane.
    Accepts (3,4) or (4,3). Returns (3,4) to match your add_plane_quad.
    """
    P = np.asarray(corners_3x4, float)
    if P.shape == (3, 4):
        P = P.T
    assert P.shape == (4, 3), f"Expected (3,4) or (4,3), got {P.shape}"

    C = P.mean(axis=0)  # centroid
    Q = P - C  # center
    # Plane basis via SVD: first two right-singular vecs span the plane
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    B = Vt[:2].T  # (3x2)

    uv = Q @ B  # project to 2D
    ang = np.arctan2(uv[:, 1], uv[:, 0])
    order = np.argsort(ang)  # CCW order

    P_ccw = P[order]  # (4,3)
    return P_ccw.T  # back to (3,4)


def order_points_on_contour(pts, flag_wraparound=False):
    """
    Order an approximately planar 3D contour into a consistent loop.

    Parameters
    ----------
    pts : array-like, shape (N, 3)
        3D points that lie (approximately) in a common plane and are
        intended to form a single closed contour (possibly unsorted).
    flag_wraparound : bool, optional
        If True, the returned array has the first point appended to the
        end, i.e., shape (N+1, 3), so that pts[0] == pts[-1]. This is
        convenient for plotting closed loops. If False (default), the
        returned array has shape (N, 3) with no duplicate endpoint.

    Returns
    -------
    pts_ord : np.ndarray
        Ordered 3D points along the contour. Shape is:
        - (N, 3) if flag_wraparound is False
        - (N+1, 3) if flag_wraparound is True

    Notes
    -----
    - The ordering is determined by:
        1. Finding the best-fit plane via SVD (principal components).
        2. Projecting the points into a 2D coordinate system within
           that plane.
        3. Sorting by polar angle (atan2) in the 2D plane.
    - This is robust to arbitrary plane orientation in 3D; there is no
      assumption that the contour lies in a coordinate-aligned plane
      such as z = const.
    """
    pts = np.asarray(pts, float)
    center = pts.mean(axis=0)  # centroid of the point cloud

    # Use SVD to find a 2D orthonormal basis for the best-fit plane.
    # Vt rows are principal directions; the first two span the plane.
    _, _, Vt = np.linalg.svd(pts - center, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]  # in-plane basis vectors, shape (3,)

    # Project points into 2D plane coordinates (u, v).
    # Each point is expressed in the [e1, e2] basis.
    uv = (pts - center) @ np.stack([e1, e2], axis=1)  # (N, 2)

    # Compute polar angle in the plane and sort points by angle.
    angles = np.arctan2(uv[:, 1], uv[:, 0])
    order = np.argsort(angles)
    pts_ord = pts[order]

    # Optionally close the loop by appending the first point at the end.
    if flag_wraparound:
        return np.vstack([pts_ord, pts_ord[0][None, :]])
    else:
        return pts_ord


class WishartPredictionsVisualization_html:
    def __init__(self, settings: Plot3DPredHTMLSettings):
        """
        Additive plotter: call add_* methods to compose a figure, then apply layout.
        All tunables live in Plot3DPredHTMLSettings.
        """
        self.st = settings
        # Configure lighting upfront (keeps add_* methods simple)
        if self.st.flag_lighting:
            self.lighting = dict(
                ambient=self.st.light_ambient,
                diffuse=self.st.light_diffuse,
                specular=self.st.light_specular,
                roughness=self.st.light_roughness,
                fresnel=self.st.light_fresnel,
            )
            lx, ly, lz = self.st.light_position
            self.light_position = dict(x=lx, y=ly, z=lz)
        else:
            self.lighting = None
            self.light_position = None

    @staticmethod
    def to_rgb_str(rgb_like):
        if isinstance(rgb_like, str):
            return rgb_like
        c = np.asarray(rgb_like, float).ravel()
        if c.max() <= 1.0:  # support 0..1 inputs
            c = np.clip(c, 0, 1) * 255.0
        r, g, b = np.clip(c, 0, 255).astype(int)[:3]
        return f"rgb({r}, {g}, {b})"

    def add_plane_quad(self, fig, corners_3x4):
        """
        Add a filled quadrilateral plane (two triangles) + its outline.
        The input is 4 coplanar corners (3x4 or 4x3); order is auto-sorted CCW.
        """
        corners_3x4_sorted = sort_plane_corners_ccw(corners_3x4)
        color = self.to_rgb_str(self.st.isoluminant_plane_color)
        edge = self.to_rgb_str(self.st.isoluminant_edge_color)
        alpha = self.st.isoluminant_plane_alpha
        P = np.asarray(corners_3x4_sorted, float).T  # (4,3), assume order 0-1-2-3 around the loop

        # Fill (two triangles) using vertex indices (i, j, k)
        fig.add_trace(
            go.Mesh3d(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],  # two triangles: (0,1,2) and (0,2,3)
                color=color,
                opacity=alpha,
                flatshading=True,
                showscale=False,
                hoverinfo="skip",
            )
        )

        # Outline
        loop = np.vstack([P, P[:1]])
        fig.add_trace(
            go.Scatter3d(
                x=loop[:, 0],
                y=loop[:, 1],
                z=loop[:, 2],
                mode="lines",
                line=dict(color=edge, width=self.st.isoluminant_plane_lw),
                hoverinfo=self.st.plane_hover_info,
                showlegend=False,
            )
        )
        return fig

    def add_reference_point(self, fig, xyz, marker=None, color=None):
        """
        Add a single 3D reference point to the figure.

        """
        xyz = np.asarray(xyz, float).ravel()
        if xyz.size != 3:
            raise ValueError(f"xyz must have 3 elements (X, Y, Z), got shape {xyz.shape}")

        # Defaults from settings if available, otherwise hard-coded
        if color is None:
            default_color = getattr(self.st, "ref_point_color", "black")
            color = self.to_rgb_str(default_color)
        else:
            color = self.to_rgb_str(color)

        fig.add_trace(
            go.Scatter3d(
                x=[xyz[0]],
                y=[xyz[1]],
                z=[xyz[2]],
                mode="markers",
                marker=dict(
                    size=self.st.ref_size,
                    color=color,
                    symbol=marker or self.st.ref_marker,  # use string as symbol
                ),
                showlegend=False,
                hoverinfo=self.st.ref_hover_info,
            )
        )
        return fig

    def add_sliced_ellipses(self, fig, sliced_ell_byPlane, color=None):
        """
        Add 3D polylines for ellipse slices (each entry is iterable of (x, y, z)).
        """
        if color is None:
            color = self.to_rgb_str(self.st.sEll_line_color)
        else:
            color = self.to_rgb_str(color)

        for xe, ye, ze in sliced_ell_byPlane:
            fig.add_trace(
                go.Scatter3d(
                    x=xe,
                    y=ye,
                    z=ze,
                    mode="lines",
                    line=dict(width=self.st.sEll_line_width, color=color),
                    showlegend=False,
                    hoverinfo=self.st.sEll_hover_info,
                )
            )
        return fig

    def add_geodesic_paths(self, fig, geodesic_paths):
        """
        Add 3D geodesic paths.
        """
        # Convert configured color to CSS rgb()
        color = self.to_rgb_str(self.st.geo_path_color)
        N = len(geodesic_paths)
        for m in range(N):
            path = np.asarray(geodesic_paths[m].T)
            # handle either (3, T) or (T, 3)
            if path.ndim == 2 and path.shape[0] == 3:
                x, y, z = path[0], path[1], path[2]
            else:
                x, y, z = path[:, 0], path[:, 1], path[:, 2]
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(width=self.st.geo_path_lw, color=color),
                    showlegend=False,
                    hoverinfo=self.st.geo_hover_info,
                )
            )
        return fig

    def add_ellipsoid_surface(self, fig, X, Y, Z, color_str):
        """Internal helper: add a single ellipsoid surface to `fig`."""
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                showscale=False,  # hide colorbar
                opacity=self.st.ell_alpha,
                colorscale=[[0, color_str], [1, color_str]],
                lighting=self.lighting,
                lightposition=self.light_position,
                hoverinfo=self.st.ell_hover_info,
            )
        )
        return fig

    def plot_ellipsoids_mesh(self, fig, model_pred, center_rgb=None):
        """
        Add 3D ellipsoid surfaces to a Plotly figure.

        Expects `self.model_pred.params_ell` to be a 3D nested indexable structure:
            params_ell[i][j][k] -> dict with keys:
                - "radii": (3,) principal radii
                - "evecs": (3,3) columns are principal axes (orthonormal)
                - "center": (3,) ellipsoid center in W-unit coordinates

        For each ellipsoid, this:
          - maps its center from W-unit to display color space,
          - builds a smooth (X,Y,Z) surface via spherical parameterization,
          - plots a `go.Surface` with fixed color (no colorbar), lighting, and opacity.
        """
        # Pull the parameter grid
        P = model_pred.params_ell  # 3D grid-like: P[i][j][k]
        n1, n2, n3 = len(P), len(P[0]), len(P[0][0])  # grid extents

        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    p = P[i][j][k]

                    # Unpack parameters with explicit dtypes/shapes
                    radii = np.asarray(p["radii"], float)  # (3,)
                    evecs = np.asarray(p["evecs"], float)  # (3,3)
                    center = np.asarray(p["center"], float).ravel()  # (3,)

                    # Use provided center_rgb for color if given; otherwise use this ellipsoid's center
                    if center_rgb is not None:
                        center_for_color = center_rgb[i, j, k]
                    else:
                        center_for_color = color_thresholds.W_unit_to_N_unit(np.asarray(center_rgb, float).ravel())

                    color_str = self.to_rgb_str(center_for_color)

                    # Build surface mesh
                    X, Y, Z = EllipsoidSurfaceMesh(
                        radii=self.st.ell_scaler * radii,
                        center=center,
                        eigenVectors=evecs,
                        nu=self.st.ell_mesh_nu,
                        nv=self.st.ell_mesh_nv,
                    )

                    # use the internal helper
                    self.add_ellipsoid_surface(fig, X, Y, Z, color_str)
        return fig

    def plot_ellipsoids_mesh_cov(self, fig, grid, cov_grid):
        base_shape = grid.shape[:3]
        for ijk in np.ndindex(*base_shape):
            center = grid[ijk]
            cov_ijk = cov_grid[ijk]
            eigenvalues, evecs, radii, *_ = covMat_to_ellParamsQ(cov_ijk)

            # Map center (W-unit) -> display color (e.g., normalized RGB) -> CSS rgb()
            center_rgb01 = color_thresholds.W_unit_to_N_unit(center)  # ensure returns in [0,1]
            color_str = self.to_rgb_str(center_rgb01)

            X, Y, Z = EllipsoidSurfaceMesh(
                radii=self.st.ell_scaler * radii,
                center=center,
                eigenVectors=evecs,
                nu=self.st.ell_mesh_nu,
                nv=self.st.ell_mesh_nv,
            )

            # use the internal helper
            self.add_ellipsoid_surface(fig, X, Y, Z, color_str)
        return fig

    def apply_3d_layout(self, fig):
        fs = int(self.st.font_size)

        axis_spec = dict(
            tickmode="array",
            tickvals=list(self.st.ticks),
            tickformat=".2f",
            range=self.st.lim,
            showspikes=False,
            # tick label font
            tickfont=dict(size=fs),
            # axis title font (new + legacy, for robustness)
            title=dict(font=dict(size=fs)),
            titlefont=dict(size=fs),
        )

        fig.update_layout(
            font=dict(family="Arial", size=fs),
            hovermode=False if self.st.disable_hover else "closest",
            scene=dict(
                aspectmode="cube",
                xaxis=dict(
                    axis_spec,
                    title=dict(text=self.st.xlabel, font=dict(size=fs)),
                    titlefont=dict(size=fs),
                ),
                yaxis=dict(
                    axis_spec,
                    title=dict(text=self.st.ylabel, font=dict(size=fs)),
                    titlefont=dict(size=fs),
                ),
                zaxis=dict(
                    axis_spec,
                    title=dict(text=self.st.zlabel, font=dict(size=fs)),
                    titlefont=dict(size=fs),
                ),
            ),
            scene_camera=dict(
                eye=dict(
                    x=self.st.camera_eye[0],
                    y=self.st.camera_eye[1],
                    z=self.st.camera_eye[2],
                ),
                center=dict(
                    x=self.st.camera_center[0],
                    y=self.st.camera_center[1],
                    z=self.st.camera_center[2],
                ),
                up=dict(
                    x=self.st.camera_up[0],
                    y=self.st.camera_up[1],
                    z=self.st.camera_up[2],
                ),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig

    # optional
    def add_dashed_line3d(self, fig, p0, p1, n_dashes=50, color="#111", width=5, hover=False):
        """
        Add a dashed 3D line between points p0 and p1.

        """
        p0 = np.asarray(p0, dtype=float).ravel()
        p1 = np.asarray(p1, dtype=float).ravel()
        assert p0.shape == (3,) and p1.shape == (3,), "p0 and p1 must be 3D points"

        # Parameterize the segment and interleave None to break into dashes
        t = np.linspace(0.0, 1.0, 2 * n_dashes + 1)  # endpoints for dash+gap pairs
        P = (1 - t)[:, None] * p0 + t[:, None] * p1  # (2*n_dashes+1, 3)

        xs, ys, zs = [], [], []
        for i in range(0, 2 * n_dashes, 2):
            xs += [P[i, 0], P[i + 1, 0], None]
            ys += [P[i, 1], P[i + 1, 1], None]
            zs += [P[i, 2], P[i + 1, 2], None]

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="all" if hover else "skip",
                showlegend=False,
            )
        )
        return fig


# %%
class WishartPredictionsGeoVisualization(PlottingTools):
    """
    Visualize geodesic-path predictions under the Wishart model.

    Responsibilities
    - Store reference / comparator stimuli and color mapping.
    - Provide helpers to format titles and pad axis limits to square ranges.
    - Plot 2D geodesic paths, samples (z0/z1/z2), and internal-noise ellipses.

    Parameters
    ----------
    ref_W : array-like (D,)
        Reference stimulus in model space.
    comp1_W : array-like (D,)
        Comparator #1 stimulus in model space.
    comp2_W_varying : array-like (N_levels, D)
        Comparator #2 levels in model space.
    color_thres : object
        Color mapping utilities (e.g., provides M_2DWToRGB, W_unit_to_N_unit).

    """

    def __init__(
        self,
        ref_W,
        comp1_W,
        comp2_W_varying,
        settings: PlotSettingsBase,
        color_thres=None,
        save_fig=False,
        save_format="png",
    ):
        """ """
        super().__init__(settings, save_fig, save_format)
        self.ref_W = ref_W
        self.ndims = ref_W.shape[0]
        self.comp1_W = comp1_W
        self.comp2_W_varying = comp2_W_varying
        self.nLevels_comp2 = comp2_W_varying.shape[0]
        self.color_thres = color_thres

    def _use_refW_as_title(self):
        """
        Return a compact, human-readable string for the reference coordinates,
        e.g., "0.2, -1.1, 3"
        """
        vals = np.asarray(self.ref_W).ravel()  # works for list/np/jax arrays
        return ", ".join((f"{v:.2f}").rstrip("0").rstrip(".") for v in vals)

    def _get_colormap(self, idx_comp2):
        """
        Return a 3×3 RGB colormap (rows: ref, comp1, comp2) for plotting.

        Parameters
        ----------
        stim_stack : ndarray
            Stimulus stack used for color mapping.
            - If ndims == 2: expected shape (3, 3) with homogeneous coords per column
              (i.e., each column is [x, y, 1]^T) for the 2D→RGB linear map.
            - If ndims == 3: expected shape (3, 3) with rows (or columns, per your
              color_thres API) representing 3D model-space coordinates in [-1, 1].

        Returns
        -------
        cmap3 : ndarray or list
            Shape (3, 3) RGB values in [0, 1] for [ref, comp1, comp2].
            If no color_thres is provided, falls back to 3 colors from Pastel1.
        """
        stims = [s for s in (self.ref_W, self.comp1_W, self.comp2_W_varying[idx_comp2]) if s is not None]
        k = len(stims)

        if self.color_thres is not None:
            if self.ndims == 2:
                # homogeneous coords per stimulus: columns [x, y, 1]^T
                stim_stack = np.c_[np.vstack(stims), np.ones((k, 1))].T  # (3, k)
                cmap3 = (self.color_thres.M_2DWToRGB @ stim_stack).T  # (k, 3)
            else:
                stim_stack = np.vstack(stims)  # (k, 3)
                cmap3 = self.color_thres.W_unit_to_N_unit(stim_stack)  # (k, 3)
        else:
            cmap = matplotlib.colormaps["Pastel1"]
            # get first k RGB rows
            cmap3 = np.array(cmap.colors[:k], dtype=float)  # (k, 3)

        return cmap3

    def _pad_limits_to_square(self, lims, target_range=None):
        """
        Pad axis limits so all dimensions have the same range.

        Parameters
        ----------
        lims : array-like of shape (D, 2)
            Per-dimension [min, max]. Example for 2D: [[xmin, xmax], [ymin, ymax]].
        target_range : float or None
            If None (default), use the maximum existing range as the target.
            If a float, force all dimensions to this range.

        Returns
        -------
        new_lims : (D, 2) ndarray
            Padded limits with equal ranges.
        lengths : (D,) ndarray
            Original per-dimension ranges.
        diffs : (D,) ndarray
            Target range minus original range (the amount added per dimension).
        pads : (D,) ndarray
            Amount padded to each side (diffs / 2).
        """
        lims = np.asarray(lims, dtype=float)
        if lims.ndim != 2 or lims.shape[1] != 2:
            raise ValueError("lims must be shape (D, 2) with [min, max] per dimension.")

        # (1) absolute length per dimension
        lengths = lims[:, 1] - lims[:, 0]
        if np.any(lengths < 0):
            raise ValueError("Each [min, max] must satisfy min <= max.")

        # choose target range
        if target_range is None:
            target = np.max(lengths)  # largest existing range
        else:
            target = float(target_range)
            if target < 0:
                raise ValueError("target_range must be non-negative.")

        # (2) per-dimension difference to target
        diffs = target - lengths

        # (3) pad half the difference to each side where needed
        pads = np.clip(diffs, 0, None) / 2.0

        # (4) build new limits
        new_lims = np.column_stack((lims[:, 0] - pads, lims[:, 1] + pads))
        return new_lims, lengths, diffs, pads

    def _scatter(self, ax, Z, color, label, settings):
        """Scatter a cloud of sample points.

        Parameters
        ----------
        ax : matplotlib Axes (2D or 3D)
            Target axes to draw on.
        Z : ndarray, shape (N, D)
            Sample coordinates (D=2 or 3). Each row is a point.
        color : array-like or tuple
            RGB(A) color to use for all points.
        label : str
            Legend label for this group.
        settings : PlotMCGeoSettings
            Provides marker size (samples_size) and alpha (samples_alpha).
        """
        ax.scatter(
            *Z.T,
            c=color,
            s=settings.samples_size,
            label=label,
            alpha=settings.samples_alpha,
        )

    def _plot_paths(self, ax, paths, settings):
        """Plot a batch of geodesic paths (one line per sample).

        Parameters
        ----------
        ax : matplotlib Axes (2D or 3D)
            Target axes to draw on.
        paths : ndarray, shape (num_samples, T, D)
            Polyline paths for each sample (T points, D=2 or 3).

        """
        for path in paths:  # iterate over samples
            ax.plot(
                *path.T,
                color=settings.path_lc,
                lw=settings.path_lw,
                alpha=settings.path_alpha,
            )

    def _plot_surface3d(self, ax, surf, color, settings):
        """Render a 3D surface (ellipsoid patch) on 3D axes.

        Parameters
        ----------
        ax : matplotlib 3D Axes
            Target axes to draw on (projection='3d').
        surf : ndarray, shape (3, M, N)
            Surface grids as (X, Y, Z).
        color : array-like or tuple
            RGB(A) face color for the surface.

        """
        X, Y, Z = surf  # expect shape (3, M, N)
        ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            color=color,
            edgecolor="none",
            antialiased=True,
            alpha=settings.surf_alpha,
        )

    def _compute_fig_bounds(self, z0, z2, z1=None):
        """Compute per-axis bounds from representative levels.

        Uses z0[0], z2[-1], z2[0] and optionally z1[0] to set global plot limits.

        Parameters
        ----------
        z0 : ndarray, shape (N_levels, N_samples, D)
            Samples around the reference.
        z2 : ndarray, shape (N_levels, N_samples, D)
            Samples for comp2 (varying by level).
        z1 : ndarray or None, shape (N_levels, N_samples, D), optional
            Samples for comp1. If None, comp1 is ignored.

        Returns
        -------
        (x_bds, y_bds) or (x_bds, y_bds, z_bds)
            Each as [min, max] rounded to 2 decimals. Returns 2D or 3D
            depending on self.ndims.
        """

        zb = [z0[0], z2[-1], z2[0]]
        if z1 is not None:
            zb.append(z1[0])
        zbd = np.vstack(zb)
        x_bds = np.round([zbd[:, 0].min(), zbd[:, 0].max()], 2)
        y_bds = np.round([zbd[:, 1].min(), zbd[:, 1].max()], 2)
        if self.ndims == 3:
            z_bds = np.round([zbd[:, 2].min(), zbd[:, 2].max()], 2)
            return x_bds, y_bds, z_bds
        else:
            return x_bds, y_bds

    def _plot_2D_predicted_geodesic_paths(
        self,
        z0,
        z2,
        paths_z0z2,
        ell,
        settings: PlotMCGeoSettings,
        z1=None,
        paths_z0z1=None,
    ):
        """
        Plot 2D geodesic paths, internal-noise ellipses, and sample clouds for each
        comp#2 level. Supports optional comp#1 inputs (z1/paths_z0z1); if omitted,
        related elements are skipped.

        Parameters
        ----------
        z0 : ndarray, shape (N_levels, N_samples, 2)
            Noisy draws around the reference stimulus in model space.
        z2 : ndarray, shape (N_levels, N_samples, 2)
            Noisy draws around comp#2 (varies with level) in model space.
        paths_z0z2 : ndarray, shape (N_levels, N_samples, T, 2)
            Geodesic paths from z0 to z2 for each sample (T waypoints).
        ell : ndarray, shape (N_stim, 2, nTheta)
            Ellipse traces for internal noise for each plotted stimulus
            (ref, optional comp#1, and each comp#2 level).
        settings : PlotMCGeoSettings
            Plot styling/config (sizes, colors, alphas, labels, legend options, etc.).
        z1 : ndarray, shape (N_levels, N_samples, 2), optional
            Noisy draws around comp#1 in model space. If None, comp#1 elements
            (ellipse, samples, paths) are omitted.
        paths_z0z1 : ndarray, shape (N_levels, N_samples, T, 2), optional
            Geodesic paths from z0 to z1 for each sample. Used only if `z1` is provided.

        """

        # Global bounds (computed once); note this mixes levels (0, last).
        # If you prefer per-level bounds, compute inside the loop from z0[a], z1[a], z2[a].
        x_bds, y_bds = self._compute_fig_bounds(z0, z2, z1=z1)

        for a in range(self.nLevels_comp2):
            cmap3 = self._get_colormap(a)

            fig, ax = plt.subplots(figsize=settings.fig_size)
            # Internal-noise ellipsoids: [ref, comp1, comp2_a]
            # ell has shape (N_stim, 3, nu, nv); here we take surface grids.
            if z1 is not None:
                ell_idx = [0, 1, a + 2]
                z_all = [z0[a], z1[a], z2[a]]
            else:
                ell_idx = [0, a + 1]
                z_all = [z0[a], z2[a]]
            for b in range(len(ell_idx)):
                # ellipses
                ax.plot(*ell[ell_idx[b]], c=np.clip(cmap3[b], 0, 1))
                # Sample clouds (z0, z1, z2) for the current level
                self._scatter(ax, z_all[b], cmap3[b], settings.samples_labels[b], settings)

            # Geodesic paths for all draws: z0→z1 and z0→z2
            if z1 is not None:
                self._plot_paths(ax, paths_z0z1[a], settings)
            self._plot_paths(ax, paths_z0z2[a], settings)

            # Make axes square in data units by padding the shorter range
            new_lims, _, _, _ = self._pad_limits_to_square(np.vstack((x_bds, y_bds)))
            ax.set_xlim(*new_lims[0])
            ax.set_ylim(*new_lims[1])
            # 4 major ticks on each axis
            ax.set_xticks(np.linspace(*new_lims[0], 4))
            ax.set_yticks(np.linspace(*new_lims[1], 4))
            # Square axes box (for a square figure, also set figsize accordingly)
            ax.set_box_aspect(1)

            # Format tick labels to 2 decimals
            fmt = mticker.FormatStrFormatter("%.2f")
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            # Light grid and axis labels
            ax.grid(True, which="both", alpha=0.1)  # add grid
            ax.set_xlabel("Model space dimension 1")
            ax.set_ylabel("Model space dimension 2")

            # Optional panel title showing the reference coordinates
            if settings.flag_show_cond_title:
                title_vals = self._use_refW_as_title()
                ax.set_title(f"Ref = [{title_vals}]", fontsize=settings.fontsize)

            # Legend (only if enabled in settings)
            if not settings.legend_off:
                plt.legend(
                    loc=settings.legend_loc,
                    bbox_to_anchor=settings.anchor_legend_box,
                    fontsize=settings.fontsize - 1,
                )
            fig.tight_layout()
            # Save the figure if the directory is set and saving is enabled.
            if settings.fig_dir and self.save_fig:
                self._save_figure(fig, settings.fig_name + f"_comp2_idx{a:02}")

    def _plot_3D_predicted_geodesic_paths(
        self,
        z0,
        z2,
        paths_z0z2,
        ell,
        settings: PlotMCGeoSettings,
        z1=None,
        paths_z0z1=None,
    ):
        """
        Plot 3D geodesic paths, internal-noise ellipsoids, and sample clouds for each
        comp#2 level. Uses a padded cube so x/y/z share the same data range (unit aspect).
        Supports optional comp#1 inputs (z1/paths_z0z1); if omitted, related elements are skipped.

        Parameters
        ----------
        z0 : ndarray, shape (N_levels, N_samples, 3)
            Noisy draws around the reference stimulus in 3D model space.
        z2 : ndarray, shape (N_levels, N_samples, 3)
            Noisy draws around comp#2 (varies with level) in 3D model space.
        paths_z0z2 : ndarray, shape (N_levels, N_samples, T, 3)
            Geodesic paths from z0 to z2 for each sample (T waypoints per path).
        ell : ndarray, shape (N_stim, 3, M, N)
            Surface grids (X, Y, Z) for internal-noise ellipsoids for each plotted
            stimulus (ref, optional comp#1, and each comp#2 level).
        settings : PlotMCGeoSettings
            Plot configuration (sizes, colors, alphas, labels, legend options, save path, etc.).
        z1 : ndarray, shape (N_levels, N_samples, 3), optional
            Noisy draws around comp#1 in 3D model space. If None, comp#1 ellipsoid,
            samples, and z0→z1 paths are omitted.
        paths_z0z1 : ndarray, shape (N_levels, N_samples, T, 3), optional
            Geodesic paths from z0 to z1 for each sample. Used only if `z1` is provided.

        """

        # Global bounds (computed once from a mix of levels).
        x_bds, y_bds, z_bds = self._compute_fig_bounds(z0, z2, z1=z1)

        for a in range(self.nLevels_comp2):
            # Colors for ref / comp1 / current comp2 level in 3D
            cmap3 = self._get_colormap(a)

            # 3D figure
            fig = plt.figure(figsize=settings.fig_size)
            ax = fig.add_subplot(111, projection="3d")

            # Internal-noise ellipsoids: [ref, comp1, comp2_a]
            # ell has shape (N_stim, 3, nu, nv); here we take surface grids.
            if z1 is not None:
                ell_idx = [0, 1, a + 2]
                z_all = [z0[a], z1[a], z2[a]]
            else:
                ell_idx = [0, a + 1]
                z_all = [z0[a], z2[a]]
            for b in range(len(ell_idx)):
                # ellipsoid
                self._plot_surface3d(ax, ell[ell_idx[b]], cmap3[b], settings)
                # Sample clouds (z0, z1, z2) for the current level
                self._scatter(ax, z_all[b], cmap3[b], settings.samples_labels[b], settings)

            # Geodesic paths for all draws: z0→z1 and z0→z2
            if z1 is not None:
                self._plot_paths(ax, paths_z0z1[a], settings)
            self._plot_paths(ax, paths_z0z2[a], settings)

            # Pad limits so all three axes share the same range (cube)
            # and place 4 major ticks on each axis.
            new_lims, _, _, _ = self._pad_limits_to_square(np.vstack((x_bds, y_bds, z_bds)))
            ax.set_xlim(*new_lims[0])
            ax.set_ylim(*new_lims[1])
            ax.set_zlim(*new_lims[2])
            ax.set_xticks(np.linspace(*new_lims[0], 4))
            ax.set_yticks(np.linspace(*new_lims[1], 4))
            ax.set_zticks(np.linspace(*new_lims[2], 4))
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel("Model space dimension 1")
            ax.set_ylabel("Model space dimension 2")
            ax.set_zlabel("Model space dimension 3")

            # Styling: lighter grid lines and semi-transparent panes
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis._axinfo["grid"]["linewidth"] = 0.3  # thinner grid
                axis._axinfo["grid"]["color"] = (0, 0, 0, 0.25)  # add some transparency
            ax.xaxis.set_pane_color((1, 1, 1, 0.1))
            ax.yaxis.set_pane_color((1, 1, 1, 0.1))
            ax.zaxis.set_pane_color((1, 1, 1, 0.1))

            # Tick label formatting (two decimals)
            fmt = mticker.FormatStrFormatter("%.2f")
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            ax.zaxis.set_major_formatter(fmt)

            # Optional panel title showing reference coordinates
            if settings.flag_show_cond_title:
                title_vals = self._use_refW_as_title()
                ax.set_title(f"Ref = [{title_vals}]", fontsize=settings.fontsize)

            # Legend (inside or outside depending on settings)
            if not settings.legend_off:
                plt.legend(
                    loc=settings.legend_loc,
                    bbox_to_anchor=settings.anchor_legend_box,
                    borderaxespad=0.0,
                    fontsize=settings.fontsize - 1,
                )

            # Save one figure per comp#2 level (zero-padded index)
            if settings.fig_dir and self.save_fig:
                self._save_figure(fig, settings.fig_name + f"_comp2_idx{a:02}")

    def plot_predicted_geodesic_paths(
        self,
        z0,
        z2,
        paths_z0z2,
        ell,
        settings: PlotMCGeoSettings,
        z1=None,
        paths_z0z1=None,
    ):
        """
        Front-end dispatcher that plots predicted geodesic paths for either 2D or 3D
        model spaces, based on `self.ndims`. Optional comp#1 inputs (z1/paths_z0z1)
        are passed through and skipped downstream if None.

        Parameters
        ----------
        z0 : ndarray, shape (N_levels, N_samples, D)
            Noisy draws around the reference stimulus (D=2 or 3).
        z2 : ndarray, shape (N_levels, N_samples, D)
            Noisy draws around comp#2 (varies with level).
        paths_z0z2 : ndarray, shape (N_levels, N_samples, T, D)
            Geodesic paths from z0 to z2 (T waypoints).
        ell : ndarray
            Internal-noise geometry:
              - 2D: (N_stim, 2, nTheta) ellipse traces
              - 3D: (N_stim, 3, M, N) surface grids (X, Y, Z)
            for [ref, (optional comp#1), each comp#2 level].
        settings : PlotMCGeoSettings
            Styling/config (sizes, colors, alphas, labels, legend, save path, etc.).
        z1 : ndarray, optional, shape (N_levels, N_samples, D)
            Noisy draws around comp#1. If None, comp#1 plots are omitted.
        paths_z0z1 : ndarray, optional, shape (N_levels, N_samples, T, D)
            Geodesic paths z0→z1. Used only if `z1` is provided.

        """
        if self.ndims == 2:
            self._plot_2D_predicted_geodesic_paths(z0, z2, paths_z0z2, ell, settings, z1, paths_z0z1)
        elif self.ndims == 3:
            self._plot_3D_predicted_geodesic_paths(z0, z2, paths_z0z2, ell, settings, z1, paths_z0z1)
        else:
            raise ValueError(f"Unidentified number of dimensions: {self.ndims}")

    def plot_diff_geodist(self, dists_z0z2, dists_z0z1, settings: PlotMCGeoSettings):
        diff_all = np.asarray(dists_z0z2) - np.asarray(dists_z0z1)
        diff_bds = [np.min(diff_all) - 1, np.max(diff_all) + 1]

        for a in range(self.nLevels_comp2):
            # Compute difference (handles JAX arrays too)
            diff = np.asarray(dists_z0z2[a]) - np.asarray(dists_z0z1[a])
            pComp2 = np.mean(diff > 0)

            fig, ax = plt.subplots(figsize=(5, 3))
            edges = np.linspace(*diff_bds, settings.diff_nBins)
            ax.hist(diff, bins=edges, color="grey", edgecolor="white")
            ax.axvline(0, linestyle="--", linewidth=1, color="k", alpha=0.8, label="0")
            ax.set_xlabel("dists_z0z2 - dists_z0z1")
            ax.set_ylabel("Count")
            if settings.flag_show_cond_title:
                title_vals = self._use_refW_as_title()
                ax.set_title(
                    f"Ref = [{title_vals}], index of Comp#2: {a}, pComp2 = {pComp2:.2f}"
                    + "\nGeodesic distance differences",
                    fontsize=settings.fontsize,
                )
            ax.grid(True, alpha=0.3)
            ax.set_xlim(diff_bds)
            num_samples = dists_z0z2.shape[1]
            ax.set_ylim([0, num_samples // 10])
            fmt = mticker.FormatStrFormatter("%d")
            ax.xaxis.set_major_formatter(fmt)
            fig.tight_layout()
            # Save the figure if the directory is set and saving is enabled.
            if settings.fig_dir and self.save_fig:
                self._save_figure(fig, "Hist_" + settings.fig_name + f"_comp2_idx{a:02}")
