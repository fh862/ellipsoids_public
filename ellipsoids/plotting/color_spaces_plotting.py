#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 14:39:57 2025

@author: fangfang
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple, Union
import sys
import os
script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from plotting.wishart_plotting import PlottingTools, PlotSettingsBase

#%%    
@dataclass
class PlotGegenfurtner(PlotSettingsBase):
    fig_size: Tuple[float, float] = (10.5, 7.5)
    fig_nrows: int = 2
    fig_ncols: int = 3
    width_ratios: List[float] = field(default_factory=lambda: [1, 1.2, 1])
    height_ratios: List[float] = field(default_factory=lambda: [1, 1])
    G_ax_ub: int = 10
    ref_lc: Union[str, np.ndarray, List[float]] = 'k'  # 'k' or RGB triplet
    ref_marker: str = '+'
    ref_ms: float = 50
    ref_lw: float = 1
    modelpred_lw: float = 3 
    grid_lw: float = 0.1
    ell_fullgrid_lw: float = 1  # Add type annotation
    xticks_DKL: List[float] = field(default_factory=lambda: [-0.03, 0.03])
    xticks_W: Union[List[float], np.ndarray] = field(default_factory=lambda: np.linspace(-0.7, 0.7, 3))
    xlabel_W: str = 'Model space dimension 1'
    ylabel_W: str = 'Model space dimension 2'
    xlabel_DKL: str = 'DKL L-M'
    ylabel_DKL: str = 'DKL S'
    xlabel_sDKL: str = 'DKL L-M (stretched)'
    ylabel_sDKL: str = 'DKL S (stretched)'
    fig_name: str = 'Comp_gegenfurtnerEllipses'

#%%
class DKLRelatedSpacesVisualization(PlottingTools):
    def __init__(self, modelpred, Gcomp, color_thres,
                 settings: PlotSettingsBase, save_fig=False, save_format = 'pdf'):
        """
        Initialize the DKLRelatedSpacesVisualization class.

        This class is used to visualize color discrimination thresholds in different 
        color spaces (e.g., Wishart model space, DKL space, stretched DKL space), 
        specifically for comparing thresholds at reference locations sampled in 
        Gegenfurtner's study.

        Parameters
        ----------
        modelpred : object
            An instance containing the predicted threshold ellipses over a grid 
            of reference stimuli (e.g., Wishart model predictions).
        
        Gcomp : dict
            A dictionary containing precomputed data for Gegenfurtner’s reference 
            stimuli and associated ellipse contours in various color spaces.
        
        color_thres : object
            Contains transformation matrices (e.g., for converting model space to RGB) 
            and possibly other threshold-related data.
        
        settings : PlotSettingsBase
            A dataclass instance specifying figure appearance settings (e.g., size, 
            font, tick configuration).
        
        save_fig : bool, optional
            Whether to save the generated plots. The default is False.
        
        save_format : str, optional
            Format for saving figures (e.g., 'pdf', 'png'). Default is 'pdf'.
        """
        super().__init__(settings, save_fig, save_format)
        self.modelpred = modelpred  # predicted threshold ellipses across a grid
        self.Gcomp = Gcomp          # precomputed ellipses and references from Gegenfurtner’s data
        self.color_thres = color_thres  # contains color transformation matrices
        self.Grefs = Gcomp['ref_pts_W'].shape[1]  # number of reference stimuli used in Gegenfurtner’s data
        self._construct_cmap_Grefs()  # generate color mapping for each reference based on model space
        
    def _construct_cmap_Grefs(self):
        """
        Construct an RGB colormap for the reference stimuli used in Gegenfurtner's paper.
    
        The colormap is computed by projecting the reference points (in model space)
        through the color transformation matrix `M_2DWToRGB`.
    
        """
        # Convert each reference point from model space to RGB using the transformation matrix
        cmap = self.color_thres.M_2DWToRGB @ self.Gcomp['ref_pts_W']
        self.cmap_Grefs = cmap.T
        
    def _construct_cmap_Wrefs(self, grid_W):
        """
        Construct an RGB colormap for a 2D grid of reference points in model (Wishart) space.
    
        Each grid point is converted to RGB space by applying the transformation matrix 
        `M_2DWToRGB` to homogeneous coordinates [x, y, 1].
    
        Parameters
        ----------
        grid_W : np.ndarray, shape (N, N, 2)
            2D grid of reference points in model space.

        """
        grid_W_r = grid_W.reshape(-1,2)
        
        self.cmap_Wrefs = self.color_thres.W2D_to_rgb(grid_W_r) 
        
    @staticmethod
    def set_square_axis(ax, ub, ticks):
        ax.set_aspect('equal', adjustable='box')  # Make the axis square
        ax.set_xlim([-ub, ub])
        ax.set_ylim([-ub, ub])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
    @staticmethod
    def draw_square_box(ax, ub):
        """
        Draw a square box centered at the origin on the given axis.
    
        """
        vertices = [(-ub, -ub), (ub, -ub), (ub, ub), (-ub, ub)]
        box = Polygon(vertices, closed=True, edgecolor= 'k', facecolor='none',
                      linewidth=1, linestyle='-')
        ax.add_patch(box)
        return box
    
    def plot_Gegenfurtner_Wishart_space(self, settings: PlotGegenfurtner, ax=None,
                                        flag_plot_ref=True):
        """
        Plot threshold ellipses at Gegenfurtner’s reference locations in model (Wishart) space.
    
        This function creates subplot (1) in the comparison panel. It shows ellipses centered at 
        reference stimuli from Gegenfurtner's experiment, transformed into model space.
        
        If called by `plot_Gegenfurtner_Wishart_space_zoomed_out`, we suppress the reference
        cross markers by setting `flag_plot_ref=False`.
    
        Parameters
        ----------
        settings : PlotGegenfurtner
            Visualization settings (e.g., figure size, marker style, line width).
            
        ax : matplotlib.axes.Axes or None, optional
            Axes object to plot into. If None, a new figure and axes are created.
    
        flag_plot_ref : bool, optional
            Whether to plot the reference location markers ('+' signs).
            Set to False when used inside the zoomed-out version to avoid redundancy.

        """
        # Create a new figure and axes if none were provided
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=settings.fig_size, dpi=settings.dpi)
        else:
            fig = ax.figure
    
        # Plot ellipses and reference points at each stimulus location
        for n in range(self.Grefs):
            if flag_plot_ref:
                #reference color illustrated as '+'
                ax.scatter(*self.Gcomp['ref_pts_W'][:2, n],
                           color=settings.ref_lc,
                           marker=settings.ref_marker,
                           lw=settings.ref_lw,
                           s=settings.ref_ms)
            #ellipses
            ax.plot(*self.Gcomp['fine_ell_W'][n], c=self.cmap_Grefs[n], lw=settings.modelpred_lw)
    
        # Plot ellipse at the center of the model grid (used as neutral reference)
        idx_row = self.modelpred.num_grid_pts1 // 2
        idx_col = self.modelpred.num_grid_pts2 // 2
        ax.plot(*self.modelpred.fitEll_unscaled[idx_row, idx_col], c='grey', lw=settings.modelpred_lw)
    
        # Optionally plot a reference marker at the origin
        if flag_plot_ref:
            ax.scatter(0, 0,
                       color=settings.ref_lc,
                       marker=settings.ref_marker,
                       s=settings.ref_ms,
                       lw=settings.ref_lw)
    
        # Set axis labels and grid
        ax.set_xlabel(settings.xlabel_W)
        ax.set_ylabel(settings.ylabel_W)
        ax.grid(True, color='grey', linewidth=settings.grid_lw)
    
        # Compute axis bound and apply square layout
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        ax_ub = np.max(np.abs([*ax_xlim, *ax_ylim]))
        #ax_ub : The computed axis bound (maximum extent of the data), used for setting square axes
        #and for drawing a bounding box in the zoomed-out subplot.
        ticks = np.linspace(-np.floor(ax_ub * 10) / 10, np.floor(ax_ub * 10) / 10, 3)
        DKLRelatedSpacesVisualization.set_square_axis(ax, ax_ub, ticks)
    
        return fig, ax, ax_ub
    
    def plot_Gegenfurtner_Wishart_space_zoomed_out(self, box_ub,
                                                   settings: PlotGegenfurtner,
                                                   ax=None):
        """
        Plot subplot (2): Zoomed-out view of the model (Wishart) space with ellipses 
        across the full reference grid.
    
        This panel shows the full 2D grid of threshold ellipses predicted by the model, 
        illustrating the broader region of color space compared to the limited subset 
        used in Gegenfurtner’s experiment.
    
        """
        # Create a new figure and axis if none provided
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=settings.fig_size, dpi=settings.dpi)
        else:
            fig = ax.figure
            
        # Flatten the 2D grid of ellipses for simplified iteration
        base_shape = self.modelpred.fitEll_unscaled.shape
        ell = self.modelpred.fitEll_unscaled.reshape(-1, *base_shape[2:])
    
        # Plot each ellipse using the corresponding colormap
        for n in range(ell.shape[0]):
            ax.plot(*ell[n], c=self.cmap_Wrefs[n], lw= settings.ell_fullgrid_lw)
    
        # Overlay Gegenfurtner's ellipses (without reference markers)
        self.plot_Gegenfurtner_Wishart_space(ax=ax, settings=settings, flag_plot_ref=False)
    
        # Draw a black square box indicating the region used in subplot (1)
        # The extent of the square box that will be overlaid to highlight the region
        # corresponding to Gegenfurtner's sampled reference points.
        DKLRelatedSpacesVisualization.draw_square_box(ax, box_ub)
    
        # Set axis limits and ticks to show the full model space
        DKLRelatedSpacesVisualization.set_square_axis(ax, 1, settings.xticks_W)
        
        return fig, ax
        
    def plot_DKL_space(self, settings: PlotGegenfurtner, ax=None):
        """
        Plot subplot (3): Threshold ellipses at Gegenfurtner’s reference locations in DKL space.
    
        This subplot shows how the ellipses derived from Gegenfurtner’s reference colors appear
        when transformed into the standard DKL space.
    
        """
        # Create a new figure and axis if none provided
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=settings.fig_size, dpi=settings.dpi)
        else:
            fig = ax.figure
    
        # Plot ellipses and reference markers at each Gegenfurtner stimulus location
        for n in range(self.Grefs):
            # Plot '+' marker at the reference stimulus location
            ax.scatter(*self.Gcomp['ref_pts_DKL'][:, n],
                       marker=settings.ref_marker,
                       c=settings.ref_lc,
                       s=settings.ref_ms,
                       lw=settings.ref_lw)
    
            # Plot the threshold ellipse centered at the reference point
            fine_ell_DKL_n = (self.Gcomp['fine_ell_DKL'][n, :]
                              + self.Gcomp['ref_pts_DKL'][:, n][:, np.newaxis])
            ax.plot(*fine_ell_DKL_n, c=self.cmap_Grefs[n], lw=settings.modelpred_lw)
    
        # Plot the ellipse at the achromatic center
        ax.plot(*self.Gcomp['fine_ell_grey_DKL'], c='grey', lw=settings.modelpred_lw)
    
        # Plot a '+' marker at the origin (achromatic point)
        ax.scatter(0, 0, color=settings.ref_lc, marker=settings.ref_marker, s=settings.ref_ms)
    
        # Set axis properties
        ax.set_aspect('equal', adjustable='box')  # Keep axes square
        ax.set_xlabel(settings.xlabel_DKL)
        ax.set_ylabel(settings.ylabel_DKL)
        ax.grid(True, color='grey', linewidth=settings.grid_lw)
        ax.set_xticks(settings.xticks_DKL)
    
        return fig, ax
            
    def plot_sDKL_space(self, settings: PlotGegenfurtner, ax=None, flag_plot_ref=True):
        """
        Plot subplot (4): Threshold ellipses in stretched DKL (sDKL) space.
    
        In sDKL space, the ellipse at the achromatic center is normalized to have unit
        length along the horizontal and vertical axes. In Gegenfurtner's study, other reference
        stimuli were evenly spaced along a circular contour (radius = 6.5) centered at the 
        achromatic point.
    
        This method may also be called by `plot_sDKL_zoomed_out`, in which case the reference 
        markers ('+' signs) are suppressed by setting `flag_plot_ref=False`.
            
        """
        # Create a new figure and axis if none provided
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=settings.fig_size, dpi=settings.dpi)
        else:
            fig = ax.figure
    
        # Plot ellipses and optionally reference markers for each sDKL reference point
        for n in range(self.Grefs):
            if flag_plot_ref:
                ax.scatter(*self.Gcomp['sDKL_circle_pts'][:,n],
                           marker=settings.ref_marker,
                           color=settings.ref_lc,
                           s=settings.ref_ms,
                           lw=settings.ref_lw)
    
            # Plot the threshold ellipse centered at the reference point
            fine_ell_sDKL_n = (self.Gcomp['fine_ell_sDKL'][n]
                               + self.Gcomp['sDKL_circle_pts'][:,n][:,np.newaxis])
            ax.plot(*fine_ell_sDKL_n, c=self.cmap_Grefs[n], lw=settings.modelpred_lw)
    
        # Plot the ellipse at the achromatic center (gray ellipse)
        ax.plot(*self.Gcomp['fine_ell_grey_sDKL'], c='grey', lw=settings.modelpred_lw)
    
        # Optionally plot a '+' marker at the origin
        if flag_plot_ref:
            ax.scatter(0, 0, color=settings.ref_lc, marker=settings.ref_marker, s=settings.ref_ms)
    
        # Set axis labels, grid, and enforce square axis limits
        ax.set_xlabel(settings.xlabel_sDKL)
        ax.set_ylabel(settings.ylabel_sDKL)
        ax.grid(True, color='grey', linewidth=settings.grid_lw)
        DKLRelatedSpacesVisualization.set_square_axis(
            ax,
            settings.G_ax_ub,
            np.linspace(-settings.G_ax_ub, settings.G_ax_ub, 5)
        )
    
        return fig, ax

    def plot_sDKL_zoomed_out(self, grid_sDKL, fine_ell_sDKL_grid, 
                             settings: PlotGegenfurtner, ax=None):
        """
        Plot subplot (5): Zoomed-out view of the stretched DKL (sDKL) space with ellipses 
        over the entire reference grid.
    
        This panel shows the full grid of predicted ellipses in sDKL space, providing context 
        for where Gegenfurtner's stimuli were sampled relative to the broader color space 
        explored in our study. A square box highlights the region covered by subplot (4).
    
        Parameters
        ----------
        grid_sDKL : np.ndarray, shape (N, N, 2)
            Grid of reference points in stretched DKL space.
    
        fine_ell_sDKL_grid : np.ndarray, shape (N, N, 2, 200)
            Predicted ellipses at each grid point in sDKL space.
    
        """
        # Create a new figure and axis if none provided
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=settings.fig_size, dpi=settings.dpi)
        else:
            fig = ax.figure
            
        # Translate each predicted ellipse to its corresponding grid center
        base_shape = fine_ell_sDKL_grid.shape
        ell_temp = fine_ell_sDKL_grid + grid_sDKL[..., np.newaxis]
        ell = ell_temp.reshape(-1, *base_shape[2:])
    
        # Plot each ellipse across the full sDKL grid
        for n in range(ell.shape[0]):
            ax.plot(*ell[n], c=self.cmap_Wrefs[n], lw=settings.ell_fullgrid_lw)
    
        # Overlay Gegenfurtner’s reference ellipses without '+' markers
        self.plot_sDKL_space(ax=ax, settings=settings, flag_plot_ref=False)
    
        # Draw a black square box indicating the region corresponding to subplot (4)
        DKLRelatedSpacesVisualization.draw_square_box(ax, settings.G_ax_ub)
    
        # Compute axis bounds based on data range, and set square layout
        ax_ub = np.ceil(np.max(np.abs(fine_ell_sDKL_grid + grid_sDKL[..., np.newaxis])) / 2) * 2 + 2
        DKLRelatedSpacesVisualization.set_square_axis(ax, ax_ub, np.linspace(-ax_ub, ax_ub, 5))
        
        return fig, ax
    
    def plot_Gegenfurtner_comparison(self, grid_sDKL, grid_W, fine_ell_sDKL_grid, 
                                     settings: PlotGegenfurtner):
        """
        Generate a 2x3 panel figure comparing threshold ellipses across different color spaces,
        including those used in Gegenfurtner's study and in our model predictions.
    
        Subplots:
        1. Threshold ellipses at Gegenfurtner’s reference locations in model (Wishart) space.
        2. Subplot (1) shown within the full model space to highlight the local sampling region.
        3. Ellipses at Gegenfurtner’s reference locations in standard DKL space.
        4. Ellipses in the stretched DKL space, where the ellipse at the achromatic center is 
           normalized to unit length along each axis.
        5. Subplot (4) shown within the context of the full stretched DKL space explored in our study.
    
        Parameters
        ----------
        grid_sDKL : np.ndarray (N x N x 2)
            Grid of reference stimuli in stretched DKL space.
    
        grid_W : np.ndarray (N x N x 2)
            Grid of reference stimuli in model (Wishart) space.
    
        fine_ell_sDKL_grid : np.ndarray (N x N x 2 x 200)
            Threshold ellipses evaluated across the sDKL grid.
    
        settings : PlotGegenfurtner
            Visualization parameters including layout and styling.
    
        Returns
        -------
        None
        """
        # Initialize the figure and layout
        fig = plt.figure(figsize=settings.fig_size, dpi=settings.dpi)
        gs = gridspec.GridSpec(settings.fig_nrows, settings.fig_ncols, 
                               width_ratios=settings.width_ratios, 
                               height_ratios=settings.height_ratios)
        
        # Assign subplots
        ax1 = fig.add_subplot(gs[0, 0])  # Top-left
        ax2 = fig.add_subplot(gs[1, 0])  # Bottom-left
        ax3 = fig.add_subplot(gs[:, 1])  # Center column (spanning rows)
        ax4 = fig.add_subplot(gs[0, 2])  # Top-right
        ax5 = fig.add_subplot(gs[1, 2])  # Bottom-right
    
        # Build the color map for the grid in Wishart space
        self._construct_cmap_Wrefs(grid_W)
        
        # Panel 1: Gegenfurtner ellipses in model space (local)
        _, _, box_ub = self.plot_Gegenfurtner_Wishart_space(settings=settings, ax=ax1)
    
        # Panel 2: Zoomed-out model space with full grid context
        self.plot_Gegenfurtner_Wishart_space_zoomed_out(box_ub, settings=settings, ax=ax2)
    
        # Panel 3: Gegenfurtner ellipses in DKL space
        self.plot_DKL_space(settings=settings, ax=ax3)
    
        # Panel 4: Gegenfurtner ellipses in stretched DKL space
        self.plot_sDKL_space(settings=settings, ax=ax4)
    
        # Panel 5: Zoomed-out stretched DKL space with full grid
        self.plot_sDKL_zoomed_out(grid_sDKL, fine_ell_sDKL_grid, settings=settings, ax=ax5)
    
        # Adjust layout to avoid overlaps
        plt.tight_layout()
    
        # Save figure if enabled
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
