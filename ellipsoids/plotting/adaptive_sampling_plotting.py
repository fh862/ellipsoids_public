#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:39:59 2024

@author: fangfang
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime
import plotly.graph_objects as go
import os
from analysis.color_thres import color_thresholds
from plotting.wishart_plotting import PlottingTools, PlotSettingsBase
from plotting.wishart_predictions_plotting import Plot3DPredHTMLSettings,\
    WishartPredictionsVisualization_html

#%%    
@dataclass
class Plot2DSamplingSettings(PlotSettingsBase):
    ticks: np.ndarray = field(default_factory=lambda: np.linspace(-0.8, 0.8, 5))  # W unit as numpy array
    linealpha: float = 0.5
    ref_marker: str = '+'
    ref_markersize: float = 20
    ref_markeralpha: float = 0.8
    ref_label: str = 'Reference stimulus'
    comp_label: str = 'Comparison stimulus'
    comp_marker: str = 'o'
    comp_markersize: float = 4
    comp_markeralpha: float = 0.8
    lw: float =  0.5
    flag_rescale_axes_label: bool = False
    flag_add_trialNum_title: bool = False
    fig_size: Tuple[float, float] = (3,3.5)
    visualize_bounds: bool = True
    bounds: List[float] = field(default_factory=lambda: [-0.6, 0.6])  # [lower_bound, upper_bound] in W_unit
    bounds_alpha: float = 0.2
    bounds_label: str = 'Bounds for the reference'
    plane_2D: str = ''
    title: str = field(default='Isoluminant plane')
    fig_name: str = field(default_factory=lambda: f'RandomSamples_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
@dataclass
class Plot3DSamplingSettings(PlotSettingsBase):
    ticks: np.ndarray = field(default_factory=lambda: np.linspace(-0.8, 0.8, 5))  # W unit as numpy array
    linealpha: float = 0.5
    ref_marker: str = '+'
    ref_markersize: float = 20
    ref_markeralpha: float = 0.8
    ref_label: str = 'Reference stimulus'
    comp_label: str = 'Comparison stimulus'
    comp_marker: str = 'o'
    comp_markersize: float = 4
    comp_markeralpha: float = 0.8
    lw: float =  0.5
    flag_rescale_axes_label: bool = False
    fig_size: Tuple[float, float] = (5, 4)
    plane_3D: str = 'RGB space'
    fig_name: str = field(default_factory=lambda: f'3D_randRef_nearContourComp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
@dataclass
class Plot3DSamplingHTMLSettings(Plot3DPredHTMLSettings):
    # Line segments between xref and x1
    line_width: float = 2.0
    line_opacity: float = 0.3

    # xref visual (drawn as "+" text)
    xref_text: str = "+"
    xref_text_size: float = 12
    xref_opacity: float = 0.3

    # x1 visual (marker)
    x1_marker_symbol: str = "circle"
    x1_marker_size: float = 2.0
    x1_opacity: float = 0.3

#%%
class SamplingRefCompPairVisualization(PlottingTools):
    def __init__(self, ndims, color_thres_data, settings: PlotSettingsBase, 
                 save_fig = False, save_format = 'pdf'):
        """
        Initialize an instance of sampling_ref_comp_pair_visualization, a subclass of
        wishart_model_basics_visualization, which extends its functionality for specific
        visualization tasks related to sampling reference and comparison stimulus pairs.

        """
        super().__init__(settings, save_fig, save_format)
        self.ndims = ndims
        self.color_thres = color_thres_data
    
    def _plot_2D_sampling(self, xref, xcomp, settings: Plot2DSamplingSettings, ax = None):
        """
        This function plots the sampled pairs of reference stimulus and comparison
        stimulus in a selected 2D plane.
    
        Parameters:
        ----------
        xref : np.array, shape: (N, 2)
            Coordinates for the reference stimuli in the 2D plane of the RGB space.
        xcomp : np.array, shape: (N, 2)
            Coordinates for the comparison stimuli that are paired with the reference stimuli.
        idx_fixedPlane : int
            Index indicating which of the RGB dimensions is fixed (0 for R, 1 for G, 2 for B).
        fixedVal : float
            The value at which the fixed dimension is held, must be between 0 and 1.
        ax (optional): matplotlib.axes.Axes
            The axes object for the plot. This will contain the visual representation of the data.
        """
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize = settings.fig_size, dpi= settings.dpi)
        else:
            fig = ax.figure
        
        # Mapping reference points to colors in RGB space, including the fixed dimension.
        if self.color_thres.plane_2D in ['Isoluminant plane', 'LSisolating plane']:
            cmap = self.color_thres.W2D_to_rgb(xref)
        else:
            cmap = self.color_thres.W_unit_to_N_unit(xref)
            cmap = np.insert(cmap, self.color_thres.fixed_color_dim,
                             np.ones((xref.shape[0],))*self.color_thres.fixed_value, 
                             axis=1)
        
        # Optional visualization of bounds as a grey patch on the plot.
        if settings.visualize_bounds:
            bds = settings.bounds
            rectangle = Rectangle((bds[0], bds[0]), bds[1] - bds[0],
                                  bds[1] - bds[0], facecolor='grey',
                                  alpha= settings.bounds_alpha)  # Adjust alpha for transparency
            rectangle.set_label(settings.bounds_label)  # Set the label here
            ax.add_patch(rectangle)
        
        # Plotting the reference and comparison points.
        ax.scatter(xref[:,0],xref[:,1], c = cmap, marker = settings.ref_marker,
                   s = settings.ref_markersize, alpha = settings.ref_markeralpha,
                   label = settings.ref_label)
        ax.scatter(xcomp[:,0], xcomp[:,1], c = cmap, marker = settings.comp_marker,
                   s = settings.comp_markersize, alpha = settings.comp_markeralpha,
                   label = settings.comp_label) 
            
        # Drawing lines connecting reference and comparison points.
        for l in range(xref.shape[0]):
            ax.plot([xref[l,0],xcomp[l,0]], [xref[l,1],xcomp[l,1]], c = cmap[l],
                    alpha = settings.linealpha,lw = settings.lw)
        
        # Configuring grid, aspect ratio, and ticks based on the plotting parameters.
        plt.grid(alpha = 0.2)
        ax.set_aspect('equal', adjustable='box')
        self._update_axes_limits(ax)
        
        # Configure tick marks for axes.
        if settings.flag_rescale_axes_label:
            self._update_axes_labels(ax, settings.ticks, 
                                     self.color_thres.W_unit_to_N_unit(settings.ticks), nsteps = 1)
        else:
            self._update_axes_labels(ax, settings.ticks, settings.ticks, nsteps = 1)
        ax.tick_params(axis='both', which='major', labelsize=settings.fontsize)

        # Optionally add a trial number to the title.
        if settings.flag_add_trialNum_title:
            self._configure_labels_and_title(ax, title = f'Trials: {xref.shape[0]}')
        else:
            self._configure_labels_and_title(ax, title = settings.title)
        
        # Set the legend with a custom location and show the plot.
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.47),
                   fontsize = settings.fontsize)
        fig.tight_layout()
        
        # Save the figure if the directory is set and saving is enabled.
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
            
        return fig, ax
            
            
    def _plot_3D_sampling(self, xref, xcomp, settings: Plot3DSamplingSettings, ax = None):
        """
        This function plots the sampled pairs of reference stimulus and comparison
        stimulus in a 3D RGB cube.

        Parameters
        ----------
        xref : np.array, shape: (N, 3)
            The reference stimulus coordinates in RGB space, where N is the number of pairs.
        xcomp : np.array, shape: (N, 3)
            The comparison (odd) stimulus coordinates, paired with the reference stimulus.
        ax (optional): matplotlib.axes.Axes
            The axes object on which the data will be plotted.

        """        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig = plt.figure(figsize = settings.fig_size, dpi = settings.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Mapping the RGB data to a [0,1] range suitable for display as colors.
        color_map_ref = self.color_thres.W_unit_to_N_unit(xref)  
        color_map_comp = self.color_thres.W_unit_to_N_unit(xcomp)
        
        # Plotting the reference stimuli.
        ax.scatter(*xref.T, c=color_map_ref,
                   marker= settings.ref_marker, s = settings.ref_markersize, 
                   alpha= settings.ref_markeralpha, label = settings.ref_label)
        # Plotting the comparison stimuli.
        ax.scatter(*xcomp.T, c=color_map_comp,
                   marker=settings.comp_marker, s = settings.comp_markersize,
                   alpha= settings.comp_markeralpha, label = settings.comp_label)
        
        # Optionally draw lines between reference and comparison points.
        for l in range(xref.shape[0]):
            ax.plot([xref[l, 0], xcomp[l, 0]],[xref[l, 1], xcomp[l, 1]],
                    [xref[l, 2], xcomp[l, 2]], color= np.array(color_map_ref[l]),
                    alpha= settings.linealpha, lw= settings.lw)

        # Configuring ticks and labels based on the settings.
        if settings.flag_rescale_axes_label:
            self._update_axes_labels(ax, settings.ticks,
                                     self.color_thres.W_unit_to_N_unit(settings.ticks), 
                                     ndims = 3)
            self._configure_labels_and_title(ax, ndims = 3, title = 'RGB cube')
        else:
            self._update_axes_labels(ax, settings.ticks, settings.ticks, ndims = 3)
            self._configure_labels_and_title(ax, ndims = 3, title = '3D cube')
        # Setting the title and labels for the axes based on 3D plane settings.
        self._update_axes_limits(ax, ndims = 3)
        ax.grid(True, linewidth = 0.2)
        # Add legend outside the plot at the bottom
        plt.legend(loc='lower center', 
                   bbox_to_anchor=(0.5, -0.3), 
                   fontsize = settings.fontsize - 1
                   )
        ax.set_box_aspect((1, 1, 1))
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.3,
                            wspace=-0.05, hspace=-0.05)
        plt.show()
        # Save the figure if required.
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
            
        return fig, ax
    
    #%%
    def plot_sampling(self, xref, xcomp, settings, ax = None):
        """
        Plots the sampling data in either 2D or 3D based on the dimensionality (ndims) of the task.
        This function is a dispatcher that calls either _plot_2D_sampling or _plot_3D_sampling
        depending on whether ndims is set to 2 or 3.
        
        Returns:
            fig, ax : A tuple containing the figure and axes objects with the plotted data.
        """
        if self.ndims == 2:
            fig, ax = self._plot_2D_sampling(xref, xcomp, settings = settings, ax = ax)
        else:
            fig, ax = self._plot_3D_sampling(xref, xcomp, settings = settings, ax = ax)
        return fig, ax 
            
#%%
class SamplingRefCompPairVisualization_html(WishartPredictionsVisualization_html):
    def __init__(self, settings: Plot3DSamplingHTMLSettings):
        """
        This class plots the adaptively sampled trials in 3D color space
        
        Parameters
        ----------
        settings : Plot3DSamplingHTMLSettings
            Visual styling and layout parameters (extends Plot3DPredHTMLSettings).
        """
        super().__init__(settings)  # sets self.st, self.lighting, self.light_position

    def color_from_W(self, x):
        """Map a W-unit coordinate to CSS rgb() string via W_to_rgb01."""
        c01 = np.asarray(color_thresholds.W_unit_to_N_unit(x), float)
        c01 = np.clip(c01, 0.0, 1.0)
        return self.to_rgb_str(c01)   # inherited staticmethod

    def add_ref_comp_segments(self, fig, xref, x1, colors_xref):
        """Add line segments from xref -> x1, colored by xref color."""
        xref = np.asarray(xref, float)
        x1   = np.asarray(x1,   float)
        N = xref.shape[0]

        for i in range(N):
            a = xref[i]
            b = x1[i]
            fig.add_trace(go.Scatter3d(
                x=[a[0], b[0]],
                y=[a[1], b[1]],
                z=[a[2], b[2]],
                mode="lines",
                line=dict(color=colors_xref[i], width=self.st.line_width),
                opacity=self.st.line_opacity,
                hoverinfo="skip",
                showlegend=False,
            ))
        return fig

    def add_markers(self, fig, xref, x1, colors_xref, colors_x1):
        """Add xref '+' text markers and x1 circle markers."""
        xref = np.asarray(xref, float)
        x1   = np.asarray(x1,   float)

        # xref as "+" text
        fig.add_trace(go.Scatter3d(
            x=xref[:, 0],
            y=xref[:, 1],
            z=xref[:, 2],
            mode="text",
            text=[self.st.xref_text] * xref.shape[0],
            textposition="middle center",
            textfont=dict(
                family=self.st.font_family,
                size=self.st.xref_text_size,
            ),
            textfont_color=colors_xref,
            opacity=self.st.xref_opacity,
            hoverinfo="skip",
            showlegend=False,
        ))

        # x1 as circle markers
        fig.add_trace(go.Scatter3d(
            x=x1[:, 0],
            y=x1[:, 1],
            z=x1[:, 2],
            mode="markers",
            marker=dict(
                symbol=self.st.x1_marker_symbol,
                size=self.st.x1_marker_size,
                color=colors_x1,
                opacity=self.st.x1_opacity,
            ),
            hoverinfo="skip",
            showlegend=False,
            name="x1",
        ))
        return fig

    def plot_sampling(self, xref, x1):
        """
        Create a 3D sampling figure showing xref–x1 pairs.

        Parameters
        ----------
        xref, x1 : array-like, shape (N, 3)
            W-unit coordinates of references and comparisons.

        Returns
        -------
        fig : go.Figure
        """
        xref = np.asarray(xref, float)
        x1   = np.asarray(x1,   float)

        # Colors from W -> RGB
        colors_xref = [self.color_from_W(r) for r in xref]
        colors_x1   = [self.color_from_W(r) for r in x1]

        fig = go.Figure()
        fig = self.add_ref_comp_segments(fig, xref, x1, colors_xref)
        fig = self.add_markers(fig, xref, x1, colors_xref, colors_x1)

        # `apply_3d_layout` is inherited from WishartPredictionsVisualization_html
        fig = self.apply_3d_layout(fig)
        return fig