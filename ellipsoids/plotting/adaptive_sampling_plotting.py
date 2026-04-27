#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:39:59 2024

@author: fangfang
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
import os
from plotly.offline.offline import get_plotlyjs
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

    @staticmethod
    def _build_segment_coords(xref, x1):
        """Flatten xref->x1 segments into Plotly-ready arrays separated by NaN."""
        xs = []
        ys = []
        zs = []
        for a, b in zip(xref, x1):
            xs.extend([a[0], b[0], np.nan])
            ys.extend([a[1], b[1], np.nan])
            zs.extend([a[2], b[2], np.nan])
        return xs, ys, zs

    @staticmethod
    def _build_segment_colors(colors_xref):
        """Repeat each ref color for the two endpoints and the separator slot."""
        seg_colors = []
        for color in colors_xref:
            seg_colors.extend([color, color, color])
        return seg_colors

    def _make_segment_trace(self, xs, ys, zs, seg_colors, name="xref-x1 pairs",
                            legendgroup=None, showlegend=False):
        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=seg_colors, width=self.st.line_width),
            opacity=self.st.line_opacity,
            hoverinfo="skip",
            showlegend=showlegend,
            legendgroup=legendgroup,
            name=name,
        )

    def _make_xref_trace(self, xref, colors_xref, legendgroup=None):
        return go.Scatter3d(
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
            legendgroup=legendgroup,
            name="xref",
        )

    def _make_x1_trace(self, x1, colors_x1, legendgroup=None):
        return go.Scatter3d(
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
            legendgroup=legendgroup,
            name="x1",
        )

    def _make_cumulative_frame(self, frame_name, trace_payloads):
        return go.Frame(
            name=frame_name,
            data=trace_payloads["data"],
            traces=trace_payloads["traces"],
        )

    @staticmethod
    def _build_exponential_counts(n_pairs):
        """Return cumulative pair counts: 1, 2, 4, ... and always include n_pairs."""
        if n_pairs <= 0:
            return []

        counts = []
        count = 1
        while count < n_pairs:
            counts.append(count)
            count *= 2

        if not counts or counts[-1] != n_pairs:
            counts.append(n_pairs)
        return counts

    @staticmethod
    def _normalize_trial_categories(trial_category, num_trial_per_category, n_pairs):
        if trial_category is None and num_trial_per_category is None:
            return None

        if trial_category is None or num_trial_per_category is None:
            raise ValueError(
                "trial_category and num_trial_per_category must either both be provided or both be None."
            )
        if len(trial_category) != len(num_trial_per_category):
            raise ValueError("trial_category and num_trial_per_category must have the same length.")
        if len(trial_category) == 0:
            raise ValueError("trial_category must contain at least one category.")

        counts = [int(n) for n in num_trial_per_category]
        if any(n <= 0 for n in counts):
            raise ValueError("Each entry in num_trial_per_category must be positive.")
        if sum(counts) != n_pairs:
            raise ValueError(
                "The sum of num_trial_per_category must equal the total number of trial pairs."
            )

        categories = []
        start = 0
        for idx, (label, count) in enumerate(zip(trial_category, counts)):
            stop = start + count
            categories.append(
                {
                    "idx": idx,
                    "label": str(label),
                    "start": start,
                    "stop": stop,
                    "count": count,
                }
            )
            start = stop
        return categories

    def _add_build_controls(self, fig, frame_names):
        slider_steps = [
            dict(
                method="animate",
                label=frame_name.split("_")[-1],
                args=[
                    [frame_name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            )
            for i, frame_name in enumerate(frame_names)
        ]

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.02,
                    y=1.08,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="Show All",
                            method="animate",
                            args=[
                                ["all_pairs"],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="Build",
                            method="animate",
                            args=[
                                frame_names,
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 120, "redraw": True},
                                    "transition": {"duration": 0},
                                    "fromcurrent": False,
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=max(len(frame_names) - 1, 0),
                    currentvalue={"prefix": "Pairs shown: "},
                    pad={"t": 52},
                    steps=slider_steps,
                )
            ],
        )

    def _add_category_legend(self, fig):
        fig.update_layout(
            legend=dict(
                title=dict(text="Categories"),
                orientation="v",
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                groupclick="togglegroup",
            ),
            margin=dict(r=180),
        )

    def _build_trace_payload(self, xref, x1, colors_xref, colors_x1, upto_idx,
                             legendgroup=None, name="xref-x1 pairs", showlegend=False):
        xref_slc = xref[:upto_idx]
        x1_slc = x1[:upto_idx]
        colors_xref_slc = colors_xref[:upto_idx]
        colors_x1_slc = colors_x1[:upto_idx]
        xs, ys, zs = self._build_segment_coords(xref_slc, x1_slc)
        seg_colors = self._build_segment_colors(colors_xref_slc)
        return [
            self._make_segment_trace(
                xs, ys, zs, seg_colors,
                name=name,
                legendgroup=legendgroup,
                showlegend=showlegend,
            ),
            self._make_xref_trace(xref_slc, colors_xref_slc, legendgroup=legendgroup),
            self._make_x1_trace(x1_slc, colors_x1_slc, legendgroup=legendgroup),
        ]

    def plot_sampling(self, xref, x1, trial_category=None, num_trial_per_category=None):
        """
        Create a 3D sampling figure showing xref–x1 pairs.

        Parameters
        ----------
        xref, x1 : array-like, shape (N, 3)
            W-unit coordinates of references and comparisons.
        trial_category : sequence of str, optional
            Category labels for contiguous blocks of trials. When provided with
            `num_trial_per_category`, category buttons are added to switch the
            figure to a specific trial block.
        num_trial_per_category : sequence of int, optional
            Number of trials in each category. The cumulative sums define the
            slices used to select trials for each category.

        Returns
        -------
        fig : go.Figure
        """
        xref = np.asarray(xref, float)
        x1   = np.asarray(x1,   float)
        if xref.shape != x1.shape:
            raise ValueError("xref and x1 must have the same shape.")
        if xref.ndim != 2 or xref.shape[1] != 3:
            raise ValueError("xref and x1 must have shape (N, 3).")
        categories = self._normalize_trial_categories(
            trial_category, num_trial_per_category, xref.shape[0]
        )

        # Colors from W -> RGB
        colors_xref = [self.color_from_W(r) for r in xref]
        colors_x1   = [self.color_from_W(r) for r in x1]

        fig = go.Figure()

        if xref.shape[0] > 0:
            if categories is None:
                for trace in self._build_trace_payload(
                    xref, x1, colors_xref, colors_x1, upto_idx=xref.shape[0]
                ):
                    fig.add_trace(trace)

                build_counts = self._build_exponential_counts(xref.shape[0])
                build_frame_names = [f"build_{count}" for count in build_counts]
                frames = [
                    self._make_cumulative_frame(
                        frame_name,
                        {
                            "data": self._build_trace_payload(
                                xref, x1, colors_xref, colors_x1, upto_idx=count
                            ),
                            "traces": [0, 1, 2],
                        },
                    )
                    for frame_name, count in zip(build_frame_names, build_counts)
                ]
                frames.append(
                    self._make_cumulative_frame(
                        "all_pairs",
                        {
                            "data": self._build_trace_payload(
                                xref, x1, colors_xref, colors_x1, upto_idx=xref.shape[0]
                            ),
                            "traces": [0, 1, 2],
                        },
                    )
                )
                fig.frames = frames
                self._add_build_controls(fig, build_frame_names)
            else:
                for cat in categories:
                    xref_cat = xref[cat["start"]:cat["stop"]]
                    x1_cat = x1[cat["start"]:cat["stop"]]
                    colors_xref_cat = colors_xref[cat["start"]:cat["stop"]]
                    colors_x1_cat = colors_x1[cat["start"]:cat["stop"]]
                    for trace in self._build_trace_payload(
                        xref_cat,
                        x1_cat,
                        colors_xref_cat,
                        colors_x1_cat,
                        upto_idx=cat["count"],
                        legendgroup=cat["label"],
                        name=cat["label"],
                        showlegend=True,
                    ):
                        fig.add_trace(trace)

                build_counts = self._build_exponential_counts(xref.shape[0])
                build_frame_names = [f"build_{count}" for count in build_counts]
                frames = []
                trace_indices = list(range(3 * len(categories)))

                for frame_name, global_count in zip(build_frame_names, build_counts):
                    frame_data = []
                    for cat in categories:
                        xref_cat = xref[cat["start"]:cat["stop"]]
                        x1_cat = x1[cat["start"]:cat["stop"]]
                        colors_xref_cat = colors_xref[cat["start"]:cat["stop"]]
                        colors_x1_cat = colors_x1[cat["start"]:cat["stop"]]
                        upto_idx = min(max(global_count - cat["start"], 0), cat["count"])
                        frame_data.extend(
                            self._build_trace_payload(
                                xref_cat,
                                x1_cat,
                                colors_xref_cat,
                                colors_x1_cat,
                                upto_idx=upto_idx,
                                legendgroup=cat["label"],
                                name=cat["label"],
                                showlegend=True,
                            )
                        )
                    frames.append(
                        self._make_cumulative_frame(
                            frame_name,
                            {
                                "data": frame_data,
                                "traces": trace_indices,
                            },
                        )
                    )

                frames.append(
                    self._make_cumulative_frame(
                        "all_pairs",
                        {
                            "data": list(fig.data),
                            "traces": trace_indices,
                        },
                    )
                )
                fig.frames = frames
                self._add_build_controls(fig, build_frame_names)
                self._add_category_legend(fig)

        # `apply_3d_layout` is inherited from WishartPredictionsVisualization_html
        fig = self.apply_3d_layout(fig)
        return fig

    def _build_category_payloads(self, xref, x1, trial_category=None, num_trial_per_category=None):
        xref = np.asarray(xref, float)
        x1 = np.asarray(x1, float)
        categories = self._normalize_trial_categories(
            trial_category, num_trial_per_category, xref.shape[0]
        )
        explicit_categories = categories is not None
        if categories is None:
            categories = [
                {
                    "idx": 0,
                    "label": "All trials",
                    "start": 0,
                    "stop": xref.shape[0],
                    "count": xref.shape[0],
                }
            ]

        colors_xref = [self.color_from_W(r) for r in xref]
        colors_x1 = [self.color_from_W(r) for r in x1]
        payloads = []
        for cat in categories:
            start = cat["start"]
            stop = cat["stop"]
            payloads.append(
                {
                    "idx": cat["idx"],
                    "label": cat["label"],
                    "xref": xref[start:stop].tolist(),
                    "x1": x1[start:stop].tolist(),
                    "colors_xref": colors_xref[start:stop],
                    "colors_x1": colors_x1[start:stop],
                }
            )
        return payloads, explicit_categories

    def write_interactive_html(
        self,
        xref,
        x1,
        output_html,
        trial_category=None,
        num_trial_per_category=None,
        page_title="Adaptive Trial Placement",
    ):
        payloads, explicit_categories = self._build_category_payloads(
            xref,
            x1,
            trial_category=trial_category,
            num_trial_per_category=num_trial_per_category,
        )

        base_fig = self.apply_3d_layout(go.Figure())
        base_fig.update_layout(
            showlegend=False,
            uirevision="sampling-html",
            margin=dict(l=0, r=0, t=0, b=0),
        )

        plotly_js = get_plotlyjs()
        base_fig_json = pio.to_json(base_fig, pretty=False)
        payloads_json = json.dumps(payloads)
        show_categories = "true" if explicit_categories else "false"

        html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #22324a;
      --muted: #6b7280;
      --edge: rgba(34, 50, 74, 0.18);
      --active: #dfeaf8;
      --active-border: #6d96cf;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .page {{
      padding: 18px 22px 26px 22px;
      box-sizing: border-box;
    }}
    .title {{
      margin: 0 0 14px 0;
      font-size: 22px;
      font-weight: 700;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(34, 50, 74, 0.10);
      border-radius: 18px;
      box-shadow: 0 18px 42px rgba(34, 50, 74, 0.08);
      overflow: hidden;
    }}
    .controls {{
      padding: 18px 18px 8px 18px;
      border-bottom: 1px solid rgba(34, 50, 74, 0.08);
    }}
    .row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .row:last-child {{
      margin-bottom: 0;
    }}
    .control-button {{
      border: 1px solid var(--edge);
      background: #ffffff;
      color: var(--ink);
      border-radius: 8px;
      padding: 10px 18px;
      font-size: 13px;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease;
    }}
    .control-button.active {{
      background: var(--active);
      border-color: var(--active-border);
    }}
    .category-button {{
      border: 1px solid var(--edge);
      background: #ffffff;
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 12px;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease, opacity 120ms ease;
    }}
    .category-button.active {{
      background: var(--active);
      border-color: var(--active-border);
    }}
    .label {{
      font-size: 13px;
      color: var(--muted);
      min-width: fit-content;
    }}
    .counter {{
      font-size: 13px;
      color: var(--muted);
      margin-left: 4px;
    }}
    .slider {{
      width: 100%;
    }}
    .slider-wrap {{
      flex: 1 1 320px;
      min-width: 240px;
    }}
    .slider-ticks {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 4px;
      margin-top: 6px;
      min-height: 32px;
    }}
    .slider-tick {{
      flex: 1 1 0;
      text-align: center;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.15;
    }}
    .slider-tick::before {{
      content: "";
      display: block;
      width: 1px;
      height: 8px;
      margin: 0 auto 4px auto;
      background: rgba(34, 50, 74, 0.28);
    }}
    .plot-wrap {{
      height: 760px;
    }}
    #sampling-plot {{
      width: 100%;
      height: 100%;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1 class="title">{page_title}</h1>
    <div class="panel">
      <div class="controls">
        <div class="row">
          <button id="show-all-button" class="control-button active" type="button">Show All</button>
          <button id="build-button" class="control-button" type="button">Build</button>
          <div id="counter-label" class="counter"></div>
        </div>
        <div id="category-row" class="row" style="display:none;">
          <div class="label">Categories:</div>
          <div id="category-buttons" class="row" style="margin-bottom:0;"></div>
        </div>
        <div class="row">
          <div class="label">Build step:</div>
          <div class="slider-wrap">
            <input id="build-slider" class="slider" type="range" min="0" max="0" value="0" step="1" disabled>
            <div id="slider-ticks" class="slider-ticks"></div>
          </div>
          <div id="slider-label" class="counter"></div>
        </div>
      </div>
      <div class="plot-wrap">
        <div id="sampling-plot"></div>
      </div>
    </div>
  </div>

  <script>{plotly_js}</script>
  <script>
    const baseFig = {base_fig_json};
    const categoryPayloads = {payloads_json};
    const showCategories = {show_categories};
    const plotDiv = document.getElementById("sampling-plot");
    const showAllButton = document.getElementById("show-all-button");
    const buildButton = document.getElementById("build-button");
    const buildSlider = document.getElementById("build-slider");
    const sliderTicks = document.getElementById("slider-ticks");
    const counterLabel = document.getElementById("counter-label");
    const sliderLabel = document.getElementById("slider-label");
    const categoryRow = document.getElementById("category-row");
    const categoryButtons = document.getElementById("category-buttons");

    let mode = "all";
    let buildStepIdx = 0;
    let latestBuildCounts = [];
    let buildTimer = null;
    const selectedCategoryIdxs = new Set(categoryPayloads.map((_, idx) => idx));
    const buildIntervalMs = 380;

    function buildExponentialCounts(nPairs) {{
      if (nPairs <= 0) return [];
      const counts = [];
      let count = 1;
      while (count < nPairs) {{
        counts.push(count);
        count *= 2;
      }}
      if (!counts.length || counts[counts.length - 1] !== nPairs) {{
        counts.push(nPairs);
      }}
      return counts;
    }}

    function getSelectedPairs() {{
      const pairs = [];
      categoryPayloads.forEach((payload, idx) => {{
        if (!selectedCategoryIdxs.has(idx)) return;
        for (let i = 0; i < payload.xref.length; i += 1) {{
          pairs.push({{
            xref: payload.xref[i],
            x1: payload.x1[i],
            refColor: payload.colors_xref[i],
            x1Color: payload.colors_x1[i],
          }});
        }}
      }});
      return pairs;
    }}

    function buildTraces(pairs) {{
      const segX = [];
      const segY = [];
      const segZ = [];
      const segColors = [];
      const xrefX = [];
      const xrefY = [];
      const xrefZ = [];
      const xrefColors = [];
      const x1X = [];
      const x1Y = [];
      const x1Z = [];
      const x1Colors = [];

      pairs.forEach((pair) => {{
        segX.push(pair.xref[0], pair.x1[0], NaN);
        segY.push(pair.xref[1], pair.x1[1], NaN);
        segZ.push(pair.xref[2], pair.x1[2], NaN);
        segColors.push(pair.refColor, pair.refColor, pair.refColor);

        xrefX.push(pair.xref[0]);
        xrefY.push(pair.xref[1]);
        xrefZ.push(pair.xref[2]);
        xrefColors.push(pair.refColor);

        x1X.push(pair.x1[0]);
        x1Y.push(pair.x1[1]);
        x1Z.push(pair.x1[2]);
        x1Colors.push(pair.x1Color);
      }});

      return [
        {{
          type: "scatter3d",
          mode: "lines",
          x: segX,
          y: segY,
          z: segZ,
          line: {{
            color: segColors,
            width: {self.st.line_width},
          }},
          opacity: {self.st.line_opacity},
          hoverinfo: "skip",
          showlegend: false,
          name: "xref-x1 pairs",
        }},
        {{
          type: "scatter3d",
          mode: "text",
          x: xrefX,
          y: xrefY,
          z: xrefZ,
          text: Array(xrefX.length).fill({self.st.xref_text!r}),
          textposition: "middle center",
          textfont: {{
            family: {self.st.font_family!r},
            size: {self.st.xref_text_size},
            color: xrefColors,
          }},
          opacity: {self.st.xref_opacity},
          hoverinfo: "skip",
          showlegend: false,
          name: "xref",
        }},
        {{
          type: "scatter3d",
          mode: "markers",
          x: x1X,
          y: x1Y,
          z: x1Z,
          marker: {{
            symbol: {self.st.x1_marker_symbol!r},
            size: {self.st.x1_marker_size},
            color: x1Colors,
            opacity: {self.st.x1_opacity},
          }},
          hoverinfo: "skip",
          showlegend: false,
          name: "x1",
        }},
      ];
    }}

    function setModeButtons() {{
      showAllButton.classList.toggle("active", mode === "all");
      buildButton.classList.toggle("active", mode === "build");
    }}

    function setCategoryButtons() {{
      if (!showCategories) return;
      Array.from(categoryButtons.children).forEach((button, idx) => {{
        button.classList.toggle("active", selectedCategoryIdxs.has(idx));
      }});
    }}

    function stopAutoBuild() {{
      if (buildTimer !== null) {{
        window.clearInterval(buildTimer);
        buildTimer = null;
      }}
    }}

    function updateSliderTicks(buildCounts) {{
      sliderTicks.innerHTML = "";
      if (!buildCounts.length) {{
        return;
      }}
      buildCounts.forEach((count) => {{
        const tick = document.createElement("div");
        tick.className = "slider-tick";
        tick.textContent = count.toLocaleString();
        sliderTicks.appendChild(tick);
      }});
    }}

    function startAutoBuild() {{
      stopAutoBuild();
      if (mode !== "build" || latestBuildCounts.length <= 1) {{
        return;
      }}
      if (buildStepIdx >= latestBuildCounts.length - 1) {{
        return;
      }}
      buildTimer = window.setInterval(() => {{
        if (mode !== "build") {{
          stopAutoBuild();
          return;
        }}
        if (buildStepIdx >= latestBuildCounts.length - 1) {{
          stopAutoBuild();
          return;
        }}
        buildStepIdx += 1;
        render(false);
      }}, buildIntervalMs);
    }}

    function updateSlider(buildCounts, visibleCount, totalCount) {{
      latestBuildCounts = buildCounts.slice();
      updateSliderTicks(buildCounts);
      if (mode === "build" && buildCounts.length > 0) {{
        buildSlider.disabled = false;
        buildSlider.min = 0;
        buildSlider.max = buildCounts.length - 1;
        buildSlider.value = buildStepIdx;
        sliderLabel.textContent = `Step ${{buildStepIdx + 1}} / ${{buildCounts.length}}`;
        counterLabel.textContent = `Pairs shown: ${{visibleCount}} of ${{totalCount}}`;
      }} else {{
        buildSlider.disabled = true;
        buildSlider.min = 0;
        buildSlider.max = 0;
        buildSlider.value = 0;
        sliderLabel.textContent = totalCount > 0 ? "Build disabled in Show All mode" : "No pairs selected";
        counterLabel.textContent = `Pairs shown: ${{visibleCount}}`;
      }}
    }}

    function render(resetBuildCounter) {{
      const selectedPairs = getSelectedPairs();
      const buildCounts = buildExponentialCounts(selectedPairs.length);

      if (mode === "build") {{
        if (resetBuildCounter) {{
          buildStepIdx = 0;
        }}
        if (buildCounts.length === 0) {{
          buildStepIdx = 0;
        }} else {{
          buildStepIdx = Math.max(0, Math.min(buildStepIdx, buildCounts.length - 1));
        }}
      }} else {{
        buildStepIdx = Math.max(buildCounts.length - 1, 0);
      }}

      const visibleCount = mode === "build"
        ? (buildCounts.length ? buildCounts[buildStepIdx] : 0)
        : selectedPairs.length;
      const visiblePairs = selectedPairs.slice(0, visibleCount);
      const traces = buildTraces(visiblePairs);
      const layout = JSON.parse(JSON.stringify(baseFig.layout || {{}}));
      const config = {{ responsive: true }};

      Plotly.react(plotDiv, traces, layout, config);
      setModeButtons();
      setCategoryButtons();
      updateSlider(buildCounts, visibleCount, selectedPairs.length);
    }}

    showAllButton.addEventListener("click", () => {{
      stopAutoBuild();
      mode = "all";
      render(false);
    }});

    buildButton.addEventListener("click", () => {{
      stopAutoBuild();
      mode = "build";
      buildStepIdx = 0;
      render(false);
      startAutoBuild();
    }});

    buildSlider.addEventListener("input", (event) => {{
      stopAutoBuild();
      mode = "build";
      buildStepIdx = Number(event.target.value || 0);
      render(false);
    }});

    if (showCategories) {{
      categoryRow.style.display = "flex";
      categoryPayloads.forEach((payload, idx) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "category-button active";
        button.textContent = payload.label;
        button.addEventListener("click", () => {{
          if (selectedCategoryIdxs.has(idx)) {{
            selectedCategoryIdxs.delete(idx);
          }} else {{
            selectedCategoryIdxs.add(idx);
          }}
          render(mode === "build");
          if (mode === "build") {{
            startAutoBuild();
          }}
        }});
        categoryButtons.appendChild(button);
      }});
    }}

    render(false);
  </script>
</body>
</html>
"""

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_str)
