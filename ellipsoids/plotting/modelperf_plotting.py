#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:03:20 2025

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Union, Optional, Any
import os
from plotting.wishart_plotting import PlottingTools, PlotSettingsBase

#%%
@dataclass
class PltBWDSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (3.6, 4)
    y_ub: float = 0.2
    num_yticks: int = 5
    flag_visualize_baseline: bool = False
    baseline_lw: float = 2
    baseline_alpha: float = 0.8
    errorbar_cs: float = 0
    errorbar_c: str = 'k'
    errorbar_m: str = 'o'
    errorbar_ms: float = 12
    errorbar_lw: float = 2
    axis_grid_alpha: float = 0.3
    x_tick_rot: int = 30
    dashed_ls: str = '--'
    dashed_lc: str = 'k'
    dashed_lw: float = 0.5
    alpha_grid: float = 0.3
    x_label: str = 'Total number of trial'
    y_label: str = 'Bures-Wasserstein distance'
    x_ticklabels: List[str] = field(default_factory=list)
    fig_name: str = 'ModelPerformance_BuresWassersteinDistance_Wishart_vs_IndvEll'

@dataclass
class PltBaselineSettings:
    linecolor: Union[Tuple[float, float, float], str] = 'k'
    linestyle: str = '-'
    linewidth: float = 2
    linealpha: float = 0.8

@dataclass
class PltSimSettings_2D(PlotSettingsBase):
    y_ub: float = 25.0
    legend_labels: List[Optional[str]] = field(default_factory=list)
    legend_title: str = ''
    cmap: List[Any] = field(default_factory=list) 
    alpha: float = 0.6
    edgecolor: Union[Tuple[float, float, float], str] = 'w'
    linestyle: str = '--'
    linewidth: float = 1
    cmap_BW: Optional[np.ndarray] = None 
    x_ub_BW: float = 0.14                
    x_ub_LU: float = 3.5                 
    nBins_curves: int = 11             
    nBins_hist: int = 33               
    fig_size: Tuple[float, float] = (4.5, 2.5) 
    nyticks: int = 3                   
    
@dataclass
class PltSimSettings_3D(PlotSettingsBase):
    y_ub: float = 100
    legend_labels: List[Optional[str]] = field(default_factory=list)
    legend_title: str = ''
    cmap: List[Any] = field(default_factory=list) 
    edgecolor: Union[Tuple[float, float, float], str] = 'w'
    linestyle: str = '--'
    linewidth: float = 1
    alpha: float = 0.6
    x_ub_BW: float = 0.18
    x_ub_LU: float = 4
    nBins_curves: int =  11
    nBins_hist: int =  33
    fig_size: Tuple[float, float] = (2.5, 3.2)
    nyticks: int = 5
    
@dataclass
class PltBaselineHistSettings:
    cmap: List[Any] = field(default_factory=list)     
    ls: List[str] = field(default_factory=list)       
    ls_median: List[str] = field(default_factory=list)
    jitter: np.ndarray = field(default_factory=lambda: np.array([]))
    lw: float = 1.0       

@dataclass
class PltVaryingHyperParamSettings(PlotSettingsBase):
    fig_size: tuple = (5, 3.25)
    fontsize: int = 9
    legend_fontsize: int = 8           
    grid_alpha: float = 0.2
    training_color: str = 'grey'
    test_color: str = 'g'
    training_fill_alpha: float = 0.3
    test_fill_alpha: float = 0.2
    marker = 'o'   
    ms: int = 4
    ls_training: str = '--'
    ls_test: str = '-'
    xlabel: str = r'Hyperparameter for irregularity ($\epsilon$)'
    xticks: Optional[np.ndarray] = None 
    xticklabels: Optional[np.ndarray] = None 
    label_CI: str = 'full range'
    fig_name: str = 'CrossValidation5folds_1reptitions_varyingHyperParam_ColorDiscrimination'
    
@dataclass
class PltVaryingTrialsNumbersSettings(PlotSettingsBase):
    fig_size: tuple = (5, 3.25)
    fontsize: int = 9
    legend_fontsize: int = 8          
    grid_alpha: float = 0.2
    ls_BWD_sum: str = '-'
    ls_BWD_CI: str = '--'
    label_line: str = 'Original dataset'
    lc: str = 'grey'
    linealpha = 1
    markeralpha = 0.5
    markersize = 3
    fill_alpha: str = 0.3
    label_CI: str = 'full range'
    ylim: Optional[List[str]] = None
    xlim: Optional[List[str]] = None
    xlabel: str = 'Included trials'
    ylabel: str = 'Sum BWD between\nthe model-estimated and\nthe ground-truth threshold ellipses'
    fig_name: str = 'TrialEfficiency_varyingTrials_Fitted_byWishart_Isoluminant plane'

#%%
class ModelPerformanceVisualization(PlottingTools):
    def __init__(self, model_perf, settings: PlotSettingsBase, 
                 save_fig=False, save_format = 'pdf'):
        super().__init__(settings, save_fig, save_format)
        self.model_perf = model_perf
        
    def _plot_baseline(self, ax, nConds_indvEll, nConds_Wishart, 
                       settings: PltBaselineSettings, cmap_corner = None):
        BW_distance_circle_median = np.median(self.model_perf.BW_distance_minEigval)
        BW_distance_corner_median = np.median(self.model_perf.BW_distance_corner, axis = 1)
        ax.plot([-nConds_Wishart-1, nConds_indvEll+2], 
                [BW_distance_circle_median, BW_distance_circle_median],
                 c = settings.linecolor,ls = settings.linestyle,
                 lw = settings.linewidth, alpha = settings.linealpha)
        for i in range(BW_distance_corner_median.shape[0]):
            if cmap_corner is not None:
                cmap_i = cmap_corner[i]
            else:
                cmap_i = 'k'
            ax.plot([-nConds_Wishart-1, nConds_indvEll+2],
                    np.array([1,1])*BW_distance_corner_median[i],
                    c = cmap_i,ls = settings.linestyle,
                    lw = settings.linewidth, alpha = settings.linealpha)
        
    def plot_BWD_indvEll_vs_Wishart(self, nConds_indvEll, BWD_median_indvEll,
                                    yerr_indvEll, nConds_Wishart, BWD_median_Wishart,
                                    yerr_Wishart, settings: PltBWDSettings, 
                                    cmap_corner = None, ax = None):
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize = settings.fig_size, dpi= settings.dpi)
        else:
            fig = ax.figure

        #plot baseline if applicable
        if settings.flag_visualize_baseline:
            self._plot_baseline(ax, nConds_indvEll, nConds_Wishart,
                                PltBaselineSettings(), cmap_corner = cmap_corner)
            
        #Wishart
        for i in range(nConds_indvEll):
            ax.errorbar(i+1, BWD_median_indvEll[i], yerr=yerr_indvEll[:,i][:,np.newaxis],
                        capsize = settings.errorbar_cs, c = settings.errorbar_c, 
                        marker = settings.errorbar_m, markersize = settings.errorbar_ms, 
                        lw = settings.errorbar_lw)
        for i in range(nConds_Wishart):
            ax.errorbar(-i-1, BWD_median_Wishart[i], yerr=yerr_Wishart[:,i][:,np.newaxis],
                         capsize = settings.errorbar_cs, c = settings.errorbar_c,  
                         marker = settings.errorbar_m, markersize = settings.errorbar_ms, 
                         lw = settings.errorbar_lw)
        ax.plot([0,0],[0, settings.y_ub], ls = settings.dashed_ls,
                 lw= settings.dashed_lw,  c = settings.dashed_lc)
        x_ticks = list(range(-nConds_Wishart,0)) + list(range(1,nConds_indvEll+1))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(settings.x_ticklabels, rotation= settings.x_tick_rot) 
        ax.set_xlabel(settings.x_label)
        ax.set_xlim([-nConds_Wishart-1, nConds_indvEll + 1])
        ax.set_yticks(np.around(np.linspace(0, settings.y_ub, settings.num_yticks),2))
        ax.set_ylim([0, settings.y_ub])
        ax.set_ylabel(settings.y_label)
        ax.grid(True, alpha = settings.alpha_grid)
        fig.tight_layout()
        # Save the figure if the directory is set and saving is enabled.
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
        return fig, ax
        
    def _plot_baseline_hist(self, ax, bin_edges, settings: PltBaselineHistSettings):
        nSets = self.model_perf.BW_benchmark.shape[0]

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for m in range(nSets):
            if len(settings.cmap) == 0: cmap_m = np.random.rand(1,3)
            else: cmap_m = settings.cmap[m];
            if len(settings.ls) == 0: ls = '--';
            else: ls = settings.ls[m]
            if len(settings.ls_median) == 0: ls_m = '-';
            else: ls_m = settings.ls_median[m]            
            median_m = np.median(self.model_perf.BW_benchmark[m].flatten())
            counts_m,_ = np.histogram(self.model_perf.BW_benchmark[m].flatten(), bins=bin_edges)
            ax.plot(bin_centers+settings.jitter[m], counts_m,  color = cmap_m,
                    ls = ls, lw = settings.lw)
            ax.plot([median_m, median_m], [0, 80], ls = ls_m, color = cmap_m
                    , lw = settings.lw)
        ax.grid(True, alpha=0.3)
        
    def plot_BWD_hist(self, bin_edges, settings: PltSimSettings_2D, ax = None):
        nSets = self.model_perf.BW_distance.shape[0]
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize = settings.fig_size, dpi= settings.dpi)
        else:
            fig = ax.figure
            
        for j in range(nSets):
            if len(settings.cmap) == 0: cmap_l = np.random.rand(1,3)
            else: cmap_l = settings.cmap[j];
            ax.hist(self.model_perf.BW_distance[j].flatten(), bins = bin_edges,
                    color = cmap_l, alpha = settings.alpha, edgecolor = settings.edgecolor,
                    label = settings.legend_labels[j])
            #plot the median
            median_j = np.median(self.model_perf.BW_distance[j].flatten())
            ax.plot([median_j,median_j], [0, settings.y_ub],color = cmap_l, 
                    linestyle = settings.linestyle, lw = settings.linewidth)
        ax.grid(True, alpha=0.3)
        return fig, ax
    
#%%
class NFoldsCrossValidationVisualization(PlottingTools):
    def __init__(self, settings: PlotSettingsBase, 
                 save_fig=False, save_format = 'pdf'):
        """
        Visualization class for N-fold cross-validation analysis of model fitting.

        Inherits from PlottingTools and provides methods for plotting negative 
        log likelihood (nLL) values as a function of hyperparameters (such as 
        decay rate and variance scale).

        """
        super().__init__(settings, save_fig, save_format)
        
    def plot_nLL_varying_hyper_param(self, hyper_param, nLL_training_avg, nLL_test_avg,
                                    nLL_training_CI, nLL_test_CI, total_folds,
                                    settings: PltVaryingHyperParamSettings, ax = None):
        """
        Plot the mean negative log likelihood (nLL) across N-fold cross-validation 
        folds as a function of the hyperparameter, along with confidence intervals.

        Parameters
        ----------
        hyper_param : np.ndarray of shape (M,)
            Array of decay rates or variance scales used for model fitting.
        nLL_training_avg : np.ndarray of shape (M,)
            Mean training nLL across all folds.
        nLL_test_avg : np.ndarray of shape (M,)
            Mean test nLL across all folds.
        nLL_training_CI : np.ndarray of shape (2, M) or a list of two np.arrays
            Lower and upper confidence bounds for training nLL.
        nLL_test_CI : np.ndarray of shape (2, M) or a list of two np.arrays
            Lower and upper confidence bounds for test nLL.
        total_folds : int
            Total number of cross-validation folds.
        settings : PltVaryingHyperParamSettings
            Plot-specific settings for colors, markers, labels, etc.
        ax : matplotlib.axes.Axes or None
            Optional matplotlib axis to plot on. If None, a new figure and axis are created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax : matplotlib.axes.Axes
            The axis on which the plot was drawn.
        """

        # Create new figure and axis if none provided
        if ax is None:
            fig, ax = plt.subplots(figsize=settings.fig_size, dpi=settings.dpi)
        else:
            fig = ax.figure

        # Plot average training/test nLL across folds
        ax.plot(hyper_param, nLL_training_avg, color=settings.training_color,
                marker=settings.marker, markersize=settings.ms,
                label=f'Mean across {total_folds} folds (training set)')
        
        ax.plot(hyper_param, nLL_test_avg, color=settings.test_color,
                marker=settings.marker, markersize=settings.ms,
                label=f'Mean across {total_folds} folds (test set)')

        # Confidence interval shading for training
        ax.fill_between(hyper_param, *nLL_training_CI, color=settings.training_color,
                        alpha=settings.training_fill_alpha,
                        label=f'{settings.label_CI} (training set)', 
                        linestyle=settings.ls_training)

        # Confidence interval shading for test
        ax.fill_between(hyper_param, *nLL_test_CI, color=settings.test_color,
                        alpha=settings.test_fill_alpha,
                        label=f'{settings.label_CI} (test set)', 
                        linestyle=settings.ls_test)

        # Axis labels and grid
        ax.grid(True, alpha=settings.grid_alpha)
        if settings.xticks is not None: ax.set_xticks(settings.xticks)
        if settings.xticklabels is not None: ax.set_xticklabels(settings.xticklabels)
        ax.set_xlabel(settings.xlabel)
        ax.set_ylabel(f'Mean negative log likelihood (nLL)\nof the model across {total_folds} folds')

        # Compute y-axis limits from confidence intervals
        y_lb = np.min(np.vstack((nLL_training_CI[0], nLL_test_CI[0])))
        y_ub = np.max(np.vstack((nLL_training_CI[1], nLL_test_CI[1])))
        ax.set_ylim([y_lb - 0.01, y_ub + 0.01])

        # Add legend
        ax.legend(fontsize=settings.legend_fontsize)
        fig.tight_layout()

        # Save the figure if needed
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
        
        return fig, ax
         
#%%    
class EvaluateTrialEfficiencyVisualization(PlottingTools):
    def __init__(self, settings: PlotSettingsBase, 
                 save_fig=False, save_format = 'pdf'):
        """
        Initialize the visualization class for evaluating trial efficiency.

        """
        super().__init__(settings, save_fig, save_format)
        
    def plot_BWD_varying_nTrials(self, nTrials, BWD, 
                                 settings: PltVaryingTrialsNumbersSettings,
                                 BWD_CI = None, nBtst = None, ax = None):    

        """
        Plot the Bures-Wasserstein Distance (BWD) as a function of the number of included trials.

        This function visualizes how model accuracy (measured via BWD to ground truth) 
        improves with additional data. The main curve shows BWD computed from the original dataset,
        and the shaded region reflects the variability across bootstrapped datasets.

        Parameters
        ----------
        nTrials : np.array (N elements)
            Number of included trials (ascending).
        BWD : np.array, shape: (N, ) 
            BWD values computed from the original dataset.
        BWD_CI : tuple or list of arrays
            Lower and upper confidence interval bounds for BWD from bootstraps (shape: 2 x N).
        nBtst : int
            Number of bootstrap datasets used.

        """

        # Create a new figure and axis if not provided externally
        if ax is None:
            fig, ax = plt.subplots(figsize=settings.fig_size, dpi=settings.dpi)
        else:
            fig = ax.figure
                    
        # Plot the line with line alpha
        ax.plot(nTrials, BWD, c=settings.lc, ls=settings.ls_BWD_sum, alpha=settings.linealpha)
        
        # Plot the markers with marker alpha and custom size
        ax.plot(nTrials, BWD, linestyle='None', marker='o',
                markerfacecolor=settings.lc, markeredgecolor=settings.lc,
                alpha=settings.markeralpha,
                markersize=settings.markersize,  
                label=settings.label_line)

        if BWD_CI is not None:
            # Add confidence interval region from bootstrap datasets
            ax.fill_between(nTrials, *BWD_CI, color=settings.lc, alpha=settings.fill_alpha,
                            label=f'{settings.label_CI} ({nBtst} bootstrapped datasets)',
                            linestyle=settings.ls_BWD_CI)

        # Axis labels and legend
        ax.grid(True, alpha=settings.grid_alpha)
        ax.set_xlabel(settings.xlabel)
        ax.set_ylabel(settings.ylabel)
        if settings.xlim is not None:
            ax.set_xlim(settings.xlim)
        if settings.ylim is not None:
            ax.set_ylim(settings.ylim)
        # Add legend title
        ax.legend(fontsize=settings.legend_fontsize)

        # Improve spacing/layout
        fig.tight_layout()

        # Save the figure to file if saving is enabled and fig_dir is provided
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
        
        return fig, ax
        
        