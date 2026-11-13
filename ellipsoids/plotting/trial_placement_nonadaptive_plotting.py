#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:12:36 2024

@author: fangfang
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Union
from datetime import datetime
from plotting.wishart_plotting import PlottingTools, PlotSettingsBase

#%%
@dataclass
class PlotWeibullPMFSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (2.2, 2.2)
    xticks: Union[List[float], np.ndarray] = field(default_factory=lambda: [0, 2, 4, 6])
    xlabel: str = r'Perceptual difference ($\Delta E$)'
    ylabel: str = 'Proportion correct'
    fig_name: str = field(default_factory=lambda: f'Weibull_PMF_{datetime.now().strftime("%Y%m%d_%H%M%S")}')


#%%
class TrialPlacementVisualization(PlottingTools):
    def __init__(self, sim_trial, settings: PlotSettingsBase, save_fig = False,
                 save_format = 'pdf'):

        super().__init__(settings, save_fig, save_format)
        self.sim_trial = sim_trial
        plt.rcParams['font.sans-serif'] = settings.fontstyle
        plt.rcParams['font.size'] = settings.fontsize

    def plot_WeibullPMF(self, x, settings: PlotWeibullPMFSettings, ax = None):
        a, b = self.sim_trial.sim['alpha'], self.sim_trial.sim['beta']
        g = self.sim_trial.sim['guessing_rate']
        y = self.sim_trial.WeibullFunc(x, a, b, g)
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, 1, dpi = settings.dpi,
                                    figsize = settings.fig_size)
        else:
            fig = ax.figure

        ax.plot(x, y, color = 'k')
        ax.set_xticks(settings.xticks)
        ax.set_yticks(np.round(np.array([1/3, 2/3, 1]),2))
        ax.set_xlabel(settings.xlabel)
        ax.set_ylabel(settings.ylabel)
        ax.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
        return fig, ax
