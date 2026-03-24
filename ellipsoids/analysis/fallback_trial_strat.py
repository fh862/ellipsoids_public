#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:05:52 2025

@author: fangfang
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.colors import BoundaryNorm
import re
import pandas as pd
from analysis.utils_load import find_files_with_prefix

#%%
class TrialSequenceAnalyzer:
    def __init__(self, expt_file_info, interleaved_trial_sequence,
                 nBlocks, nTrials_perBlock):
        """
        Initializes the class with a given interleaved trial sequence.

        Args:
            sim_interleaved_trial_sequence: Object containing trial sequences.
        """
        self.expt_file_info = expt_file_info
        self.sim_sequence = interleaved_trial_sequence
        self.nBlocks = nBlocks
        self.nTrials_perBlock = nTrials_perBlock
        self.session_today = list(self.expt_file_info.session_data.keys())[-1]

    def extract_original_final_indices(self):
        """
        Retrieves trial indices for MOCS and AEPsych trials from both initial and final sequences,
        then computes the shift in trial positions.
        """
        # Retrieve indices for MOCS trials
        self.indices_MOCS_initial = self.sim_sequence.indices_trial_type(
            self.sim_sequence.original_sequence[0], 'MOCS')
        self.indices_MOCS_final = self.sim_sequence.indices_trial_type(
            self.sim_sequence.final_sequence[0], 'MOCS')

        # Retrieve indices for AEPsych trials
        self.indices_AEPsych_initial = self.sim_sequence.indices_trial_type(
            self.sim_sequence.original_sequence[0], 'AEPsych')
        self.indices_AEPsych_final = self.sim_sequence.indices_trial_type(
            self.sim_sequence.final_sequence[0], 'AEPsych')

        # Compute shift in trial positions
        self.MOCS_indices_shift = np.array(self.indices_MOCS_initial) - \
            np.array(self.indices_MOCS_final)
        self.AEPsych_indices_shift = np.array(self.indices_AEPsych_initial) - \
            np.array(self.indices_AEPsych_final)
            
    def retrieve_list_trial_order(self, file_path):
        # Clean up the data
        filename_csv_start_str = f"Unity_trial_data_sub{self.expt_file_info.subject_id}_"+\
            f"{self.expt_file_info.subject_init}_session{self.session_today}"
        fullpath_csv = find_files_with_prefix(file_path, filename_csv_start_str)
        self.list_order_all_trial_types = pd.read_csv(fullpath_csv).iloc[:, 1].tolist()
            
    def parse_trials(self):  
        trial_names = []
        trial_ids = []
        self.trial_types_dict = {"aepsych": 0, "mocs": 1, "sobol": 2}
    
        for trial in self.list_order_all_trial_types:
            # Extract everything after 'Trial_<number>_'
            match = re.match(r"Trial_\d+_(.*)", trial.strip())
            if match:
                trial_name = match.group(1)
                trial_names.append(trial_name)
    
                # Determine code from type_map
                trial_type = trial_name.split("_")[0].lower()
                code = self.trial_types_dict.get(trial_type, -1)  # -1 if unknown
                trial_ids.append(code)
            else:
                trial_names.append(None)
                trial_ids.append(-1)
        self.list_order_trial_names = trial_names
        self.list_order_trial_ids = trial_ids
        
    def _pad_trial_ids_to_full_panels(self, n_perplot):
        """Pad trial trial_ids to fill the last subplot row with NaNs."""
        trial_ids = np.asarray(self.list_order_trial_ids)
        total = len(trial_ids)
        n_subplots = max(1, int(np.ceil(total / n_perplot)))
        pad_width = n_perplot * n_subplots - total
        # cast to float so NaN is possible
        trial_ids_f = trial_ids.astype(float)
        if pad_width > 0:
            trial_ids_f = np.pad(trial_ids_f, (0, pad_width), constant_values=np.nan)
        return trial_ids_f, n_subplots

    def plot_all_trial_type_order(self, nTrials_perplot=100):    
        # cast to float so we can pad with NaN
        trial_ids_pad, n_subplots = self._pad_trial_ids_to_full_panels(nTrials_perplot)
        
        fig, axes = plt.subplots(n_subplots, 1, figsize=(13, 1.7 * n_subplots + 1), dpi=1024)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
    
        plt.rcParams['font.sans-serif'] = ['Arial']
    
        # Define discrete colormap for 0, 1, 2
        cmap_all = plt.get_cmap('Accent')  # the second arg forces N discrete colors
        cmap = plt.matplotlib.colors.ListedColormap([cmap_all(0), cmap_all(7), cmap_all(3)])  # pick 3 discrete colors
        bounds = np.arange(len(self.trial_types_dict.keys())+1) - 0.5
        norm = BoundaryNorm(bounds, cmap.N)
    
        last_im = None
        for n in range(n_subplots):
            lb = n * nTrials_perplot
            ub = (n + 1) * nTrials_perplot
            chunk = trial_ids_pad[lb:ub].astype(float)  # cast so NaN is allowed
    
            img = axes[n].imshow(chunk[np.newaxis, :], cmap=cmap, norm=norm, aspect='auto')
            last_im = img
    
            axes[n].set_yticks([])
            axes[n].set_xticks([0, len(chunk) - 1] if len(chunk) > 1 else [0])
            axes[n].set_xticklabels([lb + 1, ub] if len(chunk) > 1 else [lb + 1])
    
            axes[n].set_title(f"Trial types [{lb + 1}–{ub}]")
    
        # Horizontal colorbar spanning full figure width
        if last_im is not None:
            cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
            cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal',
                                ticks=[0, 1, 2])
            cbar.ax.set_xticklabels(['AEPsych', 'MOCS', 'Sobol'])
            cbar.set_label('Trial Type')
    
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()
        

    def plot_shift_figure(self, cmap, norm, n_subplots, n_slc_plot, ylabel, 
                          colorbar_label, save_fig = False, output_figDir = None):
        """
        Plots the shift in presentation order for trials and saves the figure.

        Args:
            indices_initial_sequence (list): Initial sequence indices for the trials.
            indices_final_sequence (list): Final sequence indices for the trials.
            cmap (Colormap): Colormap to use for visualizing shifts.
            norm (Normalize): Normalization object for mapping shift values to colors.
            n_subplots (int): Number of subplots in the figure.
            n_slc_plot (int): Number of trials per slice.
            ylabel (str): Label for the y-axis.
            colorbar_label (str): Label for the colorbar.
        """
        # Create the figure and subplots
        fig, axes = plt.subplots(n_subplots, 1, figsize=(13, 1.7 * n_subplots), dpi=1024)
        plt.rcParams['font.sans-serif'] = ['Arial']

        for n in range(n_subplots):
            trial_lb_n = n * n_slc_plot // 2
            trial_ub_n = (n + 1) * n_slc_plot // 2
            
            # Slice the data for the current subplot
            i_initial = self.indices_MOCS_initial[trial_lb_n:trial_ub_n]
            i_final = self.indices_MOCS_final[trial_lb_n:trial_ub_n]

            # Plot shifts
            for m in range(len(i_initial)):
                color = cmap(norm(i_final[m] - i_initial[m]))
                axes[n].plot([i_initial[m], i_final[m]], [0, 1], ls='-', color=color)

            # Set axis limits, ticks, and labels
            axes[n].set_ylim([0, 1])
            axes[n].set_yticks([0, 1])
            axes[n].set_xlim([trial_lb_n * 2, trial_ub_n * 2])
            axes[n].set_yticklabels(['Original', 'Final'])

            if n == n_subplots // 2:  # Add ylabel to the middle subplot
                axes[n].set_ylabel(ylabel)

        axes[-1].set_xlabel('Trial Number')  # Add xlabel to the last subplot

        # Add a colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, label=colorbar_label, orientation='horizontal', pad=0.2, aspect=80)

        # Adjust layout to avoid overlap
        fig.subplots_adjust(left=0.05, right=0.95, top=0.975, bottom=0.28, hspace=0.5)

        # Show the figure
        plt.show()
        
        if (save_fig == True) and (output_figDir is not None):
            # Save figure
            fig.savefig(output_figDir + '/Shift_MOCS_PresentationOrder_'+\
                        f'simulatedTask_wMOCS_AEPsych_sub{self.expt_file_info.subject_id}_'+\
                        f'session{self.expt_file_info.session_num}.pdf', bbox_inches='tight')
        
    def _prepare_binary_sequence(self, slc = 'initial'):
        """
        Creates a binary sequence indicating MOCS and AEPsych trials.

        Args:
            indices_MOCS (list): Indices of MOCS trials.
            indices_AEPsych (list): Indices of AEPsych trials.

        Returns:
            np.ndarray: Binary matrix reshaped into (nSessions, nTrials_perBlock).
        """
        binary_seq = np.full((int(self.nTrials_perBlock * self.nBlocks), 1), np.nan)
        if slc == 'initial':
            binary_seq[self.indices_MOCS_initial] = 1
            binary_seq[self.indices_AEPsych_initial] = 0
        elif slc == 'final':
            binary_seq[self.indices_MOCS_final] = 1
            binary_seq[self.indices_AEPsych_final] = 0            
        return np.reshape(binary_seq, (self.nBlocks, self.nTrials_perBlock))

    def _plot_heatmap(self, ax, data, title, yticklbl, xticklbl=None):
        """
        Plots a heatmap for trial sequences.

        Args:
            ax (AxesSubplot): Matplotlib axis object.
            data (ndarray): Binary sequence data to plot.
            title (str): Title of the subplot.
            yticklbl (list): Y-axis tick labels.
            xticklbl (list, optional): X-axis tick labels.
        """
        im = ax.imshow(data, cmap="Accent")
        ax.set_title(title)
        ax.set_yticks(np.linspace(0, self.nBlocks, 5))
        ax.set_yticklabels(yticklbl)
        ax.set_ylabel('Session number')
        if xticklbl is not None:
            ax.set_xticks(np.linspace(0, self.nTrials_perBlock, 9))
            ax.set_xticklabels(xticklbl)
            ax.set_xlabel('Trial number')
        else:
            ax.set_xticks([])
        return im

    def plot_presentation_order_heatmap(self, save_fig = False, output_figDir = None):
        """
        Generates heatmaps showing the initial and final presentation order of trials.

        Args:
            output_figDir_sims (str): Directory to save the output figure.
            SUBJ (str): Subject ID for saving the file.
        """

        # Create binary sequences
        binary_initial_seq = self._prepare_binary_sequence(slc = 'initial')
        binary_final_seq = self._prepare_binary_sequence(slc = 'final')

        # Compute the number of MOCS trials per session
        initial_nMOCS_perSession = np.sum(binary_initial_seq, axis=1)
        final_nMOCS_perSession = np.sum(binary_final_seq, axis=1)

        # Create a GridSpec layout
        fig4 = plt.figure(figsize=(14, 4), dpi=1024)
        gs = GridSpec(2, 2, width_ratios=[12, 1], height_ratios=[1, 1], figure=fig4, wspace=0.1)

        # Main heatmap axes
        ax_main1 = fig4.add_subplot(gs[0, 0])
        ax_main2 = fig4.add_subplot(gs[1, 0])

        # Side plot axes
        ax_side1 = fig4.add_subplot(gs[0, 1])
        ax_side2 = fig4.add_subplot(gs[1, 1])

        ax_side1.plot(initial_nMOCS_perSession - self.nTrials_perBlock // 2,
                      list(range(self.nBlocks))[::-1], color='k')
        ax_side2.plot(final_nMOCS_perSession - self.nTrials_perBlock // 2,
                      list(range(self.nBlocks))[::-1], color='k')

        ax_side1.set_title('Deviation')
        ax_side1.set_xlabel('')
        ax_side1.set_xticks([])
        ax_side2.set_xticks([-10, 0, 10])
        ax_side1.set_yticks([])
        ax_side2.set_yticks([])

        # Generate tick labels
        xticklbl = [int(x + 1) for x in np.linspace(0, self.nTrials_perBlock - 1, 9)]
        yticklbl = [int(y + 1) for y in np.linspace(0, self.nBlocks - 1, 5)]

        # Plot Initial and Final sequences
        self._plot_heatmap(ax_main1, binary_initial_seq, 
                           'Initially pre-generated trial sequence (Gray: MOCS)', yticklbl)
        self._plot_heatmap(ax_main2, binary_final_seq, 
                           'Final trial sequence after dynamic updates (Gray: MOCS)', yticklbl, xticklbl)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

        if (save_fig == True) and (output_figDir is not None):
            # Save figure
            fig4.savefig(output_figDir + '/Pattern_PresentationOrder_'+\
                         f'simulatedTask_wMOCS_AEPsych_sub{self.expt_file_info.subject_id}_'+\
                         f'session{self.session_today}.pdf', bbox_inches='tight')