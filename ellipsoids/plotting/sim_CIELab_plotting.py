#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:15:34 2024

@author: fangfang
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, ClassVar, Sequence
from datetime import datetime
import os
from plotting.wishart_plotting import PlottingTools, PlotSettingsBase

#%%
@dataclass
class PlotPrimariesSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (2, 2)
    visualize_primaries: bool = True
    cmap: np.ndarray = field(default_factory=lambda: np.array([[178, 34, 34], 
                                                               [0, 100, 0], 
                                                               [0, 0, 128]]) / 255)
    ls: List[str] = field(default_factory=lambda: [':', '-', '--'])
    ylim: List[float] = field(default_factory=list)
    lw: float = 2
    fig_name: str = field(default_factory=lambda: f'Monitor_primaries_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
@dataclass
class PlotTconesSettings(PlotSettingsBase):
    cmap: np.ndarray = field(default_factory=lambda: np.array([[178, 34, 34], 
                                                                [0, 100, 0], 
                                                                [0, 0, 128]]) / 255)
    ylim: List[float] = field(default_factory=list)
    fig_size: Tuple[float, float] = (2,2)
    fig_name: str = field(default_factory=lambda: f'T_cones_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
@dataclass
class PlotRGBToLABSettings(PlotSettingsBase): 
    fig_size: Tuple[float, float] = (4,10)
    visualize_raw_data: bool = False
    rgb_lim: List[float] = field(default_factory=lambda: [0, 1])
    rgb_ticks: List[float] = field(default_factory=lambda: np.linspace(0.2, 0.8, 3).tolist())
    lab_viewing_angle: List[float] = field(default_factory=lambda: [30, -25])
    lab_ticks: List[float] = field(default_factory=lambda: [-60, 0, 60])
    lab_lim_margin: float = 10
    lab_xylim: Optional[List[float]] = None
    lab_zlim: Optional[List[float]] = None
    lab_scatter_ms: float = 5
    lab_scatter_alpha: float = 0.5
    lab_scatter_edgecolor: Union[str, None] = 'none'
    fig_name: str = field(default_factory=lambda: f'RGBToLAB_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
@dataclass
class Plot2DSinglePlaneSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (6, 6)
    visualize_raw_data: bool = False
    rgb_background: Optional[List[float]] = None
    plane_names: ClassVar[List[str]] = ['GB plane', 'RB plane', 'RG plane']
    plane_2D: str = 'Isoluminant plane'
    xlabel: str = 'Wishart space dimemsion 1'
    ylabel: str = 'Wishart space dimension 2'
    ref_mc: List[float] = field(default_factory=lambda: [0, 0, 0])
    ref_ms: float = 40
    ref_lw: float = 2
    ell_lc: List[float] = field(default_factory=lambda: [0, 0, 0])
    ell_ls: str = '-'
    ell_lw: float = 2
    data_m: str = 'o'
    data_alpha: float = 1.0
    data_ms: float = 20
    data_mc: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    ticks: List[float] = field(default_factory=lambda: np.linspace(0, 1, 5).tolist())
    lim: List[float] = field(default_factory=lambda: [0, 1])
    fig_name: str = field(default_factory=lambda: f'Isothreshold_contour_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
@dataclass
class Plot3DSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (8, 8)
    visualize_ref: bool = True
    visualize_ellipsoids: bool = True
    visualize_thresholdPoints: bool = False
    threshold_points: Optional[np.ndarray] = None
    ref_color: Optional[List[float]] = None
    ref_ms: float = 10
    ref_lw: float = 0.5
    surf_color: Optional[List[float]] = None
    surf_alpha: float = 0.5
    scatter_color: Optional[List[float]] = None
    scatter_alpha: float = 0.5
    scatter_ms: float = 3
    lim: List[float] = field(default_factory=lambda: [0, 1])
    ticks: List[float] = field(default_factory=lambda: np.linspace(0.2, 0.8, 3).tolist())
    view_angle: List[float] = field(default_factory=lambda: [30, -120])
    title: str = 'RGB space'
    flag_input_W: bool = False
    fig_name: str = field(default_factory=lambda: f'Isothreshold_ellipsoids_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
@dataclass
class PlotStimAtThresSettings(PlotSettingsBase):
    fig_size_for1: Tuple[float, float] = (2, 2.5)
    fig_name: str = field(default_factory=lambda: f'color_patches_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
@dataclass
class PlotDeltaESettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (3, 4)
    ylim: List[float] = field(default_factory=lambda: [-2, 30])
    marker_size: int = 200
    lw: float = 2
    lc: str = 'k'
    ylabel: str = 'Delta E'
    fig_name: str = field(default_factory=lambda: f'deltaE_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

@dataclass
class PlotLabSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (6,5)
    xlim: List[float] = field(default_factory=lambda: [-120, 120])
    ylim: List[float] = field(default_factory=lambda: [-120, 120])
    zlim: List[float] = field(default_factory=lambda: [0, 140])
    xticks: Union[Sequence[float], np.ndarray] = field(default_factory=lambda: np.linspace(-100, 100, 5))
    yticks: Union[Sequence[float], np.ndarray] = field(default_factory=lambda: np.linspace(-100, 100, 5))
    zticks: Union[Sequence[float], np.ndarray] = field(default_factory=lambda: np.linspace(0, 150, 5))
    bg_alpha: float = 0.04
    grid_alpha: float = 0.1
    scatter_ms: float = 1
    xlabel: str = 'a*'
    ylabel: str = 'b*'
    zlabel: str = 'L*'
    title: str = 'L*a*b* within monitor''s gamut'
    fig_name: str = 'L*a*b*_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

#%%
class CIELabVisualization(PlottingTools):
    def __init__(self, sim_CIE, settings: PlotSettingsBase, save_fig = False, save_format = 'pdf'):
        super().__init__(settings, save_fig, save_format)
        self.sim_CIE = sim_CIE
    
    def plot_primaries(self, settings: PlotPrimariesSettings, rgb = None, ax = None):
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, 1, dpi = settings.dpi, figsize = settings.figsize)
        else:
            fig = ax.figure
            
        if self.pltP['visualize_primaries']:
            for i in range(self.sim_CIE.nPlanes):
                ax.plot(self.sim_CIE.B_MONITOR[:,i],
                        c = settings.cmap[i],
                        lw = settings.lw)
        if rgb is not None:
            for j in range(rgb.shape[1]):
                spd_j = self.sim_CIE.B_MONITOR @ rgb[:,j]
                ax.plot(spd_j, c= 'k', 
                        linestyle = settings.ls[j],
                        lw = settings.lw)
            if len(settings.ylim) != 0:
                ax.set_ylim(settings.ylim)
            ax.set_xticks([])
            ax.set_yticks([])
        # Save the plot with bbox_inches='tight' to ensure labels are not cropped
        plt.show()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, os.path.join(settings.fig_dir, settings.fig_name))
        return fig, ax
    
    def plot_Tcones(self, settings: PlotTconesSettings, ax = None):
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, 1, dpi = settings.dpi, figsize = settings.fig_size)
        else:
            fig = ax.figure

        for i in range(self.sim_CIE.nPlanes):
            ax.plot(self.sim_CIE.T_CONES[i], c = settings.cmap[i])
            if len(settings.ylim) != 0:
                ax.set_ylim(settings.ylim)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, os.path.join(settings.fig_dir, settings.fig_name))
        return fig, ax
    
    def plot_RGB_to_LAB(self, ref_rgb, ref_lab, settings: PlotRGBToLABSettings, ax = None):
        self.ndims = 3
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(2, 1, dpi = settings.dpi, 
                                    figsize = settings.figsize, 
                                    subplot_kw={'projection': '3d'})
        else:
            fig = ax.figure
            
        #colormap
        cmap_rgb = np.moveaxis(ref_rgb, 0, -1)
        colors_flat = cmap_rgb.reshape(-1, 3)
            
        #RGB space
        r_slc, g_slc, b_slc = ref_rgb
        r_slc_f = r_slc.flatten()
        g_slc_f = g_slc.flatten()
        b_slc_f = b_slc.flatten()
        ax[0].scatter(r_slc_f, g_slc_f, b_slc_f, c = colors_flat)
        self._update_axes_limits(ax[0], lim = settings.rgb_lim)
        self._update_axes_labels(ax[0], settings.rgb_ticks, settings.rgb_ticks,nsteps =1)
        self._configure_labels_and_title(ax, title='RGB cube')
        ax[0].set_box_aspect([1,1,1])

        #CIELAB SPACE
        L_slc, A_slc, B_slc = ref_lab
        L_slc_f = L_slc.flatten()
        A_slc_f = A_slc.flatten()
        B_slc_f = B_slc.flatten()
        ax[1].scatter(A_slc_f, B_slc_f, L_slc_f, c = colors_flat, 
                   marker = 'o',s= settings.lab_scatter_ms,
                   alpha = settings.lab_scatter_alpha,
                   edgecolor= settings.lab_scatter_edgecolor)
        
        if settings.lab_xylim is None: 
            xymin = np.min([np.min(A_slc_f), np.min(B_slc_f)])
            xymax = np.max([np.max(A_slc_f), np.max(B_slc_f)])
            xylim = np.array([-1,1])* np.max([np.abs(xymin), xymax]) +\
                np.array([-1,1])*settings.lab_lim_margin
        else:
            xylim = settings.lab_xylim
            
        if settings.lab_zlim is None:
            zmin = np.min(L_slc_f)
            zmax = np.max(L_slc_f)
            zlim = np.array([zmin, zmax]) + np.array([-1,1])*settings.lab_lim_margin
        else:
            zlim = settings.lab_zlim
        # Project the surface onto the XY plane (Z = min(Z))
        ax[1].scatter(A_slc_f, B_slc_f, zlim[0] * np.ones_like(B_slc_f),
                        c=colors_flat, edgecolor= settings.lab_scatter_edgecolor,  
                        marker = 'o',s = 2, alpha=0.05)

        # Project the surface onto the XZ plane (Y = max(Y))
        ax[1].scatter(A_slc_f, xylim[1] * np.ones_like(B_slc_f), L_slc_f,
                        c=colors_flat, edgecolor=settings.lab_scatter_edgecolor, 
                        marker = 'o',s= 4,  alpha=0.02)

        # Project the surface onto the YZ plane (X = min(X))
        ax[1].scatter(xylim[0] * np.ones_like(A_slc_f), B_slc_f, L_slc_f,
                        c=colors_flat, edgecolor=settings.lab_scatter_edgecolor, 
                        marker = 'o',s = 4,  alpha=0.02)

        print(xylim)
        ax[1].set_xlim(xylim); ax[1].set_ylim(xylim); ax[1].set_zlim(zlim)
        ax[1].set_xlabel('a'); ax[1].set_ylabel('b'); ax[1].set_zlabel('L')
        ax[1].set_xticks(settings.lab_ticks); ax[1].set_yticks(settings.lab_ticks)
        ax[1].set_box_aspect([1,1,1])
        ax[1].set_title('CIELab space')
        ax[1].view_init(*settings.lab_viewing_angle)
        # Save the plot with bbox_inches='tight' to ensure labels are not cropped
        plt.show()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, os.path.join(settings.fig_dir, settings.fig_name))

        return fig, ax

    def plot_2D_all_planes(self, grid_est, fitEllipse, settings: Plot2DSinglePlaneSettings,
                           rawData=None, ax=None):
        """
        Generate multiple subplots (one for each plane) to visualize isothreshold contours.
        Calls `plot_2D_single_plane` for each plane and modifies plot parameters as needed.
        """
        num_grid_pts_x, num_grid_pts_y = grid_est.shape[0:2]
    
        # Create a figure with multiple subplots if `ax` is not provided.
        if ax is None:
            fig, axes = plt.subplots(1, self.sim_CIE.nPlanes, figsize=(20, 6), 
                                     dpi=settings.dpi)
        else:
            fig = ax.figure
            axes = ax
    
        # Generate plots for each plane
        for p in range(self.sim_CIE.nPlanes):
            if settings.rgb_background is not None:
                bg = settings.rgb_background[p]
            else:
                bg = None
            settings.plane_2D = settings.plane_names[p]
            self.plot_2D_single_plane(grid_est, fitEllipse[p], settings = settings, 
                                      rawData = rawData[p], 
                                      ax = axes[p], rgb_background = bg)
    
        # Save and display the figure
        plt.show()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, os.path.join(settings.fig_dir, settings.fig_name))
    
        return fig, axes
    
    
    def plot_2D_single_plane(self, grid_est, fitEllipse, settings, 
                             rawData=None, ax=None, rgb_background = None):
        """
        Generate a single plot for a specific plane index.
        All default settings are defined here, but can be overridden via kwargs.
        """
        num_grid_pts_x, num_grid_pts_y = grid_est.shape[0:2]
        
        # Determine if `ell_lc` is a single color or a color matrix
        ell_lc_is_matrix = isinstance(settings.ell_lc, np.ndarray) and \
            settings.ell_lc.shape == (num_grid_pts_x, num_grid_pts_y, 3)
        ref_mc_is_matrix = isinstance(settings.ref_mc, np.ndarray) and \
            settings.ref_mc.shape == (num_grid_pts_x, num_grid_pts_y, 3)
        data_mc_is_matrix = isinstance(settings.data_mc, np.ndarray) and \
            settings.data_mc.shape == (num_grid_pts_x, num_grid_pts_y, 3)
    
        if ax is None:
            fig, ax = plt.subplots(figsize=settings.fig_size, dpi=settings.dpi)
        else:
            fig = ax.figure
    
        # Set background if provided
        if rgb_background is not None:
            ax.imshow(rgb_background, extent=[0, 1, 0, 1], origin='lower')
    
        # Plot reference locations, ellipses, and threshold data
        for i in range(num_grid_pts_x):
            for j in range(num_grid_pts_y):
                # Reference location
                ref_color = settings.ref_mc[i, j] if ref_mc_is_matrix else settings.ref_mc
                ax.scatter(*grid_est[i, j], s=settings.ref_ms, c= ref_color,
                           marker='+', linewidth=settings.ref_lw)
    
                # Ellipses
                ellipse_color = settings.ell_lc[i, j] if ell_lc_is_matrix else settings.ell_lc
                ax.plot(*fitEllipse[i, j], linestyle=settings.ell_ls,
                        color=ellipse_color, linewidth=settings.ell_lw)
    
                # Threshold points
                if settings.visualize_raw_data and rawData is not None:
                    data_color = settings.data_mc[i, j] if data_mc_is_matrix else settings.data_mc
                    ax.scatter(*rawData[i, j], marker=settings.data_m,
                               color= data_color, s=settings.data_ms,
                               alpha= settings.data_alpha)
    
        # Configure plot limits, ticks, and labels
        self._update_axes_limits(ax, lim= settings.lim)
        self._update_axes_labels(ax, settings.ticks, settings.ticks, nsteps=1)
        self._configure_labels_and_title(ax, settings.plane_2D)
    
        return fig, ax

    #%%
    def plot_3D(self, grid_est, fitEllipsoid, settings: Plot3DSettings, 
                nTheta = 200, nPhi = 100, ax = None):        
        self.ndims = 3
        nRef = grid_est.shape[0]
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig = plt.figure(figsize = settings.fig_size,dpi = settings.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        ax.set_box_aspect([1,1,1])
        for i in range(nRef):
            cmap_i =grid_est[i]
            if settings.flag_input_W: cmap_i = (cmap_i + 1)/2
            # if settings.ref_color is not None: 
            #     ref_color = settings.ref_color
            # else: ref_color = cmap_i
            # ax.scatter(cmap_i[0],cmap_i[1],cmap_i[2], s =settings.ref_ms, 
            #            c = ref_color, marker='+',
            #            linewidth= settings.ref_lw)
        
            if settings.visualize_ellipsoids:
                if settings.surf_color is not None: 
                    surf_color = settings.surf_color
                else: surf_color = cmap_i
                ell_i = fitEllipsoid[i]
                if nPhi*nTheta != ell_i.shape[-1]:
                    raise ValueError('The size of grid points (nPhi x nTheta) does not'+\
                                     ' equal to the default values! Please pass the correct'+\
                                     ' nPhi and nTheta.')
                ell_i_x = ell_i[0].reshape(nPhi, nTheta)
                ell_i_y = ell_i[1].reshape(nPhi, nTheta)
                ell_i_z = ell_i[2].reshape(nPhi, nTheta)
                ax.plot_surface(ell_i_x, ell_i_y, ell_i_z,
                                color= np.array(surf_color), edgecolor='none', 
                                alpha= settings.surf_alpha)

            if settings.visualize_thresholdPoints and settings.threshold_points is not None:
                if settings.scatter_color is not None: 
                    scatter_color = settings.scatter_color
                else: scatter_color = cmap_i
                tp = settings.threshold_points[i]
                tp_x, tp_y, tp_z = tp[:,:,0], tp[:,:,1], tp[:,:,2]
                tp_x_f = tp_x.flatten()
                tp_y_f = tp_y.flatten()
                tp_z_f = tp_z.flatten()
                ax.scatter(tp_x_f, tp_y_f, tp_z_f,
                           s=settings.scatter_ms, 
                           c= scatter_color, edgecolor = 'none',
                           alpha= settings.scatter_alpha)
        self._update_axes_limits(ax, lim = settings.lim, ndims = self.ndims)
        self._configure_labels_and_title(ax, ndims= self.ndims, title = settings.title)
        self._update_axes_labels(ax, settings.ticks, settings.ticks,
                                 ndims = self.ndims, nsteps = 1)
        ax.view_init(elev=settings.view_angle[0], azim=settings.view_angle[1])   # Adjust viewing angle for better visualization
        plt.show()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, os.path.join(settings.fig_dir, settings.fig_name))
        return fig, ax
           
    #%%
    def plot_Lab_inGamut(self, Lab, rgb, settings: PlotLabSettings, ax=None):
        """
        Plot CIELab points that are within the monitor gamut in a 3D axes.
    
        Parameters
        ----------
        Lab : np.ndarray, shape (N, 3)
            Points in CIELab order [L*, a*, b*] for each of N samples.
        rgb : np.ndarray, shape (N, 3)
            Per-point RGB colors used to color the markers (expected in [0, 1]).
        settings : PlotLabSettings
            Plot configuration (limits, ticks, labels, title, sizes, paths, etc.).
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D or None, optional
            Existing 3D axes to draw into. If None, a new figure/axes is created.

        """
    
        # Reorder Lab → abL so that: x=a*, y=b*, z=L* (shape still (N, 3)).
        abL = Lab[:, [1, 2, 0]]
    
        # Create a new figure/axes if none provided; otherwise reuse the passed axes.
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'},
                                   figsize=settings.fig_size)
        else:
            fig = ax.figure
    
        # Scatter all points at once; RGB colors per point, no depth shading for color fidelity.
        ax.scatter(*abL.T, c=rgb, s=settings.scatter_ms, depthshade=False)
    
        # Axis labels and bounds.
        ax.set_xlabel(settings.xlabel)
        ax.set_ylabel(settings.ylabel)
        ax.set_zlabel(settings.zlabel)
        ax.set_xlim(settings.xlim)
        ax.set_ylim(settings.ylim)
        ax.set_zlim(settings.zlim)
    
        # Title, aspect, grid, and background styling.
        ax.set_title(settings.title, fontsize=settings.fontsize)
        ax.set_box_aspect((1, 1, 1))  # equal aspect for the 3D box
        ax.grid(True, alpha=settings.grid_alpha)
    
        # Set pane (back wall) colors with desired alpha (RGBA).
        # Note: (0,0,0,alpha) is translucent black; adjust in settings if you prefer light panes.
        pane_rgba = (0, 0, 0, settings.bg_alpha)
        ax.xaxis.set_pane_color(pane_rgba)
        ax.yaxis.set_pane_color(pane_rgba)
        ax.zaxis.set_pane_color(pane_rgba)
    
        # Ticks (ensure settings.xticks/yticks/zticks are sequences or ndarrays).
        ax.set_xticks(settings.xticks)
        ax.set_yticks(settings.yticks)
        ax.set_zticks(settings.zticks)
    
        # Layout and optional save.
        fig.tight_layout()
        plt.show()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, os.path.join(settings.fig_dir, settings.fig_name))
    
        return fig, ax
    
    def plot_Lab_inGamut_flattened(self, Lab, rgb, fixed_dim, settings: Plot3DSettings, ax=None):
        """
        Plot a 2D “slice” of CIELab by fixing one dimension and drawing the other two.
    
        Parameters
        ----------
        Lab : np.ndarray, shape (N, 3)
            Points in CIELab order [L*, a*, b*] for N samples.
        rgb : np.ndarray, shape (N, 3)
            Per-point RGB colors (expected in [0, 1]) used to color markers.
        fixed_dim : int
            Which Lab dimension is held constant and therefore *not* plotted:
                0 → fix L*, plot (a*, b*)
                1 → fix a*, plot (L*, b*)
                2 → fix b*, plot (L*, a*)
        settings : Plot3DSettings
            Plot config (limits, ticks, labels, title, figure size, etc.).
            Note: although named “3D”, we use only the relevant fields here for 2D.
        ax : matplotlib.axes.Axes or None
            Existing 2D axes to draw into; if None, a new figure/axes is created.
    
        Notes
        -----
        - This is the 2D counterpart to `plot_Lab_inGamut`: it collapses one Lab axis.
        - Axis labels/limits/ticks are chosen based on `fixed_dim` so they remain
          consistent with your Settings object (xlim/ylim for a*, b*; zlim for L*).
          
        """
    
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(figsize=settings.fig_size)
        else:
            fig = ax.figure
    
        # Determine which two dimensions to draw after removing the fixed one.
        # Example: fixed_dim=0 → varied_dims=[1,2] → (a*, b*)
        varied_dims = list(range(3))
        varied_dims.pop(fixed_dim)
    
        # Scatter the slice using the two varying coordinates; color by per-point RGB.
        ax.scatter(Lab[:, varied_dims[0]], Lab[:, varied_dims[1]],
                   c=rgb, s=settings.scatter_ms)
    
        # Keep geometry faithful (equal units on both axes).
        ax.set_aspect('equal', adjustable='box')
    
        # Choose appropriate labels/limits/ticks depending on which axis is fixed.
        if fixed_dim == 0:  # fix L*, plot (a*, b*)
            xlbl, ylbl = settings.xlabel, settings.ylabel
            ax.set_xlim(settings.xlim); ax.set_ylim(settings.ylim)
            ax.set_xticks(settings.xticks); ax.set_yticks(settings.yticks)
        elif fixed_dim == 1:  # fix a*, plot (L*, b*)
            xlbl, ylbl = settings.zlabel, settings.ylabel
            ax.set_xlim(settings.zlim); ax.set_ylim(settings.ylim)
            ax.set_xticks(settings.zticks); ax.set_yticks(settings.yticks)
        else:  # fixed_dim == 2: fix b*, plot (L*, a*)
            xlbl, ylbl = settings.zlabel, settings.xlabel
            ax.set_xlim(settings.zlim); ax.set_ylim(settings.xlim)
            ax.set_xticks(settings.zticks); ax.set_yticks(settings.xticks)
    
        # Apply labels/title and light grid.
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title(settings.title, fontsize=settings.fontsize)
        ax.grid(True, alpha=0.3)
    
        # Tight layout to avoid clipped labels.
        fig.tight_layout()
    
        # Optional save (if enabled on the class and a directory is provided).
        plt.show()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, os.path.join(settings.fig_dir, settings.fig_name))
    
        return fig, ax
    
    #%%
    @staticmethod
    def visualize_stimuli_at_thres(s_rgb, settings: PlotStimAtThresSettings,
                                   ax = None, label_rgb = True, save_fig = False):
        
        """
        Visualizes a set of stimuli at threshold by displaying each as a square 
        filled with its corresponding RGB color.
        
        Parameters:
        - s_rgb (numpy.ndarray): Array of shape (3, n), where each column represents an RGB color.
        - ax (matplotlib.axes.Axes, optional): Array of matplotlib axes. 
            If None, a new figure and axes are created.
        - label_rgb (bool, optional): Whether to label each square with its RGB values. 
            Default is True.
        
        """
        
        # Determine the number of stimuli (i.e., the number of RGB values) to display.
        n = s_rgb.shape[1]
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, n,figsize=(n*settings.fig_size_for1[0], 
                                                  settings.fig_size_for1[1]), 
                                    dpi= settings.dpi)
        else:
            fig = ax.figure        
            
        # Ensure ax is iterable (in case n == 1 and ax is a single object)
        if n == 1: ax = [ax]
            
        # Loop through each RGB value and create a corresponding color square.
        for i in range(n):
            # Create a 1x1 square filled with the ith RGB color.
            color_square = np.full((1, 1, 3), s_rgb[:,i])
            # Display the color square using imshow on the ith axis.
            ax[i].imshow(color_square, extent = [0,1,0,1])
            # If label_rgb is True, add the RGB values as the title of the square.
            if label_rgb:
                # Convert the RGB values to integers and round them
                rgb_int = np.round(color_square[0, 0, :] * 255).astype(int)
                # Set the title with the RGB values
                ax[i].set_title(f'[R, G, B]: [{rgb_int[0]}, {rgb_int[1]}, {rgb_int[2]}]', 
                                fontsize = settings.fontsize)
            # Remove the axes for better visualization 
            ax[i].axis('off')
        # Show the figure after all subplots have been drawn
        if settings.fig_dir and save_fig:
            plt.savefig(os.path.join(settings.fig_dir, settings.fig_name + '.pdf'))
        # Display the plot
        plt.show()
            
        return fig, ax
        
    @staticmethod
    def plot_deltaE(deltaE, comp_rgb, settings: PlotDeltaESettings, ax = None, 
                    save_fig = False):
        
        if ax is None:
            fig, ax =  plt.subplots(1, 1,figsize= settings.fig_size, dpi= settings.dpi)
        else:
            fig = ax.figure         
        x = np.array(list(range(len(deltaE))))
        ax.plot(x, deltaE, lw= settings.lw, c = settings.lc)
        ax.scatter(x, deltaE, c = comp_rgb, s = settings.marker_size)
        ax.set_ylim(settings.ylim)
        ax.set_xticks([])
        ax.set_ylabel(settings.ylabel)
        ax.set_yticks(np.linspace(0, settings.ylim[1],5))
        # Show the figure after all subplots have been drawn
        if settings.fig_dir and save_fig:
            plt.savefig(settings.fig_dir + settings.fig_name)
        plt.show()
        return fig, ax
        
        
        