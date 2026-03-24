#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:27:16 2026

@author: fangfang
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence
import os
import jax.numpy as jnp

@dataclass
class DatasetConfig_4D:
    # required inputs
    base_dir: str
    subN: int

    # core fields
    flag_load_datafile: bool
    totalSessions: Optional[int]
    nSession: Optional[int]
    path_str: str
    plane_2D: str
    file_date: str

    # optional core fields
    adaptation_cond_str: str = ''
    bds_bruteforce: Sequence[float] = (0.0005, 0.3)
    exptCond: Optional[str] = None
    file_name: Optional[str] = None

    # grid settings
    num_grid_pts: Optional[int] = None
    num_grid_pts1: Optional[int] = None
    num_grid_pts2: Optional[int] = None
    grid_lim: float = 0.7
    grid_lim_1: float = 0.45
    grid_lim_2: float = 0.7

    # optional fields for simulated data
    coloralg: Optional[str] = None

    # derived fields
    grid: jnp.ndarray = field(init=False)
    grid_1d: Optional[jnp.ndarray] = field(init=False, default=None)
    grid_1: Optional[jnp.ndarray] = field(init=False, default=None)
    grid_2: Optional[jnp.ndarray] = field(init=False, default=None)
    str_ext_s: str = field(init=False, default='')

    def __post_init__(self):
        # session suffix
        if self.flag_load_datafile and self.totalSessions is not None and self.nSession is not None:
            self.str_ext_s = (
                f'_{self.nSession}of{self.totalSessions}sessions'
                if self.nSession < self.totalSessions else ''
            )
        else:
            self.str_ext_s = ''

        # auto-generate file name for simulated data
        if not self.flag_load_datafile and self.file_name is None:
            if self.coloralg is None:
                raise ValueError("coloralg must be provided for simulated data.")
            self.file_name = (
                f'Sim4dTask_colorDiscrimination_EAVC_6000Trials_'
                f'300_300_300_5100_sub{self.subN}_gt{self.coloralg}.pkl'
            )

        # build grid
        if self.num_grid_pts is not None:
            self.grid_1d = jnp.linspace(-self.grid_lim, self.grid_lim, self.num_grid_pts)
            self.grid = jnp.stack(
                jnp.meshgrid(self.grid_1d, self.grid_1d, indexing='ij'),
                axis=-1
            )
        elif self.num_grid_pts1 is not None and self.num_grid_pts2 is not None:
            self.grid_1 = jnp.linspace(-self.grid_lim_1, self.grid_lim_1, self.num_grid_pts1)
            self.grid_2 = jnp.linspace(-self.grid_lim_2, self.grid_lim_2, self.num_grid_pts2)
            g1, g2 = jnp.meshgrid(self.grid_1, self.grid_2, indexing='ij')
            self.grid = jnp.stack([g1, g2], axis=-1)
        else:
            raise ValueError("Grid specification is incomplete.")

    @classmethod
    def human_isoluminant(cls, base_dir: str, subN: int):
        return cls(
            base_dir=base_dir,
            subN=subN,
            flag_load_datafile=True,
            totalSessions=12,
            nSession=12,
            path_str=os.path.join(
                base_dir, 'ELPS_analysis', 'Experiment_DataFiles', 'pilot2', f'sub{subN}'
            ),
            plane_2D='Isoluminant plane',
            file_date='02242025',
            adaptation_cond_str='',
            exptCond='_4dExpt_Isoluminant plane',
            num_grid_pts=7,
            bds_bruteforce=[0.0005, 0.3],
        )

    @classmethod
    def human_varying_background(cls, base_dir: str, subN: int):
        return cls(
            base_dir=base_dir,
            subN=subN,
            flag_load_datafile=True,
            totalSessions=20,
            nSession=20,
            path_str=os.path.join(
                base_dir, 'ELPS_analysis', 'Experiment_DataFiles',
                '4D_Expt_varyingBackground', f'sub{subN}'
            ),
            plane_2D='Isoluminant plane',
            file_date='10062025',
            adaptation_cond_str='_gray',
            exptCond='_4dExpt_Isoluminant plane',
            num_grid_pts=7,
            bds_bruteforce=[0.0005, 0.3],
        )

    @classmethod
    def human_ls_isolating(cls, base_dir: str, subN: int):
        return cls(
            base_dir=base_dir,
            subN=subN,
            flag_load_datafile=True,
            totalSessions=15,
            nSession=15,
            path_str=os.path.join(
                base_dir, 'ELPS_analysis', 'Experiment_DataFiles',
                '4D_Expt_dichromats', f'sub{subN}'
            ),
            plane_2D='LSisolating plane',
            file_date='11172025',
            adaptation_cond_str='',
            exptCond='_4dExpt_LSisolating plane',
            num_grid_pts1=5,
            num_grid_pts2=7,
            bds_bruteforce=[0.0005, 0.55],
        )

    @classmethod
    def simulated_isoluminant(cls, base_dir: str, subN: int):
        return cls(
            base_dir=base_dir,
            subN=subN,
            flag_load_datafile=False,
            totalSessions=None,
            nSession=None,
            path_str=os.path.join(
                base_dir, 'META_analysis', 'Simulation_DataFiles',
                '4dTask', 'CIE'
            ),
            plane_2D='Isoluminant plane',
            file_date='02242025',
            adaptation_cond_str='',
            num_grid_pts=7,
            bds_bruteforce=[0.0005, 0.3],
            coloralg='CIE1994',
            # file_name optional → auto-generated
        )
    
    def print_summary(self):
        print("---- Dataset Config ----")
        print(f"totalSessions   : {self.totalSessions}")
        print(f"nSession        : {self.nSession}")
        print(f"path_str        : {self.path_str}")
        print(f"plane_2D        : {self.plane_2D}")
    
        if self.num_grid_pts1 is not None and self.num_grid_pts2 is not None:
            print(f"num_grid_pts1   : {self.num_grid_pts1}")
            print(f"num_grid_pts2   : {self.num_grid_pts2}")
        elif self.num_grid_pts is not None:
            print(f"num_grid_pts    : {self.num_grid_pts}")
        else:
            print("num_grid_pts    : None")
    
        print(f"bds_bruteforce  : {self.bds_bruteforce}")
        print("------------------------")