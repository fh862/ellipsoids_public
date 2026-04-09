#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:27:16 2026

@author: fangfang
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence
import os
import re
import jax.numpy as jnp
import numpy as np
from dconfig.mocs_config_mixin import MOCSConfigMixin

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
    opt_total_steps: int = 1500
    exptCond: Optional[str] = None
    file_name: Optional[str] = None

    # grid settings
    num_grid_pts: Optional[int] = None
    num_grid_pts1: Optional[int] = None
    num_grid_pts2: Optional[int] = None
    grid_lim: float = 0.7
    grid_lim_1: float = 0.45
    grid_lim_2: float = 0.7
    
    # stimulus dimension and rgb of adapting points
    stim_dims: int = 2
    psyfield_dims: int = 4
    bg_rgb: Optional[jnp.ndarray] = None

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
                jnp.meshgrid(self.grid_1d, self.grid_1d),
                axis=-1
            )
        elif self.num_grid_pts1 is not None and self.num_grid_pts2 is not None:
            self.grid_1 = jnp.linspace(-self.grid_lim_1, self.grid_lim_1, self.num_grid_pts1)
            self.grid_2 = jnp.linspace(-self.grid_lim_2, self.grid_lim_2, self.num_grid_pts2)
            g1, g2 = jnp.meshgrid(self.grid_1, self.grid_2)
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
            bg_rgb = None,
            num_grid_pts=7,
            bds_bruteforce=[0.0005, 0.3],
        )

    @classmethod
    def human_varying_background(
        cls,
        base_dir: str,
        subN: int,
        adaptation_cond_str: str = '_gray',
    ):
        varyingbg_lookup = {
            '_gray': dict(
                file_date='10062025',
                bg_rgb=np.array([0.6014, 0.6200, 0.6234]),
            ),
            '_blue': dict(
                file_date='10062025',
                bg_rgb=np.array([0.4741, 0.6123, 0.9673]),
            ),
            '_orange': dict(
                file_date='02012026',
                bg_rgb=np.array([0.7622, 0.6004, 0.3853]),
            ),
        }

        if adaptation_cond_str not in varyingbg_lookup:
            raise ValueError(
                "adaptation_cond_str must be one of '_gray', '_blue', or '_orange'."
            )

        varyingbg_cfg = varyingbg_lookup[adaptation_cond_str]

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
            file_date=varyingbg_cfg['file_date'],
            adaptation_cond_str=adaptation_cond_str,
            exptCond='_4dExpt_Isoluminant plane',
            bg_rgb=varyingbg_cfg['bg_rgb'],
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
            bg_rgb = None,
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
            bg_rgb = None,
            num_grid_pts=7,
            bds_bruteforce=[0.0005, 0.3],
            coloralg='CIE1994',
            # file_name optional → auto-generated
        )

    @classmethod
    def infer_from_selection(
        cls,
        base_dir: str,
        subN: int,
        input_fileDir: str,
        file_name: str,
    ):
        """
        Infer the appropriate 4D dataset configuration from a user-selected file.

        This is mainly intended for plotting / evaluation scripts where the user
        picks a fitted-model pickle from disk and we want to recover the matching
        DatasetConfig_4D automatically instead of manually toggling between:
          - human_isoluminant
          - human_varying_background
          - human_ls_isolating
          - simulated_isoluminant

        Inference rules:
          - If the selected path contains '4D_Expt_varyingBackground', use the
            varying-background config and parse the adapting condition from the
            filename suffix (e.g. '_gray.pkl', '_blue.pkl', '_orange.pkl').
          - If the selected path contains '4D_Expt_dichromats', or the filename
            mentions 'LSisolating plane', use the LS-isolating config.
          - If the selected path contains
            'META_analysis/ModelFitting_DataFiles/4dTask/CIE', use the
            simulated isoluminant config.
          - Otherwise, if the selected path contains 'pilot2', or the filename
            mentions 'Isoluminant plane', use the human isoluminant config.

        Raises:
          ValueError if the selection does not match any known 4D dataset
          pattern, or if a varying-background file does not include a
          recognizable condition suffix.
        """
        selected_path = os.path.join(input_fileDir, file_name)

        if '4D_Expt_varyingBackground' in selected_path:
            match = re.search(r'_(gray|blue|orange)(?=\.pkl$)', file_name)
            if match is None:
                raise ValueError(
                    "Could not infer varying-background condition from file name. "
                    "Expected a suffix like '_gray.pkl', '_blue.pkl', or '_orange.pkl'."
                )
            return cls.human_varying_background(base_dir, subN, f"_{match.group(1)}")

        if '4D_Expt_dichromats' in selected_path or 'LSisolating plane' in file_name:
            return cls.human_ls_isolating(base_dir, subN)

        if 'META_analysis/ModelFitting_DataFiles/4dTask/CIE' in selected_path:
            return cls.simulated_isoluminant(base_dir, subN)

        if 'pilot2' in selected_path or 'Isoluminant plane' in file_name:
            return cls.human_isoluminant(base_dir, subN)

        raise ValueError(
            "Could not infer DatasetConfig_4D from the selected path/file name: "
            f"{selected_path}"
        )
    
    def print_summary(self):
        print("---- Dataset Config ----")
        print(f"totalSessions   : {self.totalSessions}")
        print(f"nSession        : {self.nSession}")
        print(f"path_str        : {self.path_str}")
        print(f"plane_2D        : {self.plane_2D}")
        print(f"cal_file_date   : {self.file_date}")
    
        if self.num_grid_pts1 is not None and self.num_grid_pts2 is not None:
            print(f"num_grid_pts1   : {self.num_grid_pts1}")
            print(f"num_grid_pts2   : {self.num_grid_pts2}")
        elif self.num_grid_pts is not None:
            print(f"num_grid_pts    : {self.num_grid_pts}")
        else:
            print("num_grid_pts    : None")
    
        print(f"bds_bruteforce  : {self.bds_bruteforce}")
        print("------------------------")


@dataclass
class DatasetConfig_4D_MOCS(MOCSConfigMixin):
    """
    Configuration helper for analyses that compare MOCS thresholds against
    Wishart-model predictions in the 4D task.

    Unlike DatasetConfig_4D, which is centered on loading adaptive-session data
    and fitting the WPPM, this class only stores the metadata needed to load:
      1. a Wishart fit to compare against, and
      2. a MOCS (validation) dataset, whose field names differ slightly between human
         and simulated data files.

    The main purpose of this class is to normalize those format differences so
    MOCS analysis scripts can work with a common schema.
    """

    base_dir: str
    subN: int
    flag_load_datafile: bool
    stim_dims: int
    plane_2D: Optional[str]
    file_date: Optional[str]
    wishart_dir: Optional[str] = None
    wishart_file_name: Optional[str] = None
    mocs_data_dir: Optional[str] = None
    mocs_data_file_name: Optional[str] = None
    mocs_fit_dir: Optional[str] = None
    mocs_fit_file_name: Optional[str] = None
    coloralg: Optional[str] = None

    # Keys used to normalize human vs simulated MOCS pickles.
    mocs_key_map: dict = field(init=False)

    def __post_init__(self):
        if self.flag_load_datafile:
            self.mocs_key_map = {
                "xref_unique": "xref_unique_MOCS",
                "refStimulus": "refStimulus_MOCS",
                "compStimulus": "compStimulus_MOCS",
                "responses": "responses_MOCS",
                "nRefs": "nRefs_MOCS",
                "nLevels": "nLevels_MOCS",
                "nTrials": "nTrials_MOCS",
            }
        else:
            self.mocs_key_map = {
                "xref_unique": "xref_unique",
                "refStimulus": "refStimulus",
                "compStimulus": "compStimulus",
                "responses": "responses",
                "nRefs": "nRefs",
                "nLevels": "nLevels",
                "nTrials": "num_trials",
            }

    @classmethod
    def human_isoluminant(cls, base_dir: str, subN: int, decay_rate: float = 0.4,
                          var_scaler: float = 0.0003):
        plane_2D = "Isoluminant plane"
        wishart_dir = os.path.join(
            base_dir, "ELPS_analysis", "Experiment_DataFiles", "pilot2", f"sub{subN}", "fits"
        )
        wishart_file_name = (
            f"Fitted_ColorDiscrimination_4dExpt_{plane_2D}_sub{subN}_"
            f"decayRate{decay_rate}_varScaler{var_scaler}_nBasisDeg5.pkl"
        )
        mocs_fit_file_name = (
            f"Fitted_weibull_psychometric_func_{plane_2D}_6000totalTrials_25refs"
            f"_MOCS_sub{subN}_decayRate{decay_rate}_varScaler{var_scaler}_nBasisDeg5.pkl"
        )
        return cls(
            base_dir=base_dir,
            subN=subN,
            flag_load_datafile=True,
            stim_dims=2,
            plane_2D=plane_2D,
            file_date="02242025",
            wishart_dir=wishart_dir,
            wishart_file_name=wishart_file_name,
            mocs_data_dir=wishart_dir,
            mocs_data_file_name=wishart_file_name,
            mocs_fit_dir=wishart_dir,
            mocs_fit_file_name = mocs_fit_file_name,
        )

    @classmethod
    def simulated_isoluminant(
        cls,
        base_dir: str,
        subN: int = 1,
        coloralg: str = "CIE1994",
        decay_rate: float = 0.4,
        var_scaler: float = 0.0003,
        sobol_seed: int = 2000,
    ):
        wishart_dir = os.path.join(
            base_dir, "META_analysis", "ModelFitting_DataFiles", "4dTask", "CIE", f"sub{subN}"
        )
        mocs_data_dir = os.path.join(
            base_dir, "ELPS_analysis", "Simulation_DataFiles", "MOCS", "gt_CIE"
        )
        wishart_file_name = (
            "Fitted_Sim4dTask_colorDiscrimination_EAVC_6000Trials_"
            f"300_300_300_5100_sub{subN}_gt{coloralg}_"
            f"decayRate{decay_rate}_varScaler{var_scaler}_nBasisDeg5.pkl"
        )
        mocs_data_file_name = (
            f"Sim2dTask_colorDiscrimination_Isoluminant plane_MOCStrials_"
            f"25refs_12levels_20trialsPerLevel_sub{coloralg}_Sobol_seed{sobol_seed}.pkl"
        )
        mocs_fit_file_name = (
            f"Fitted_weibull_psychometric_func_Isoluminant plane_240totalTrials_"
            f"25refs_MOCS_sub{coloralg}_decayRate{decay_rate}_varScaler{var_scaler}_nBasisDeg5.pkl"
        )
        return cls(
            base_dir=base_dir,
            subN=subN,
            flag_load_datafile=False,
            stim_dims=2,
            plane_2D="Isoluminant plane",
            file_date="02242025",
            wishart_dir=wishart_dir,
            wishart_file_name=wishart_file_name,
            mocs_data_dir=mocs_data_dir,
            mocs_data_file_name=mocs_data_file_name,
            mocs_fit_dir=mocs_data_dir,
            mocs_fit_file_name=mocs_fit_file_name,
            coloralg=coloralg,
        )

    def print_summary(self):
        print("---- 4D MOCS Config ----")
        print(f"flag_load_datafile : {self.flag_load_datafile}")
        print(f"stim_dims          : {self.stim_dims}")
        print(f"plane_2D           : {self.plane_2D}")
        print(f"file_date          : {self.file_date}")
        print(f"wishart_dir        : {self.wishart_dir}")
        print(f"wishart_file_name  : {self.wishart_file_name}")
        print(f"mocs_data_dir      : {self.mocs_data_dir}")
        print(f"mocs_data_file_name: {self.mocs_data_file_name}")
        print(f"mocs_fit_dir       : {self.mocs_fit_dir}")
        print(f"mocs_fit_file_name : {self.mocs_fit_file_name}")
        print("------------------------")
