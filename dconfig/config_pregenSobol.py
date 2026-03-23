#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:26:30 2026

@author: fangfang

Configuration class for pre-generating Sobol trials used in color discrimination experiments.

Overview
--------
This module defines a dataclass (`PregenSobolConfig`) that stores all parameters
required to generate Sobol-sampled stimulus pairs for different experimental paradigms.

The configuration includes:
    - Dimensionality of the stimulus and psychophysical field
    - Number of sessions and trials per session
    - Bounds of the Sobol sampling space
    - Scaling factors controlling stimulus differences
    - Optional catch trial definitions

Design goals
------------
- Centralize all experiment-specific parameters in one place
- Enable easy switching between experiment types via factory methods
- Ensure consistency and reproducibility across scripts
- Maintain backward compatibility with existing pickle formats

Factory methods
---------------
Each classmethod returns a predefined configuration for a specific experiment:
    - isoluminant_2D4D
    - rgbcube_3D_dichromat
    - LSisolating_dichromat
    - LSisolating_dichromat_expanded
    - adaptation_round1
    - adaptation_round2

Notes
-----
- `num_repeats` is computed automatically to ensure balanced use of `sobol_scaler`
- Catch trials are optional and validated during initialization
- The `to_legacy_dict()` method provides compatibility with older analysis scripts

"""

from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np

@dataclass
class PregenSobolConfig:
    # dimensions
    stim_dims: int
    psyfield_dims: int

    # number of sessions and trials
    nSessions: int
    nTrials_sobol_perSession: int

    # optional metadata
    plane_2D: Optional[str] = None
    file_date: Optional[str] = None

    # sampling settings
    lb_sobol_trials: Sequence[float] = None
    ub_sobol_trials: Sequence[float] = None
    sobol_scaler: Sequence[float] = None

    # experiment structure
    flag_addCatchTrials: bool = False

    # optional catch trials
    delta_catchTrials_unique: Optional[np.ndarray] = None
    percent_catchTrials: Optional[float] = None

    # derived: number of repetitions of sobol_scaler needed to fill all trials
    num_repeats: int = field(init=False)

    def __post_init__(self):
        # Ensure total trials can be evenly divided across scaling factors
        if self.nTrials_sobol_perSession % len(self.sobol_scaler) != 0:
            raise ValueError("nTrials must be multiple of sobol_scaler length")
    
        # Number of times each scaler set is repeated
        self.num_repeats = self.nTrials_sobol_perSession // len(self.sobol_scaler)
    
        # Validate catch trial configuration
        if self.flag_addCatchTrials and self.delta_catchTrials_unique is None:
            raise ValueError("Catch trials enabled but delta_catchTrials_unique not provided")

    # --------------------------------------------------
    # factory constructors
    # --------------------------------------------------

    @classmethod
    def isoluminant_2D4D(cls):
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D='Isoluminant plane',
            file_date='02242025',
            nTrials_sobol_perSession=1200,
            lb_sobol_trials=[-0.75, -0.75, -0.25, -0.25],
            ub_sobol_trials=[0.75, 0.75, 0.25, 0.25],
            sobol_scaler=[2/8, 3/8, 4/8],
            flag_addCatchTrials=False,
            nSessions=15,
        )

    @classmethod
    def rgbcube_3D_dichromat(cls):
        return cls(
            stim_dims=3,
            psyfield_dims=3,
            nTrials_sobol_perSession=900,
            lb_sobol_trials=[-1, -1, -1/3],
            ub_sobol_trials=[1, 1, 1/3],
            sobol_scaler=[0.15, 0.45, 0.75],
            flag_addCatchTrials=False,
            nSessions=5,
        )

    @classmethod
    def LSisolating_dichromat(cls):
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D='LSisolating plane',
            file_date='11172025',
            nTrials_sobol_perSession=2400,
            lb_sobol_trials=[-0.75, -0.75, -0.25, -0.25],
            ub_sobol_trials=[0.75, 0.75, 0.25, 0.25],
            sobol_scaler=[2/8, 3/8, 4/8],
            flag_addCatchTrials=True,
            percent_catchTrials=0.05,
            nSessions=30,
            delta_catchTrials_unique=np.array([
                [-0.25, -0.25],
                [-0.25,  0.25],
                [ 0.25, -0.25],
                [ 0.25,  0.25],
            ]),
        )

    @classmethod
    def LSisolating_dichromat_expanded(cls):
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D='LSisolating plane',
            file_date='11172025',
            nTrials_sobol_perSession=2400,
            lb_sobol_trials=[-0.55, -0.7, -0.45, -0.3],
            ub_sobol_trials=[0.55, 0.7, 0.45, 0.3],
            sobol_scaler=[4/8, 6/8, 1],
            flag_addCatchTrials=True,
            percent_catchTrials=0.05,
            nSessions=30,
            delta_catchTrials_unique=np.array([
                [-0.45, -0.3],
                [-0.45,  0.3],
                [ 0.45, -0.3],
                [ 0.45,  0.3],
            ]),
        )

    @classmethod
    def adaptation_round1(cls):
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D='Isoluminant plane',
            file_date='10062025',
            nTrials_sobol_perSession=2400,
            lb_sobol_trials=[-0.75, -0.75, -0.25, -0.25],
            ub_sobol_trials=[0.75, 0.75, 0.25, 0.25],
            sobol_scaler=[2/8, 3/8, 4/8],
            flag_addCatchTrials=True,
            percent_catchTrials=0.05,
            nSessions=30,
            delta_catchTrials_unique=np.array([
                [-0.25, -0.25],
                [-0.25,  0.25],
                [ 0.25, -0.25],
                [ 0.25,  0.25],
            ]),
        )

    @classmethod
    def adaptation_round2(cls):
        """
        Same as round1 but with updated:
            - fewer trials per session
            - more sessions
            - updated calibration file
        """
        return cls(
            stim_dims=2,
            psyfield_dims=4,
            plane_2D='Isoluminant plane',
            file_date='02012026',
            nTrials_sobol_perSession=900,
            lb_sobol_trials=[-0.75, -0.75, -0.25, -0.25],
            ub_sobol_trials=[0.75, 0.75, 0.25, 0.25],
            sobol_scaler=[2/8, 3/8, 4/8],
            flag_addCatchTrials=True,
            percent_catchTrials=0.05,
            nSessions=40,
            delta_catchTrials_unique=np.array([
                [-0.25, -0.25],
                [-0.25,  0.25],
                [ 0.25, -0.25],
                [ 0.25,  0.25],
            ]),
        )
    
    def print_summary(self):
        print("---- Sobol Sampling Config ----")
        print(f"stim_dims                : {self.stim_dims}")
        print(f"psyfield_dims            : {self.psyfield_dims}")
        print(f"nTrials_sobol_perSession : {self.nTrials_sobol_perSession}")
        print(f"lb_sobol_trials          : {self.lb_sobol_trials}")
        print(f"ub_sobol_trials          : {self.ub_sobol_trials}")
        print(f"sobol_scaler             : {self.sobol_scaler}")
        print(f"flag_addCatchTrials      : {self.flag_addCatchTrials}")
        print(f"nSessions                : {self.nSessions}")
    
        if self.flag_addCatchTrials and self.delta_catchTrials_unique is not None:
            print("delta_catchTrials_unique :")
            print(self.delta_catchTrials_unique)
        else:
            print("delta_catchTrials_unique : None")
    
        print("--------------------------------")
        
    def to_legacy_dict(self):
        return {"nTrials_sobol_perSession": self.nTrials_sobol_perSession,
                "lb_sobol_trials": self.lb_sobol_trials,
                "ub_sobol_trials": self.ub_sobol_trials,
                "sobol_scaler": self.sobol_scaler,
                }
    