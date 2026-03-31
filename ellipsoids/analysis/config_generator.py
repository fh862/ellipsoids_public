#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:54:18 2024

@author: fangfang
"""

import configparser
import os
import io

class ConfigGenerator:
    def __init__(self, expt_dim, load_default_config = True, base_path = '',
                 load_file_name = '', version_new = False):
        """
        Initializes the configuration generator with specified dimensions and optional paths.
        
        Args:
            expt_dim (int): Expected dimension of the experiment. Must be one of [2, 3, 4, 6].
            load_default_config (bool): Flag to determine whether to load a default configuration.
            base_path (str): Base directory path where configuration files might be located.
            load_file_name (str): Specific file name to load the configuration from.
        
        Raises:
            ValueError: If the experiment dimension is not one of the allowed values.
        """

        if expt_dim not in (2, 3, 4, 5, 6):
            raise ValueError("The color discrimination task must be either 2, 3, 4, 5 or 6.")
        self.expt_dim            = expt_dim
        self.load_default_config = load_default_config
        self.base_path           = base_path
        self.load_file_name      = load_file_name
        self.version_new         = version_new
        self.config_parser       = configparser.ConfigParser()
        self._generate_configurations()
            
    def _default_config_str(self):
        """Generates the default configuration based on the experiment dimension."""


        config_common_part = """
        stimuli_per_trial = 1 
        outcome_types     = [binary] 
        strategy_names    = [init_strat_small, init_strat_medium, init_strat_large, opt_strat] 
        target            = 0.667 

        [init_strat_small]
        min_asks    = 20
        generator   = SobolGenerator 
        
        [init_strat_medium]
        min_asks    = 20
        generator   = SobolGenerator 

        [init_strat_large]
        min_asks    = 20
        generator   = SobolGenerator 

        [opt_strat]
        min_asks     = 100
        refit_every  = 20
        generator    = OptimizeAcqfGenerator 
        acqf         = EAVC
        model        = GPClassificationModel
        
        [GPClassificationModel]
        max_fit_time = 3
        
        [OptimizeAcqfGenerator]
        restarts     = 10 
        samps        = 1000 
        max_gen_time = 1
        use_gpu      = False

        [EAVC]
        target       = 0.667
        objective    = ProbitObjective
        """
        
        # Dimension-specific configuration settings
        if self.expt_dim == 2:
            config_dim_specific_part = """
            [common]
            parnames          = [delta_dim1, delta_dim2] 
            lb                = [-0.25, -0.25]    
            ub                = [0.25, 0.25] 
            """
        # Additional conditions for other dimensions
        elif self.expt_dim == 3:
            config_dim_specific_part = """
            [common]
            parnames          = [delta_dim1, delta_dim2, delta_dim3] 
            lb                = [-0.25, -0.25, -0.25]    
            ub                = [0.25, 0.25, 0.25] 
            """
        elif self.expt_dim == 4:
            config_dim_specific_part = """
            [common]
            parnames          = [ref_dim1, ref_dim2, delta_dim1, delta_dim2]
            lb                = [-0.75, -0.75, -0.25, -0.25]    
            ub                = [0.75, 0.75, 0.25, 0.25] 
            """
        elif self.expt_dim == 5:
            config_dim_specific_part = """
            [common]
            parnames          = [ref_dim1, ref_dim2, delta_dim1, delta_dim2, ancillary]
            lb                = [-0.75, -0.75, -0.25, -0.25, -0.7]    
            ub                = [0.75, 0.75, 0.25, 0.25, 0.7] 
            """            
        else:
            config_dim_specific_part = """
            [common]
            parnames          = [ref_dim1, ref_dim2, ref_dim3, delta_dim1, delta_dim2, delta_dim3]
            lb                = [-0.75, -0.75, -0.75, -0.25, -0.25, -0.25]    
            ub                = [0.75, 0.75, 0.75, 0.25, 0.25, 0.25] 
            """
        
        config_str = config_dim_specific_part + config_common_part
        self.config_parser.read_string(config_str)
    
    def _find_exact_path(self):
        """
        Searches recursively within the base directory for the exact path of the specified file.
        
        Returns:
            str: Full path to the found file.
        
        Raises:
            FileNotFoundError: If the file is not found within the directory.
        """
        for root, dirs, files in os.walk(self.base_path):
            if self.load_file_name in files:
                return os.path.join(root, self.load_file_name)
        raise FileNotFoundError(f"File {self.load_file_name} not found in directory {self.base_path}.")
    
    def _load_config(self, exact_path):
        """
        Loads a configuration from a specified file path and checks its consistency.
        
        Args:
            exact_path (str): Path to the configuration file to be loaded.
        """
        # load the configuration file into the ConfigParser instance
        self.config_parser.read(exact_path)
        self._consistency_check()
    
    def _string_to_list(self, section, field):
        """
        Converts a string representation of a list from the config into an actual list, stripping whitespace and brackets.
        
        Args:
            section (str): Section in the config file from where the list is retrieved.
            field (str): Field within the section that contains the list as a string.
        
        Returns:
            list: The list obtained from the config string.
        """
        param_str = self.config_parser.get(section,field)
        #remove surrrounding square brakets and split the string in to a list
        param_split = param_str.strip('[]').split(',')
        #remove any leading or trailing whitespace from each parameter name
        param = [name.strip() for name in param_split]
        return param
    
    def _consistency_check(self):
        """
        Ensures that the parameter lists in the configuration match the specified experimental dimensions.
        
        Raises:
            ValueError: If the lengths of parameter lists do not match the experiment dimensions.
        """
        parnames_list = self._string_to_list('common', 'parnames')
        if self.version_new:
            lb_list = [float(self.config_parser.get(p, 'lower_bound')) for p in parnames_list]
            ub_list = [float(self.config_parser.get(p, 'upper_bound')) for p in parnames_list]
        else:
            lb_list = self._string_to_list('common', 'lb')
            ub_list = self._string_to_list('common', 'ub')
        if len(parnames_list) != self.expt_dim:
            raise ValueError(f"The number of parameters ({len(parnames_list)}) "+\
                             "does not match the experiment dimension ({self.expt_dim}).")
        if len(lb_list) != self.expt_dim:
            raise ValueError(f"The number of lower bounds ({len(lb_list)}) does "+\
                             "not match the experiment dimension ({self.expt_dim}).")
        if len(ub_list) != self.expt_dim:
            raise ValueError(f"The number of upper bounds ({len(ub_list)}) does not "+\
                             "match the experiment dimension ({self.expt_dim}).")

    def _generate_configurations(self):
        """
        Generates or loads configuration based on initialization parameters.
        Loads from file if a path and file name are provided, otherwise loads default configuration.
        """
        if not self.load_default_config and self.base_path and self.load_file_name:
            if os.path.exists(self.base_path):
                file_path = self._find_exact_path()
                self._load_config(file_path)
                print("Local configuration loaded.")
            else:
                raise FileNotFoundError(f"The path '{self.base_path}' does not exist.")
        #if we want to use default configuration
        else:
            if self.load_default_config:
                self._default_config_str()
                print("Default configuration loaded.")
               
    #%%
    def modify_configurations(self, section, field, val):
        """
        Modifies a specific field in the configuration with a new value.
        
        Parameters:
            section (str): The section in the configuration file.
            field (str): The field within the section to modify.
            val (str): The new value to set for the field.
        """
        if not self.config_parser.has_section(section):
            raise ValueError(f"The section '{section}' does not exist in the configuration.")
        if not self.config_parser.has_option(section, field):
            raise ValueError(f"The field '{field}' does not exist in the section '{section}'.")

        self.config_parser.set(section, field, val)
        self._consistency_check()
        
    def get_config_as_string(self):
        """
        Returns the current configuration as a string.
        
        Returns:
            str: The configuration data formatted as a string.
        """
        with io.StringIO() as stream:
            self.config_parser.write(stream)
            stream.seek(0)
            return stream.read()
            
        
        
        
        
        
        
        
        