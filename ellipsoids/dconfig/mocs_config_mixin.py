#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import dill as pickled


class MOCSConfigMixin:
    @property
    def wishart_full_path(self):
        if self.wishart_dir is None or self.wishart_file_name is None:
            return None
        return os.path.join(self.wishart_dir, self.wishart_file_name)

    @property
    def mocs_data_full_path(self):
        if self.mocs_data_dir is None or self.mocs_data_file_name is None:
            return None
        return os.path.join(self.mocs_data_dir, self.mocs_data_file_name)

    @property
    def mocs_fit_full_path(self):
        if self.mocs_fit_dir is None or self.mocs_fit_file_name is None:
            return None
        return os.path.join(self.mocs_fit_dir, self.mocs_fit_file_name)

    def extract_mocs_fields(self, vars_dict):
        """
        Return a normalized dictionary of MOCS arrays / counts regardless of
        whether the source pickle came from a human or simulated dataset.
        """
        return {
            key: vars_dict[src_key]
            for key, src_key in self.mocs_key_map.items()
        }

    @property
    def mocs_source_full_path(self):
        """
        Path to the pickle that stores the MOCS fields for this configuration.

        For human data, the MOCS summaries are stored inside the fitted Wishart
        pickle. For simulated data, they live in a separate MOCS pickle.
        """
        return self.wishart_full_path if self.flag_load_datafile else self.mocs_data_full_path

    def load_mocs_data(self):
        """
        Load and normalize MOCS fields from the appropriate pickle source.
        """
        source_path = self.mocs_source_full_path
        if source_path is None:
            raise ValueError("This config does not define a valid MOCS source path.")

        with open(source_path, "rb") as f:
            vars_dict = pickled.load(f)

        return self.extract_mocs_fields(vars_dict)
