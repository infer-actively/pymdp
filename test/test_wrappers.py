import os
import unittest
from pathlib import Path
import shutil
import tempfile

import numpy as np
import itertools
import pandas as pd
from pandas.testing import assert_frame_equal

from pymdp.utils import create_A_matrix_stub, read_A_matrix, create_B_matrix_stubs, read_B_matrices

tmp_path = Path('tmp_dir')

if not os.path.isdir(tmp_path):
    os.mkdir(tmp_path)

class TestWrappers(unittest.TestCase):

    def test_A_matrix_stub(self):
        """
        This tests the construction of a 2-modality, 2-hidden state factor pandas MultiIndex dataframe using 
        the `model_labels` dictionary, which contains the modality- and factor-specific levels, labeled with string
        identifiers
        """

        model_labels = {
            "observations": {
                "grass_observation": [
                    "wet",
                    "dry"            
                    ],
                "weather_observation": [
                    "clear",
                    "rainy",
                    "cloudy"
                ]
            },
            "states": {
                "weather_state": ["raining", "clear"],
                "sprinkler_state": ["on", "off"],
            },
        }

        num_hidden_state_factors = len(model_labels["states"])
        
        expected_A_matrix_stub = create_A_matrix_stub(model_labels)
    
        temporary_file_path = (tmp_path / "A_matrix_stub.xlsx").resolve()
        expected_A_matrix_stub.to_excel(temporary_file_path)
        actual_A_matrix_stub = read_A_matrix(temporary_file_path, num_hidden_state_factors)

        os.remove(temporary_file_path)

        frames_are_equal = assert_frame_equal(expected_A_matrix_stub, actual_A_matrix_stub) is None
        self.assertTrue(frames_are_equal)
    
    def test_B_matrix_stub(self):
        """
        This tests the construction of a 1-modality, 2-hidden state factor, 2 control factor pandas MultiIndex dataframe using 
        the `model_labels` dictionary, which contains the hidden-state-factor- and control-factor-specific levels, labeled with string
        identifiers
        """

        model_labels = {
            "observations": {
                "reward outcome": [
                    "win",
                    "loss"            
                    ]
            },
            "states": {
                "location": ["start", "arm1", "arm2"],
                "bandit_state": ["high_rew", "low_rew"]
            },
            "actions": {
                "arm_play": ["play_arm1", "play_arm2"],
                "bandit_state_control": ["null"]
            }
        }
        
        B_stubs = create_B_matrix_stubs(model_labels)
    
        xls_path = (tmp_path / "B_matrix_stubs.xlsx").resolve()

        with pd.ExcelWriter(xls_path) as writer:
            for factor_name, B_stub_f in B_stubs.items():
                B_stub_f.to_excel(writer,'%s' % factor_name)

        read_in_B_stubs = read_B_matrices(xls_path)

        os.remove(xls_path)

        all_stub_compares = [assert_frame_equal(stub_og, stub_read_in) for stub_og, stub_read_in in zip(*[B_stubs.values(), read_in_B_stubs.values()])]
        self.assertTrue(all(stub_compare is None for stub_compare in all_stub_compares))
       
if __name__ == "__main__":
    unittest.main()
    