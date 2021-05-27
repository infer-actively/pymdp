import os
import unittest
from pathlib import Path
import shutil
import tempfile

import numpy as np
import itertools
import pandas as pd
from pandas.testing import assert_frame_equal

from pymdp.core.utils import create_A_matrix_stub, read_A_matrix

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
        
        expected_A_matrix_stub = create_A_matrix_stub(model_labels)
    
        temporary_file_path = (tmp_path / "A_matrix_stub.xlsx").resolve()
        expected_A_matrix_stub.to_excel(temporary_file_path)
        actual_A_matrix_stub = read_A_matrix(temporary_file_path)

        os.remove(temporary_file_path)

        frames_are_equal = assert_frame_equal(expected_A_matrix_stub, actual_A_matrix_stub) is None
        self.assertTrue(frames_are_equal)
       
if __name__ == "__main__":
    unittest.main()
    