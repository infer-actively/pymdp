import os
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from pymdp.utils import Dimensions, get_model_dimensions_from_labels, create_A_matrix_stub, read_A_matrix, create_B_matrix_stubs, read_B_matrices

tmp_path = Path('tmp_dir')

if not os.path.isdir(tmp_path):
    os.mkdir(tmp_path)

class TestWrappers(unittest.TestCase):

    def test_get_model_dimensions_from_labels(self):
        """
        Tests model dimension extraction from labels including observations, states and actions.
        """
        model_labels = {
            "observations": {
                "species_observation": [
                    "absent",
                    "present",
                ],
                "budget_observation": [
                    "high",
                    "medium",
                    "low",
                ],
            },
            "states": {
                "species_state": [
                    "extant",
                    "extinct",
                ],
            },
            "actions": {
                "conservation_action": [
                    "manage",
                    "survey",
                    "stop",
                ],
            },
        }

        want = Dimensions(
            num_observations=[2, 3],
            num_observation_modalities=2,
            num_states=[2],
            num_state_factors=1,
            num_controls=[3],
            num_control_factors=1,
        )

        got = get_model_dimensions_from_labels(model_labels)

        self.assertEqual(want.num_observations, got.num_observations)
        self.assertEqual(want.num_observation_modalities, got.num_observation_modalities)
        self.assertEqual(want.num_states, got.num_states)
        self.assertEqual(want.num_state_factors, got.num_state_factors)
        self.assertEqual(want.num_controls, got.num_controls)
        self.assertEqual(want.num_control_factors, got.num_control_factors)

    def test_A_matrix_stub(self):
        """
        This tests the construction of a 2-modality, 2-hidden state factor pandas MultiIndex dataframe using 
        the `model_labels` dictionary, which contains the modality- and factor-specific levels, labeled with string
        identifiers.

        Note: actions are ignored when creating an A matrix stub
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
            "actions": {
                "actions": ["something", "nothing"],
            }
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
    