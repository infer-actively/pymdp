import unittest
from pymdp.legacy.utils import Dimensions, get_model_dimensions_from_labels

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
       
if __name__ == "__main__":
    unittest.main()
    