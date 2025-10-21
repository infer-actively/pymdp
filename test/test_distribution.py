import unittest
from pymdp import distribution
import numpy as np
class TestDists(unittest.TestCase):

    def test_distribution_slice(self):
        controls = ["up", "down"]
        locations = ["A", "B", "C", "D"]

        data = np.zeros((len(locations), len(locations), len(controls)))
        transition = distribution.Distribution(
            {"location": locations},
            {"location": locations, "control": controls},
            data,
        )
        self.assertEqual(transition["A", "B", "up"], 0.0)
        self.assertEqual(transition[:, "B", "up"].shape, (4,))
        self.assertEqual(transition["A", "B", :].shape, (2,))
        self.assertEqual(transition[:, "B", :].shape, (4, 2))
        self.assertEqual(transition[:, :, :].shape, (4, 4, 2))
        self.assertEqual(transition[0, "B", 0], 0.0)
        self.assertEqual(transition[:, "B", 0].shape, (4,))

        transition["A", "B", "up"] = 0.5
        self.assertEqual(transition["A", "B", "up"], 0.5)
        transition[:, "B", "up"] = np.ones(4)
        self.assertTrue(np.all(transition[:, "B", "up"] == 1.0))

    def test_distribution_get_set(self):
        controls = ["up", "down"]
        locations = ["A", "B", "C", "D"]

        data = np.zeros((len(locations), len(locations), len(controls)))
        transition = distribution.Distribution(
            {"location": locations},
            {"location": locations, "control": controls},
            data,
        )

        self.assertEqual(
            transition.get({"location": "A"}, {"location": "B"}).shape, (2,)
        )
        self.assertEqual(
            transition.get(
                {"location": "A", "control": "up"}, {"location": "B"}
            ),
            0.0,
        )
        self.assertEqual(transition.get({"control": "up"}).shape, (4, 4))

        transition.set(
            {"location": "A", "control": "up"}, {"location": "B"}, 0.5
        )
        self.assertEqual(
            transition.get(
                {"location": "A", "control": "up"}, {"location": "B"}
            ),
            0.5,
        )
        transition.set(
            {"location": 0, "control": "up"}, {"location": "B"}, 0.7
        )
        self.assertEqual(
            transition.get(
                {"location": "A", "control": "up"}, {"location": "B"}
            ),
            0.7,
        )
        transition.set({"location": "A"}, {"location": "B"}, np.ones(2))
        self.assertTrue(
            np.all(transition.get({"location": "A"}, {"location": "B"}) == 1.0)
        )

    def test_agent_compile(self):
        model_example = {
            "observations": {
                "observation_1": {"size": 10, "depends_on": ["factor_1"]},
                "observation_2": {
                    "elements": ["A", "B"],
                    "depends_on": ["factor_1"],
                },
            },
            "controls": {
                "control_1": {"size": 2},
                "control_2": {"elements": ["X", "Y"]},
            },
            "states": {
                "factor_1": {
                    "elements": ["II", "JJ", "KK"],
                    "depends_on": ["factor_1", "factor_2"],
                    "controlled_by": ["control_1", "control_2"],
                },
                "factor_2": {
                    "elements": ["foo", "bar"],
                    "depends_on": ["factor_2"],
                    "controlled_by": ["control_2"],
                },
            },
        }
        model = distribution.compile_model(model_example)
        self.assertEqual(len(model.B), 2)
        self.assertEqual(len(model.A), 2)
        self.assertEqual(model.B[0].data.shape, (3, 3, 2, 2, 2))
        self.assertEqual(model.B[1].data.shape, (2, 2, 2))
        self.assertEqual(model.A[0].data.shape, (10, 3))
        self.assertEqual(model.A[1].data.shape, (2, 3))
        self.assertIsNotNone
        self.assertIsNotNone(model.A[0][:, "II"])
        self.assertIsNotNone(model.A[1][1, :])
        self.assertIsNotNone(model.B_action_dependencies)
        self.assertIsNotNone(model.num_controls)
        self.assertEqual(model.B_action_dependencies, [[0, 1], [1]])
        self.assertEqual(model.num_controls, [2, 2])

    def test_tensor_shape_change_protection(self):
        """
        Test that directly setting a tensor with a different shape
        than the original tensor raises an exception.
        """
        locations = ["here", "there", "everywhere"]
        data = np.zeros((len(locations), len(locations)))
        dist = distribution.Distribution({"location": locations}, {"location": locations}, data)

        # Attempting to set data with a mismatched shape should raise a ValueError
        with self.assertRaises(ValueError):
            dist.data = np.zeros((len(locations), len(locations) + 1))

        # Setting data with the same shape should not raise an exception
        try:
            dist.data = np.ones((len(locations), len(locations)))
        except ValueError:
            self.fail("Setting tensor with the same shape should not raise a ValueError")
