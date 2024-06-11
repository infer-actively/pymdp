import unittest
from pymdp.jax import distribution
import numpy as np


class TestDists(unittest.TestCase):

    def test_distribution_slice(self):
        controls = ["up", "down"]
        locations = ["A", "B", "C", "D"]

        data = np.zeros((len(locations), len(locations), len(controls)))
        transition = distribution.Distribution(
            data,
            {"location": locations},
            {"location": locations, "control": controls},
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
            data,
            {"location": locations},
            {"location": locations, "control": controls},
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
                    "depends_on_states": ["factor_1", "factor_2"],
                    "depends_on_control": ["control_1", "control_2"],
                },
                "factor_2": {
                    "elements": ["foo", "bar"],
                    "depends_on_states": ["factor_2"],
                    "depends_on_control": ["control_2"],
                },
            },
        }
        like, trans = distribution.compile_model(model_example)
        self.assertEqual(len(trans), 2)
        self.assertEqual(len(like), 2)
        self.assertEqual(trans[0].data.shape, (3, 3, 2, 2, 2))
        self.assertEqual(trans[1].data.shape, (2, 2, 2))
        self.assertEqual(like[0].data.shape, (10, 3))
        self.assertEqual(like[1].data.shape, (2, 3))
        self.assertIsNotNone
        self.assertIsNotNone(like[0][:, "II"])
        self.assertIsNotNone(like[1][1, :])
