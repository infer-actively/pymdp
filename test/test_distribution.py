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

    def test_get_set_block_with_lists_on_two_axes(self):
        """Lists on two axes select the block they name, in declared axis order."""
        likelihood = distribution.Distribution(
            {"obs": ["r", "g", "b"]}, {"state": ["s0", "s1", "s2"]}, np.arange(9.0).reshape(3, 3)
        )

        block = likelihood.get(event={"obs": ["r", "g"]}, batch={"state": ["s0", "s1"]})
        self.assertEqual(block.shape, (2, 2))
        self.assertTrue(np.array_equal(block, np.array([[0.0, 1.0], [3.0, 4.0]])))

        likelihood.set(
            event={"obs": ["r", "g"]}, batch={"state": ["s0", "s1"]},
            values=np.array([[10.0, 11.0], [12.0, 13.0]]),
        )
        expected = np.arange(9.0).reshape(3, 3)
        expected[:2, :2] = [[10.0, 11.0], [12.0, 13.0]]
        self.assertTrue(np.array_equal(likelihood.data, expected))

    def test_get_keeps_declared_axis_order_across_untouched_axes(self):
        """A list after an untouched axis must not migrate to the front."""
        transition = distribution.Distribution(
            {"obs": ["o0", "o1", "o2"]},
            {"state": ["s0", "s1", "s2", "s3"], "ctrl": ["up", "down"]},
            np.arange(24.0).reshape(3, 4, 2),
        )

        got = transition.get(event={"obs": "o0"}, batch={"ctrl": ["up", "down"]})
        self.assertEqual(got.shape, (4, 2))
        self.assertTrue(np.array_equal(got, np.arange(24.0).reshape(3, 4, 2)[0, :, :]))

        got = transition.get(event={"obs": ["o0", "o1"]}, batch={"ctrl": ["up", "down"]})
        self.assertEqual(got.shape, (2, 4, 2))

    def test_get_without_lists_returns_a_view(self):
        likelihood = distribution.Distribution(
            {"obs": ["r", "g", "b"]}, {"state": ["s0", "s1", "s2"]}, np.arange(9.0).reshape(3, 3)
        )
        got = likelihood.get(event={"obs": "r"})
        self.assertTrue(np.shares_memory(got, likelihood.data))

    def test_pointwise_zips_lists_along_one_axis(self):
        likelihood = distribution.Distribution(
            {"obs": ["r", "g", "b"]}, {"state": ["s0", "s1", "s2"]}, np.arange(9.0).reshape(3, 3)
        )
        got = likelihood.get(
            event={"obs": ["r", "g"]}, batch={"state": ["s0", "s1"]}, pointwise=True
        )
        self.assertTrue(np.array_equal(got, np.array([0.0, 4.0])))

        likelihood.set(
            event={"obs": ["r", "g"]}, batch={"state": ["s0", "s1"]},
            values=[0.9, 0.8], pointwise=True,
        )
        self.assertEqual(likelihood.data[0, 0], 0.9)
        self.assertEqual(likelihood.data[1, 1], 0.8)
        self.assertEqual(likelihood.data[0, 1], 1.0)

    def test_pointwise_writes_an_off_diagonal(self):
        """An off-diagonal transition: up moves one location back and stops at the top."""
        locs = ["l0", "l1", "l2", "l3"]
        transition = distribution.Distribution(
            {"loc": locs}, {"loc": locs, "ctrl": ["up", "down"]}, np.zeros((4, 4, 2))
        )
        transition.set(
            event={"loc": ["l0", "l0", "l1", "l2"]},
            batch={"loc": ["l0", "l1", "l2", "l3"], "ctrl": "up"},
            values=1.0, pointwise=True,
        )
        up = transition.data[:, :, 0]
        self.assertTrue(np.array_equal(up.sum(axis=0), np.ones(4)))
        self.assertEqual(up[0, 0], 1.0)
        self.assertEqual(up[0, 1], 1.0)
        self.assertEqual(up[1, 2], 1.0)
        self.assertEqual(up[2, 3], 1.0)
        self.assertTrue(np.array_equal(transition.data[:, :, 1], np.zeros((4, 4))))

    def test_points_accessor_round_trip(self):
        likelihood = distribution.Distribution(
            {"obs": ["A", "B", "E"]}, {"state": ["C", "D", "F"], "ctrl": ["up", "down"]},
            np.zeros((3, 3, 2)),
        )
        likelihood.points[["A", "B"], ["C", "D"], "up"] = [0.9, 0.8]
        got = likelihood.points[["A", "B"], ["C", "D"], "up"]
        self.assertTrue(np.array_equal(got, np.array([0.9, 0.8])))
        self.assertAlmostEqual(float(likelihood.data.sum()), 1.7)

    def test_points_zip_axis_position_with_a_slice(self):
        """The zip axis sits where the first list appears; a slice keeps its own axis."""
        likelihood = distribution.Distribution(
            {"obs": ["A", "B", "E"]}, {"state": ["C", "D", "F"], "ctrl": ["up", "down"]},
            np.arange(18.0).reshape(3, 3, 2),
        )
        got = likelihood.points[["A", "B"], :, "up"]
        self.assertEqual(got.shape, (2, 3))
        base = np.arange(18.0).reshape(3, 3, 2)
        self.assertTrue(np.array_equal(got, np.stack([base[0, :, 0], base[1, :, 0]])))

    def test_points_resolves_shared_key_names_by_axis(self):
        """An event and a batch axis sharing a key name must resolve independently."""
        shared = distribution.Distribution(
            {"loc": ["a", "b"]}, {"loc": ["b", "a"]}, np.zeros((2, 2))
        )
        shared.data[0, 1] = 1.0
        self.assertTrue(
            np.array_equal(shared.points[["a"], ["a"]], np.array([1.0]))
        )
        got = shared.get(event={"loc": ["a"]}, batch={"loc": ["a"]}, pointwise=True)
        self.assertTrue(np.array_equal(got, np.array([1.0])))

    def test_pointwise_ragged_lists_raise(self):
        likelihood = distribution.Distribution(
            {"obs": ["r", "g", "b"]}, {"state": ["s0", "s1", "s2"]}, np.zeros((3, 3))
        )
        with self.assertRaisesRegex(ValueError, r"same length, got lengths \[2, 3\]"):
            likelihood.get(
                event={"obs": ["r", "g"]}, batch={"state": ["s0", "s1", "s2"]}, pointwise=True
            )
        with self.assertRaisesRegex(ValueError, "needs a list of labels on at least one axis"):
            likelihood.get(event={"obs": "r"}, pointwise=True)
        with self.assertRaisesRegex(ValueError, "needs a list of labels on at least one axis"):
            likelihood.points["r", "s0"]

    def test_brackets_select_the_block_lists_name(self):
        """Lists in brackets select a block, the same one get and set select."""
        likelihood = distribution.Distribution(
            {"obs": ["r", "g", "b"]}, {"state": ["s0", "s1", "s2"]}, np.arange(9.0).reshape(3, 3)
        )

        block = likelihood[["r", "g"], ["s0", "s1"]]
        self.assertEqual(block.shape, (2, 2))
        self.assertTrue(np.array_equal(block, np.array([[0.0, 1.0], [3.0, 4.0]])))
        self.assertTrue(
            np.array_equal(
                block, likelihood.get(event={"obs": ["r", "g"]}, batch={"state": ["s0", "s1"]})
            )
        )

        likelihood[["r", "g"], ["s0", "s1"]] = np.array([[10.0, 11.0], [12.0, 13.0]])
        expected = np.arange(9.0).reshape(3, 3)
        expected[:2, :2] = [[10.0, 11.0], [12.0, 13.0]]
        self.assertTrue(np.array_equal(likelihood.data, expected))

    def test_brackets_keep_declared_axis_order_across_untouched_axes(self):
        """A list after an untouched axis must not migrate to the front."""
        transition = distribution.Distribution(
            {"obs": ["o0", "o1", "o2"]},
            {"state": ["s0", "s1", "s2", "s3"], "ctrl": ["up", "down"]},
            np.arange(24.0).reshape(3, 4, 2),
        )
        got = transition[["o0", "o1"], :, ["up", "down"]]
        self.assertEqual(got.shape, (2, 4, 2))
        self.assertTrue(
            np.array_equal(
                got, transition.get(event={"obs": ["o0", "o1"]}, batch={"ctrl": ["up", "down"]})
            )
        )

    def test_brackets_without_lists_return_a_view(self):
        transition = distribution.Distribution(
            {"obs": ["o0", "o1", "o2"]},
            {"state": ["s0", "s1", "s2", "s3"], "ctrl": ["up", "down"]},
            np.arange(24.0).reshape(3, 4, 2),
        )
        self.assertTrue(np.shares_memory(transition["o0"], transition.data))
        self.assertTrue(np.shares_memory(transition[:, "s1", :], transition.data))

    def test_set_keeps_declared_axis_order_across_untouched_axes(self):
        """set writes into the block in declared order, label order included."""
        transition = distribution.Distribution(
            {"obs": ["o0", "o1", "o2"]},
            {"state": ["s0", "s1", "s2", "s3"], "ctrl": ["up", "down"]},
            np.zeros((3, 4, 2)),
        )
        values = np.arange(100.0, 116.0).reshape(2, 4, 2)
        transition.set(
            event={"obs": ["o0", "o1"]}, batch={"ctrl": ["down", "up"]}, values=values
        )

        expected = np.zeros((3, 4, 2))
        expected[:2, :, 1] = values[:, :, 0]
        expected[:2, :, 0] = values[:, :, 1]
        self.assertTrue(np.array_equal(transition.data, expected))

    def test_block_set_broadcasts_values_across_the_block(self):
        """Lists on two axes write the whole block; pointwise=True writes the diagonal."""
        locs = ["l0", "l1", "l2", "l3"]

        block = distribution.Distribution(
            {"loc": locs}, {"loc": locs, "ctrl": ["up", "down"]}, np.zeros((4, 4, 2))
        )
        block.set(event={"loc": locs}, batch={"loc": locs, "ctrl": "up"}, values=np.ones(4))
        self.assertTrue(np.array_equal(block.data[:, :, 0], np.ones((4, 4))))
        self.assertTrue(np.array_equal(block.data[:, :, 1], np.zeros((4, 4))))

        diagonal = distribution.Distribution(
            {"loc": locs}, {"loc": locs, "ctrl": ["up", "down"]}, np.zeros((4, 4, 2))
        )
        diagonal.set(
            event={"loc": locs}, batch={"loc": locs, "ctrl": "up"},
            values=np.ones(4), pointwise=True,
        )
        self.assertTrue(np.array_equal(diagonal.data[:, :, 0], np.eye(4)))

    def test_pointwise_set_values_orientation_with_a_leading_slice(self):
        """The zip axis sits where the first list appears, after the slice's own axis."""
        likelihood = distribution.Distribution(
            {"obs": ["A", "B", "E"]}, {"state": ["C", "D", "F"], "ctrl": ["up", "down"]},
            np.zeros((3, 3, 2)),
        )
        values = np.arange(6.0).reshape(3, 2)
        likelihood.points[:, ["C", "D"], "up"] = values

        self.assertTrue(np.array_equal(likelihood.points[:, ["C", "D"], "up"], values))
        self.assertTrue(np.array_equal(likelihood.data[:, 0, 0], values[:, 0]))
        self.assertTrue(np.array_equal(likelihood.data[:, 1, 0], values[:, 1]))
        self.assertEqual(likelihood.data[:, 2, :].sum(), 0.0)

        with self.assertRaises(ValueError):
            likelihood.points[:, ["C", "D"], "up"] = values.T

        through_set = distribution.Distribution(
            {"obs": ["A", "B", "E"]}, {"state": ["C", "D", "F"], "ctrl": ["up", "down"]},
            np.zeros((3, 3, 2)),
        )
        through_set.set(
            batch={"state": ["C", "D"], "ctrl": "up"}, values=values, pointwise=True
        )
        self.assertTrue(np.array_equal(through_set.data, likelihood.data))

        with self.assertRaises(ValueError):
            through_set.set(
                batch={"state": ["C", "D"], "ctrl": "up"}, values=values.T, pointwise=True
            )

        two_lists = distribution.Distribution(
            {"obs": ["A", "B", "E"]}, {"state": ["C", "D", "F"], "ctrl": ["up", "down"]},
            np.zeros((3, 3, 2)),
        )
        paired = np.arange(10.0, 16.0).reshape(3, 2)
        two_lists.set(
            batch={"state": ["C", "D"], "ctrl": ["up", "down"]},
            values=paired, pointwise=True,
        )
        self.assertTrue(np.array_equal(two_lists.data[:, 0, 0], paired[:, 0]))
        self.assertTrue(np.array_equal(two_lists.data[:, 1, 1], paired[:, 1]))
        self.assertEqual(two_lists.data.sum(), paired.sum())

    def test_pointwise_with_one_list_matches_block_selection(self):
        likelihood = distribution.Distribution(
            {"obs": ["r", "g", "b"]}, {"state": ["s0", "s1", "s2"]}, np.arange(9.0).reshape(3, 3)
        )
        got = likelihood.get(event={"obs": ["r", "g"]}, pointwise=True)
        self.assertEqual(got.shape, (2, 3))
        self.assertTrue(np.array_equal(got, np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])))
        self.assertTrue(np.array_equal(got, likelihood.get(event={"obs": ["r", "g"]})))

    def test_points_accepts_int_and_mixed_labels(self):
        likelihood = distribution.Distribution(
            {"obs": ["A", "B", "E"]}, {"state": ["C", "D", "F"], "ctrl": ["up", "down"]},
            np.arange(18.0).reshape(3, 3, 2),
        )
        by_label = likelihood.points[["A", "B"], ["C", "D"], "up"]
        self.assertTrue(np.array_equal(by_label, np.array([0.0, 8.0])))
        self.assertTrue(np.array_equal(likelihood.points[[0, 1], [0, 1], 0], by_label))
        self.assertTrue(np.array_equal(likelihood.points[[0, "B"], ["C", 1], "up"], by_label))

    def test_too_many_indices_names_the_axis_count(self):
        likelihood = distribution.Distribution(
            {"obs": ["r", "g", "b"]}, {"state": ["s0", "s1", "s2"]}, np.zeros((3, 3))
        )
        with self.assertRaisesRegex(IndexError, "it has 2 axes, but 3 were indexed"):
            likelihood["r", "s0", :]
        with self.assertRaisesRegex(IndexError, "it has 2 axes, but 3 were indexed"):
            likelihood.points[["r", "g"], ["s0", "s1"], :]
        with self.assertRaisesRegex(IndexError, "it has 2 axes, but 3 were indexed"):
            likelihood["r", "s0", "s1"]

    def test_pointwise_is_keyword_only(self):
        likelihood = distribution.Distribution(
            {"obs": ["r", "g", "b"]}, {"state": ["s0", "s1", "s2"]}, np.zeros((3, 3))
        )
        with self.assertRaises(TypeError):
            likelihood.get({"state": ["s0", "s1"]}, {"obs": ["r", "g"]}, True)
        with self.assertRaises(TypeError):
            likelihood.set({"state": ["s0", "s1"]}, {"obs": ["r", "g"]}, 1.0, True)
        self.assertEqual(likelihood.data.sum(), 0.0)

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
      