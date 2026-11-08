import numpy as np
from pymdp.utils import norm_dist
from typing import Any, Iterator, Optional


class Distribution:

    def __init__(self, event: dict, batch: dict = {}, data: np.ndarray | None = None) -> None:
        self.event = event
        self.batch = batch

        self.event_indices = {
            key: {v: i for i, v in enumerate(values)}
            for key, values in event.items()
        }
        self.batch_indices = {
            key: {v: i for i, v in enumerate(values)}
            for key, values in batch.items()
        }

        if data is not None:
            self._data = data
        else:
            shape = []
            for v in event.values():
                shape.append(len(v))
            for v in batch.values():
                shape.append(len(v))
            self._data = np.zeros(shape)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        if hasattr(self, '_data') and self._data is not None:
            if self._data.shape != value.shape:
                raise ValueError(f"When setting the tensor directly, the new shape {value.shape} must match the existing tensor shape {self._data.shape}")
        self._data = value

    def get(
        self, batch: dict | None = None, event: dict | None = None, *, pointwise: bool = False
    ) -> np.ndarray:
        """Read the block the labels name, in declared axis order.

        Lists on several axes select the block they describe. pointwise=True
        zips them into single points instead, as described on set.
        """
        event_slices = self._get_slices(event, self.event_indices, self.event)
        batch_slices = self._get_slices(batch, self.batch_indices, self.batch)

        slices = event_slices + batch_slices
        return self._data[self._index(slices, pointwise)]

    def set(
        self,
        batch: dict | None = None,
        event: dict | None = None,
        values: np.ndarray | float | int | None = None,
        *,
        pointwise: bool = False,
    ) -> None:
        """Write values into the block the labels name, in declared axis order.

        values is broadcast into that block by the usual numpy rules, so a
        value whose shape is smaller than the block is repeated across it.
        Writing lists of labels on two axes therefore fills the whole block:

            transition.set(
                event={"loc": locs}, batch={"loc": locs, "ctrl": "up"},
                values=np.ones(4),
            )

        writes 1.0 to all 16 entries of the 4 by 4 block, because a (4,)
        value is broadcast into every row of it. Before block selection
        existed this same expression wrote the 4 diagonal entries. Pass
        pointwise=True to keep that meaning: the lists are then zipped into
        4 points and values is matched against those points instead.
        """
        event_slices = self._get_slices(event, self.event_indices, self.event)
        batch_slices = self._get_slices(batch, self.batch_indices, self.batch)

        slices = event_slices + batch_slices
        self._data[self._index(slices, pointwise)] = values

    def _index(self, slices: list[Any], pointwise: bool) -> tuple:
        """Turn per-axis indices into a tuple that selects what the labels name.

        Without lists this returns the entries untouched, so plain label and
        slice access stays basic indexing and keeps returning views.

        With a list on any axis, every axis is converted to an index array:
        slices become arange and each list is reshaped onto its own output
        dimension. No slices are left in the tuple, so numpy has nothing to
        separate and the result keeps the declared axis order. Two lists select
        the block they describe rather than its diagonal. Scalars stay scalar
        and their axes collapse, as with basic indexing.

        With pointwise=True the lists zip instead: they must share one length,
        they are consumed together along one output axis placed where the first
        list appears, and the remaining axes keep declared order. This is the
        old broadcast behaviour made explicit, for writing diagonals and
        per-point updates that block selection cannot express.
        """
        has_list = any(isinstance(s, list) for s in slices)
        if not has_list:
            if pointwise:
                raise ValueError("pointwise indexing needs a list of labels on at least one axis")
            return tuple(slices)

        if pointwise:
            lengths = {len(s) for s in slices if isinstance(s, list)}
            if len(lengths) > 1:
                raise ValueError(
                    f"pointwise lists must all have the same length, got lengths {sorted(lengths)}"
                )
            n_out = 1 + sum(1 for s in slices if isinstance(s, slice))
        else:
            n_out = sum(1 for s in slices if isinstance(s, (list, slice)))

        out: list[Any] = []
        dim = 0
        point_dim = None
        for axis, s in enumerate(slices):
            if isinstance(s, list):
                if pointwise:
                    if point_dim is None:
                        point_dim = dim
                        dim += 1
                    use = point_dim
                else:
                    use = dim
                    dim += 1
                shape = [1] * n_out
                shape[use] = -1
                out.append(np.asarray(s, dtype=int).reshape(shape))
            elif isinstance(s, slice):
                shape = [1] * n_out
                shape[dim] = -1
                out.append(np.arange(self._data.shape[axis]).reshape(shape))
                dim += 1
            else:
                out.append(s)
        return tuple(out)

    @property
    def points(self) -> "_PointwiseView":
        """Positional pointwise indexing, the zip counterpart of block access.

        d.points[["A", "B"], ["C", "D"], "up"] reads or writes the two entries
        (A, C, up) and (B, D, up) rather than the 2 by 2 block. Lists zip along
        one shared axis and must have equal lengths.
        """
        return _PointwiseView(self)

    def _get_slices(self, keys: dict | None, indices: dict, full_indices: dict) -> list[Any]:
        slices = []
        if keys is None:
            return [slice(None)] * len(full_indices)
        for key in full_indices:
            if key in keys:
                if isinstance(keys[key], list):
                    slices.append(
                        [self._get_index(v, indices[key]) for v in keys[key]]
                    )
                else:
                    slices.append(self._get_index(keys[key], indices[key]))
            else:
                slices.append(slice(None))
        return slices

    def _get_index(self, key: Any, index_map: dict) -> int:
        if isinstance(key, int):
            return key
        else:
            return index_map[key]

    def _get_index_from_axis(self, axis: int, element: Any) -> int | slice | list[int]:
        if isinstance(element, slice):
            return slice(None)
        if axis < len(self.event):
            key = list(self.event.keys())[axis]
            index_map = self.event_indices[key]
        else:
            key = list(self.batch.keys())[axis - len(self.event)]
            index_map = self.batch_indices[key]
        if isinstance(element, list):
            return [self._get_index(v, index_map) for v in element]
        return self._get_index(element, index_map)

    def _positional_slices(self, indices: Any) -> list[Any]:
        """Resolve a bracket index tuple to one entry per axis.

        Keys are positional over the event axes and then the batch axes. An
        event axis and a batch axis may share a key name while carrying
        different label orders, so axes are resolved by position and never by
        name. Axes left off the end of the tuple are taken whole.

        Both bracket paths use this, so d[...] and d.points[...] accept the
        same keys and differ only in how the lists are consumed.
        """
        if not isinstance(indices, tuple):
            indices = (indices,)
        n_axes = len(self.event) + len(self.batch)
        if len(indices) > n_axes:
            raise IndexError(
                f"too many indices for Distribution: it has {n_axes} axes, "
                f"but {len(indices)} were indexed"
            )
        out = [
            self._get_index_from_axis(axis, element)
            for axis, element in enumerate(indices)
        ]
        out.extend([slice(None)] * (n_axes - len(indices)))
        return out

    def __getitem__(self, indices: Any) -> np.ndarray:
        """Read by label. Lists select the block they name, as get does."""
        slices = self._positional_slices(indices)
        return self._data[self._index(slices, pointwise=False)]

    def __setitem__(self, indices: Any, value: Any) -> None:
        """Write by label. Lists select the block they name, as set does."""
        slices = self._positional_slices(indices)
        self._data[self._index(slices, pointwise=False)] = value

    def normalize(self) -> None:
        self.data = norm_dist(self.data)

    def __repr__(self) -> str:
        return f"Distribution({self.event}, {self.batch})\n {self.data}"


class _PointwiseView:
    """Bracket access on Distribution.points.

    Keys are positional over the event axes then the batch axes, the same order
    __getitem__ uses. Labels, integers, lists of either, and bare slices are
    accepted; lists zip rather than selecting a block. At least one axis must
    receive a list: a selection of labels and slices alone names no points to
    zip and raises ValueError, so d.points["A", "C", "up"] is not a way to read
    one entry. Use d["A", "C", "up"] for that.
    """

    def __init__(self, dist: Distribution) -> None:
        self._dist = dist

    def _translate(self, indices: Any) -> list[Any]:
        # Positional resolution is shared with Distribution's own brackets, so
        # both accept the same keys and reject the same arities.
        return self._dist._positional_slices(indices)

    def __getitem__(self, indices: Any) -> np.ndarray:
        slices = self._translate(indices)
        return self._dist._data[self._dist._index(slices, pointwise=True)]

    def __setitem__(self, indices: Any, values: Any) -> None:
        slices = self._translate(indices)
        self._dist._data[self._dist._index(slices, pointwise=True)] = values


class DistributionIndexer(dict):
    """
    Helper class to allow for indexing of distributions by their event keys.
    Acts as a list otherwise ...
    """

    def __init__(self, distributions: list[Distribution]) -> None:
        super().__init__()
        self.distributions = distributions
        for d in distributions:
            for key in d.event:
                self[key] = d

    def __getitem__(self, key: str | int) -> Distribution:
        if isinstance(key, int):
            return self.distributions[key]
        else:
            if key not in self.keys():
                raise KeyError(
                    f"Key {key} not found in " + str([k for k in self.keys()])
                )
            return super().__getitem__(key)

    def __iter__(self) -> Iterator[Distribution]:
        return iter(self.distributions)


class Model(dict):

    def __init__(
        self,
        likelihoods: list[Distribution],
        transitions: list[Distribution],
        preferred_observations: list[Distribution],
        priors: list[Distribution],
        preferred_states: list[Distribution],
        B_action_dependencies: Optional[list[list[int]]]=None,
        num_controls: Optional[list[int]]=None,
    ) -> None:
        super().__init__()
        super().__setitem__("A", likelihoods)
        super().__setitem__("B", transitions)
        super().__setitem__("C", preferred_observations)
        super().__setitem__("D", priors)
        super().__setitem__("H", preferred_states)
        super().__setitem__("B_action_dependencies", B_action_dependencies)
        super().__setitem__("num_controls", num_controls)

    def __getattr__(self, key: str) -> Any:
        if key in ["A", "B", "C", "D", "H"]:
            return DistributionIndexer(self[key])
        elif key == "B_action_dependencies":
            return self["B_action_dependencies"]
        elif key == "num_controls":
            return self["num_controls"]
        raise AttributeError("Model only supports attributes A,B,C,D,H, B_action_dependencies, and num_controls")


def compile_model(config: dict[str, Any]) -> Model:
    """Compile a model from a config.

    Takes a model description dictionary and builds the corresponding
    Likelihood and Transition tensors. The tensors are filled with only
    zeros and need to be filled in later by the caller of this function.
    ---
    The config  should consist of three top-level keys:
        * observations
        * controls
        * states
    where each entry consists of another dictionary with the name of the
    modality as key and the modality description.

    The modality description should consist out of either a `size` or `elements`
    field indicating the named elements or the size of the integer array.
    In the case of an observation the `depends_on` field needs to be present to
    indicate what state factor links to this observation. In the case of states
    the `depends_on` and `controlled_by` fields are needed.
    ---
    example config:
    { "observations": {
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
    }}
    """
    # these are needed to get the ordering of the dimensions correct for pymdp
    state_dependencies = dict()
    control_dependencies = dict()
    likelihood_dependencies = dict()
    transition_events = dict()
    likelihood_events = dict()
    labels = dict()
    shape = dict()
    for mod in config:
        for k, v in config[mod].items():
            for keyword in v:
                match keyword:
                    case "elements":
                        shape[k] = len(v[keyword])
                        labels[k] = [name for name in v[keyword]]
                    case "size":
                        shape[k] = v[keyword]
                        labels[k] = list(range(v[keyword]))
                    case "depends_on":
                        if mod == "states":
                            state_dependencies[k] = [
                                name for name in v[keyword]
                            ]
                            if k in v[keyword]:
                                transition_events[k] = labels[k]
                        else:
                            likelihood_dependencies[k] = [
                                name for name in v[keyword]
                            ]
                            likelihood_events[k] = labels[k]
                    case "controlled_by":
                        control_dependencies[k] = [name for name in v[keyword]]

    transitions = []
    for event, description in transition_events.items():
        arr_shape = [len(description)]
        batch_descr = dict()
        event_descr = {event: description}
        for dep in state_dependencies[event]:
            arr_shape.append(shape[dep])
            batch_descr[dep] = labels[dep]
        for dep in control_dependencies[event]:
            arr_shape.append(shape[dep])
            batch_descr[dep] = labels[dep]
        arr = np.zeros(arr_shape)
        transitions.append(Distribution(event_descr, batch_descr, arr))

    priors = []
    for event, description in transition_events.items():
        arr_shape = [len(description)]
        arr = np.ones(arr_shape) / len(description)
        event_descr = {event: description}
        priors.append(Distribution(event_descr, data=arr))

    likelihoods = []
    for event, description in likelihood_events.items():
        arr_shape = [len(description)]
        batch_descr = dict()
        event_descr = {event: description}
        for dep in likelihood_dependencies[event]:
            arr_shape.append(shape[dep])
            batch_descr[dep] = labels[dep]
        arr = np.zeros(arr_shape)
        likelihoods.append(Distribution(event_descr, batch_descr, arr))

    preferred_observations = []
    for event, description in likelihood_events.items():
        arr_shape = [len(description)]
        arr = np.zeros(arr_shape)
        event_descr = {event: description}
        preferred_observations.append(Distribution(event_descr, data=arr))

    preferred_states = []
    for event, description in transition_events.items():
        arr_shape = [len(description)]
        arr = np.ones(arr_shape) / len(description)
        event_descr = {event: description}
        preferred_states.append(Distribution(event_descr, data=arr))

    B_action_dependencies = [
        [list(config["controls"].keys()).index(i) for i in s["controlled_by"]] 
        for s in config["states"].values()
    ]

    num_controls = []
    for c in config["controls"].values():
        if "elements" in c:
            num_controls.append(len(c["elements"]))
        elif "size" in c:
            num_controls.append(int(c["size"]))
        else:
            raise ValueError(f"Control {c} has no elements or size")

    return Model(
        likelihoods, transitions, preferred_observations, priors, preferred_states, B_action_dependencies, num_controls
    )


def get_dependencies(
    likelihoods: list[Distribution], transitions: list[Distribution]
) -> tuple[list[list[int]], list[list[int]]]:
    likelihood_dependencies = dict()
    transition_dependencies = dict()
    states = [list(trans.event.keys())[0] for trans in transitions]
    for like in likelihoods:
        likelihood_dependencies[list(like.event.keys())[0]] = [
            states.index(name) for name in like.batch.keys()
        ]
    for trans in transitions:
        transition_dependencies[list(trans.event.keys())[0]] = [
            states.index(name) for name in trans.batch.keys() if name in states
        ]
    return list(likelihood_dependencies.values()), list(
        transition_dependencies.values()
    )


if __name__ == "__main__":
    controls = ["up", "down"]
    locations = ["A", "B", "C", "D"]

    data = np.zeros((len(locations), len(locations), len(controls)))
    transition = Distribution(
        {"location": locations},
        {"location": locations, "control": controls},
        data,
    )

    assert transition["A", "B", "up"] == 0.0
    assert transition[:, "B", "up"].shape == (4,)
    assert transition["A", "B", :].shape == (2,)
    assert transition[:, "B", :].shape == (4, 2)
    assert transition[:, :, :].shape == (4, 4, 2)
    assert transition[0, "B", 0] == 0.0
    assert transition[:, "B", 0].shape == (4,)

    transition["A", "B", "up"] = 0.5
    assert transition["A", "B", "up"] == 0.5
    transition[:, "B", "up"] = np.ones(4)
    assert np.all(transition[:, "B", "up"] == 1.0)

    assert transition.get({"location": "A"}, {"location": "B"}).shape == (2,)
    assert (
        transition.get({"location": "A", "control": "up"}, {"location": "B"})
        == 0.0
    )
    assert transition.get({"control": "up"}).shape == (4, 4)

    transition.set({"location": "A", "control": "up"}, {"location": "B"}, 0.5)
    assert (
        transition.get({"location": "A", "control": "up"}, {"location": "B"})
        == 0.5
    )
    transition.set({"location": 0, "control": "up"}, {"location": "B"}, 0.7)
    assert (
        transition.get({"location": "A", "control": "up"}, {"location": "B"})
        == 0.7
    )
    transition.set({"location": "A"}, {"location": "B"}, np.ones(2))
    assert np.all(transition.get({"location": "A"}, {"location": "B"}) == 1.0)

    model_example = {
        "observations": {
            "observation_1": {"size": 10, "depends_on": ["factor_1"]},
            "observation_2": {
                "elements": ["A", "B"],
                "depends_on": ["factor_2"],
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
    model = compile_model(model_example)
    like = model.A
    trans = model.B
    assert len(trans) == 2
    assert len(like) == 2
    assert trans[0].data.shape == (3, 3, 2, 2, 2)
    assert trans[1].data.shape == (2, 2, 2)
    assert like[0].data.shape == (10, 3)
    assert like[1].data.shape == (2, 2)
    assert like["observation_1"][:, "II"] is not None
    assert like["observation_2"][1, :] is not None
    A_deps, B_deps = get_dependencies(like, trans)
    print(A_deps, B_deps)

    model_description = {
        "observations": {
            "o1": {"elements": ["A", "B", "C", "D"], "depends_on": ["s1"]},
        },
        "controls": {"c1": {"elements": ["up", "down"]}},
        "states": {
            "s1": {
                "elements": ["A", "B", "C", "D"],
                "depends_on": ["s1"],
                "controlled_by": ["c1"],
            },
        },
    }

    model = compile_model(model_description)
