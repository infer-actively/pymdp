import numpy as np
from pymdp.utils import norm_dist


class Distribution:

    def __init__(self, event: dict, batch: dict = {}, data: np.ndarray = None):
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
            self.data = data
        else:
            shape = []
            for v in event.values():
                shape.append(len(v))
            for v in batch.values():
                shape.append(len(v))
            self.data = np.zeros(shape)

    def get(self, batch=None, event=None):
        event_slices = self._get_slices(event, self.event_indices, self.event)
        batch_slices = self._get_slices(batch, self.batch_indices, self.batch)

        slices = event_slices + batch_slices
        return self.data[tuple(slices)]

    def set(self, batch=None, event=None, values=None):
        event_slices = self._get_slices(event, self.event_indices, self.event)
        batch_slices = self._get_slices(batch, self.batch_indices, self.batch)

        slices = event_slices + batch_slices
        self.data[tuple(slices)] = values

    def _get_slices(self, keys, indices, full_indices):
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

    def _get_index(self, key, index_map):
        if isinstance(key, int):
            return key
        else:
            return index_map[key]

    def _get_index_from_axis(self, axis, element):
        if isinstance(element, slice):
            return slice(None)
        if axis < len(self.event):
            key = list(self.event.keys())[axis]
            index_map = self.event_indices[key]
        else:
            key = list(self.batch.keys())[axis - len(self.event)]
            index_map = self.batch_indices[key]
        return self._get_index(element, index_map)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        index_list = [
            self._get_index_from_axis(i, idx) for i, idx in enumerate(indices)
        ]
        return self.data[tuple(index_list)]

    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)
        index_list = [
            self._get_index_from_axis(i, idx) for i, idx in enumerate(indices)
        ]
        self.data[tuple(index_list)] = value

    def normalize(self):
        self.data = norm_dist(self.data)

    def __repr__(self):
        return f"Distribution({self.event}, {self.batch})\n {self.data}"


class DistributionIndexer(dict):
    """
    Helper class to allow for indexing of distributions by their event keys.
    Acts as a list otherwise ...
    """

    def __init__(self, distributions: list[Distribution]):
        super().__init__()
        self.distributions = distributions
        for d in distributions:
            for key in d.event:
                self[key] = d

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.distributions[key]
        else:
            if key not in self.keys():
                raise KeyError(
                    f"Key {key} not found in " + str([k for k in self.keys()])
                )
            return super().__getitem__(key)

    def __iter__(self):
        return iter(self.distributions)


class Model(dict):

    def __init__(
        self,
        likelihoods: list[Distribution],
        transitions: list[Distribution],
        preferred_outcomes: list[Distribution],
        priors: list[Distribution],
        preferred_states: list[Distribution],
    ):
        super().__init__()
        super().__setitem__("A", likelihoods)
        super().__setitem__("B", transitions)
        super().__setitem__("C", preferred_outcomes)
        super().__setitem__("D", priors)
        super().__setitem__("H", preferred_states)

    def __getattr__(self, key):
        if key in ["A", "B", "C", "D", "H"]:
            return DistributionIndexer(self[key])
        raise AttributeError("Model only supports attributes A,B,C and D")


def compile_model(config):
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

    preferred_outcomes = []
    for event, description in likelihood_events.items():
        arr_shape = [len(description)]
        arr = np.zeros(arr_shape)
        event_descr = {event: description}
        preferred_outcomes.append(Distribution(event_descr, data=arr))

    preferred_states = []
    for event, description in transition_events.items():
        arr_shape = [len(description)]
        arr = np.ones(arr_shape) / len(description)
        event_descr = {event: description}
        preferred_states.append(Distribution(event_descr, data=arr))

    return Model(
        likelihoods, transitions, preferred_outcomes, priors, preferred_states
    )


def get_dependencies(likelihoods, transitions):
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
