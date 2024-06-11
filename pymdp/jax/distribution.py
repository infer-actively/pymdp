import numpy as np


class Distribution:

    def __init__(self, data: np.ndarray, event: dict, batch: dict):
        self.data = data
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
    the `depends_on_states` and `depends_on_control` fields are needed.
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
            "depends_on_states": ["factor_1", "factor_2"],
            "depends_on_control": ["control_1", "control_2"],
        },
        "factor_2": {
            "elements": ["foo", "bar"],
            "depends_on_states": ["factor_2"],
            "depends_on_control": ["control_2"],
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
                    case "depends_on_states":
                        state_dependencies[k] = [name for name in v[keyword]]
                        if k in v[keyword]:
                            transition_events[k] = labels[k]
                    case "depends_on_control":
                        control_dependencies[k] = [name for name in v[keyword]]
                    case "depends_on":
                        likelihood_dependencies[k] = [
                            name for name in v[keyword]
                        ]
                        likelihood_events[k] = labels[k]
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
        transitions.append(Distribution(arr, event_descr, batch_descr))
    likelihoods = []
    for event, description in likelihood_events.items():
        arr_shape = [len(description)]
        batch_descr = dict()
        event_descr = {event: description}
        for dep in likelihood_dependencies[event]:
            arr_shape.append(shape[dep])
            batch_descr[dep] = labels[dep]
        arr = np.zeros(arr_shape)
        likelihoods.append(Distribution(arr, event_descr, batch_descr))
    return likelihoods, transitions


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
        data,
        {"location": locations},
        {"location": locations, "control": controls},
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
    like, trans = compile_model(model_example)
    assert len(trans) == 2
    assert len(like) == 2
    assert trans[0].data.shape == (3, 3, 2, 2, 2)
    assert trans[1].data.shape == (2, 2, 2)
    assert like[0].data.shape == (10, 3)
    assert like[1].data.shape == (2, 2)
    assert like[0][:, "II"] is not None
    assert like[1][1, :] is not None
    A_deps, B_deps = get_dependencies(trans, like)
    print(A_deps, B_deps)

    model = {
        "observations": {
            "o1": {"elements": ["A", "B", "C", "D"], "depends_on": ["s1"]},
        },
        "controls": {"c1": {"elements": ["up", "down"]}},
        "states": {
            "s1": {
                "elements": ["A", "B", "C", "D"],
                "depends_on_states": ["s1"],
                "depends_on_control": ["c1"],
            },
        },
    }

    As, Bs = compile_model(model)
