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


def compile(config):
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
            for kw in v:
                match kw:
                    case "elements":
                        shape[k] = len(v[kw])
                        labels[k] = [name for name in v[kw]]
                    case "size":
                        shape[k] = v[kw]
                        labels[k] = list(range(v[kw]))
                    case "depends_on_states":
                        state_dependencies[k] = [name for name in v[kw]]
                        if k in v[kw]:
                            transition_events[k] = labels[k]
                    case "depends_on_control":
                        control_dependencies[k] = [name for name in v[kw]]
                    case "depends_on":
                        likelihood_dependencies[k] = [name for name in v[kw]]
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
    return transition, likelihoods


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
    trans, like = compile(model_example)
