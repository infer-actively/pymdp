import networkx as nx
from jax import numpy as jnp, random as jr, tree_util as jtu
from typing import Optional, List, Tuple
from jaxtyping import PRNGKeyArray
from .env import PymdpEnv
import warnings

def generate_connected_clusters(cluster_size=2, connections=2):
    edges = []
    connecting_node = 0
    while connecting_node < connections * cluster_size:
        edges += [(connecting_node, a) for a in range(connecting_node + 1, connecting_node + cluster_size + 1)]
        connecting_node = len(edges)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph, {
        "locations": [
            (f"hallway {i}" if len(list(graph.neighbors(loc))) > 1 else f"room {i}")
            for i, loc in enumerate(graph.nodes)
        ]
    }

class GraphEnv(PymdpEnv):
    """
    A simple environment where an agent can move around a graph and search an object.
    The agent observes its own location, as well as whether the object is at its location.
    """

    def __init__(self, graph: nx.Graph, object_location: Optional[int] = None, agent_location: Optional[int] = None, key: Optional[PRNGKeyArray] = None):

        A, A_dependencies = self.generate_A(graph)
        B, B_dependencies = self.generate_B(graph)

        if object_location is None:
            key = jr.PRNGKey(0) if key is None else key
            _, _key = jr.split(key)
            object_location = jr.randint(_key, shape=(), minval=0, maxval=len(graph.nodes) + 1)  # +1 for "not here"
        if agent_location is None:
            key = jr.PRNGKey(1) if key is None else key
            _, _key = jr.split(key)
            agent_location = jr.randint(_key, shape=(), minval=0, maxval=len(graph.nodes))

        D = self.generate_D(graph, object_location, agent_location)

        super().__init__(A=A, B=B, D=D, A_dependencies=A_dependencies, B_dependencies=B_dependencies)

    def generate_A(self, graph: nx.Graph):
        A = []
        A_dependencies = []

        num_locations = len(graph.nodes)
        num_object_locations = num_locations + 1  # +1 for "not here"
        p = 1.0  # probability of seeing object if it is at the same location as the agent

        # agent location modality
        A.append(jnp.eye(num_locations))
        A_dependencies.append([0])

        # object visibility modality
        A.append(jnp.zeros((2, num_locations, num_object_locations)))

        for agent_loc in range(num_locations):
            for object_loc in range(num_locations):
                if agent_loc == object_loc:
                    # object seen
                    A[1] = A[1].at[0, agent_loc, object_loc].set(1 - p)
                    A[1] = A[1].at[1, agent_loc, object_loc].set(p)
                else:
                    A[1] = A[1].at[0, agent_loc, object_loc].set(p)
                    A[1] = A[1].at[1, agent_loc, object_loc].set(1.0 - p)

        # object not here, we can't see it anywhere
        A[1] = A[1].at[0, :, -1].set(1.0)
        A[1] = A[1].at[1, :, -1].set(0.0)

        A_dependencies.append([0, 1])
        return A, A_dependencies

    def generate_B(self, graph: nx.Graph):
        B = []
        B_dependencies = []

        num_locations = len(graph.nodes)
        num_object_locations = num_locations + 1

        # agent location transitions, based on graph connectivity
        B.append(jnp.zeros((num_locations, num_locations, num_locations)))
        for action in range(num_locations):
            for from_loc in range(num_locations):
                for to_loc in range(num_locations):
                    if action == to_loc:
                        # we transition if connected in graph
                        if graph.has_edge(from_loc, to_loc):
                            B[0] = B[0].at[to_loc, from_loc, action].set(1.0)
                        else:
                            B[0] = B[0].at[from_loc, from_loc, action].set(1.0)

        B_dependencies.append([0])

        # objects don't move
        B.append(jnp.zeros((num_object_locations, num_object_locations, 1)))
        B[1] = B[1].at[:, :, 0].set(jnp.eye(num_object_locations))
        B_dependencies.append([1])

        return B, B_dependencies
    
    def generate_D(self, graph: nx.Graph, object_location: int, agent_location: int):

        num_locations = len(graph.nodes)
        num_object_locations = num_locations + 1

        states = [num_locations, num_object_locations]
        D = [jnp.zeros(s) for s in states]

        # set the start locations
        D[0] = D[0].at[agent_location].set(1.0)
        D[1] = D[1].at[object_location].set(1.0)

        return D
    
    def generate_env_params(self,  graph: nx.Graph, key=None, object_locations: Optional[List[int]]=None, agent_locations: Optional[List[int]]=None, batch_size: Optional[int]=None):
        """
        Override of `generate_env_params` from `Env` class that returns batched environmental parameters, with
        the option to randomize initial object and agent locations (lists of integers) which, if provided, should be of length `batch_size`
        and are used to set the initial locations in the D vector for each batch element.
        
        If batch size is provided but object_locations and agent_locations are not, then random locations will be sampled for each batch element.
        args:
            graph: networkx Graph object representing the environment
            key: JAX random key (used to sample random initial locations if object_locations and agent_locations are not provided)
            object_locations: list of integers specifying initial object locations for each batch element
            agent_locations: list of integers specifying initial agent locations for each batch element
            batch_size: integer specifying the number of batch elements. If None but object_locations and agent_locations are provided, batch_size is inferred from their length.
        returns:
            env_params: dictionary containing batched environmental parameters
        """
        if batch_size is None:
            if object_locations is not None and agent_locations is not None:
                batch_size = len(object_locations)
            else:
                warnings.warn("Neither `batch_size` nor `object_locations` nor `agent_locations` are provided, so just returning the unbatched env params")
                return super().generate_env_params(key=key, batch_size=batch_size)

        num_locations = len(graph.nodes)
        num_object_locations = num_locations + 1

        if object_locations is None or agent_locations is None:
            if key is None:
                raise ValueError("A random key must be provided to sample random initial locations.")
            keys = jr.split(key, 2)
            if object_locations is None:
                object_locations = jr.randint(keys[0], shape=(batch_size,), minval=0, maxval=num_object_locations)
            if agent_locations is None:
                agent_locations = jr.randint(keys[1], shape=(batch_size,), minval=0, maxval=num_locations)

        D = [jnp.zeros((batch_size, s)) for s in [num_locations, num_object_locations]]

        D[0] = D[0].at[jnp.arange(batch_size), agent_locations].set(1.0)
        D[1] = D[1].at[jnp.arange(batch_size), object_locations].set(1.0)

        expand_to_batch = lambda x: jnp.broadcast_to(jnp.asarray(x), (batch_size,) + x.shape)

        env_params ={**jtu.tree_map(expand_to_batch, {"A": self.A, "B": self.B}), **{"D": D}}
        
        return env_params
