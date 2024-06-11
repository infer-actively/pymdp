import networkx as nx
import jax.numpy as jnp

from .env import PyMDPEnv


class GraphEnv(PyMDPEnv):
    """
    A simple environment where an agent can move around a graph and search an object.
    The agent observes its own location, as well as whether the object is at its location.
    """

    def __init__(self, graph: nx.Graph, object_locations: list[int], agent_locations: list[int]):
        batch_size = len(object_locations)

        A, A_dependencies = self.generate_A(graph)
        A = [jnp.broadcast_to(a, (batch_size,) + a.shape) for a in A]
        B, B_dependencies = self.generate_B(graph)
        B = [jnp.broadcast_to(b, (batch_size,) + b.shape) for b in B]
        D = self.generate_D(graph, object_locations, agent_locations)

        params = {
            "A": A,
            "B": B,
            "D": D,
        }

        dependencies = {
            "A": A_dependencies,
            "B": B_dependencies,
        }

        super().__init__(params, dependencies)

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

    def generate_D(self, graph: nx.Graph, object_locations: list[int], agent_locations: list[int]):
        batch_size = len(object_locations)
        num_locations = len(graph.nodes)
        num_object_locations = num_locations + 1

        states = [num_locations, num_object_locations]
        D = []
        for s in states:
            D.append(jnp.zeros((batch_size, s)))

        # set the start locations
        for i in range(batch_size):
            D[0] = D[0].at[i, agent_locations[i]].set(1.0)
            D[1] = D[1].at[i, object_locations[i]].set(1.0)

        return D
