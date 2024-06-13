from typing import Optional

import networkx as nx
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.gaussian_process import GaussianProcessRegressor

from .utils import sample_disjoint_uniform

SEM_TYPES = {'gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson', 'mlp', 'mlp-non-add', 'mim', 'mim-non-add',
             'gp', 'gp-add'}
GRAPH_TYPES = {'ER', 'SF'}


class CausalGraph:
    def __init__(self,
                 n_nodes: int,
                 p: float,
                 graph_type='ER',
                 sem_type='gauss',
                 w_ranges=((-2.0, -0.5), (0.5, 2.0)),
                 seed=None
                 ):
        """
        Initialize the CausalGraph instance with a random DAG based on the input parameters.
        :param n_nodes: The number of nodes in the DAG.
        :param p: The probability of creating an edge between two nodes in range [0, 1].
        :param graph_type: The type of graph generation method. Default is 'ER' for Erdos-Renyi.
        :param w_ranges: Disjoint weight ranges for generating edge weights. Default is ((-2.0, -0.5), (0.5, 2.0)).
        :param seed: The seed for random number generators to ensure reproducibility.
        """
        if sem_type not in SEM_TYPES:
            raise ValueError('unknown SEM type')

        assert graph_type in GRAPH_TYPES, "Graph type must be one of 'ER', 'SF'"

        self.rng = np.random.default_rng(seed)
        self.initial_rng_state = self.rng.bit_generator.state  # Save the initial state of the RNG
        self.n_nodes = n_nodes
        self.p = p
        self.graph_type = graph_type
        self.sem_type = sem_type
        self.w_ranges = w_ranges
        self.model = self.create_random_dag()
        self.W = self.generate_weights()

    def create_random_dag(self) -> nx.DiGraph:
        """
        Create a random directed acyclic graph (DAG) based on the input parameters and graph type.

        :return: A random networkx DiGraph instance.
        """

        def add_edges_to_ensure_acyclic(G: nx.Graph) -> nx.DiGraph:
            """
            Add edges to a graph and orient them from lower-numbered to higher-numbered nodes to ensure it's acyclic.

            :param G: A networkx Graph instance.
            :return: A networkx DiGraph instance.
            """
            DAG = nx.DiGraph()
            DAG.add_nodes_from(G.nodes())
            for edge in G.edges():
                if edge[0] < edge[1]:
                    DAG.add_edge(edge[0], edge[1])
                else:
                    DAG.add_edge(edge[1], edge[0])
            return DAG

        if self.graph_type == 'ER':
            G = nx.erdos_renyi_graph(self.n_nodes, self.p, seed=self.rng)
        elif self.graph_type == 'SF':
            # Barabasi-Albert graph with preferential attachment
            # Each newly added node is attached to m existing nodes. If we start with m nodes, we get E=m(n-m) edges.
            # We can calculate the edge probability in any graph as p = 2*E/(n*(n-1)), where E is number of edges.
            # It is just the edges E divided by the total possible number of edges given by n*(n-1)/2.
            # We substitute E from BA model and get m ~ p*n/2 for n>>m.
            m = round(self.p * self.n_nodes / 2)
            if m < 1:
                m = 1
                print('Edge probability and n_nodes parameters result in m<1 edge attachments for SF graph.'
                      'm has been set to 1.'
                      'Change edge probability p or n_nodes for different behaviour.')
            G = nx.barabasi_albert_graph(self.n_nodes, m=m, seed=self.rng)

        DAG = add_edges_to_ensure_acyclic(G)

        assert nx.is_directed_acyclic_graph(DAG), "The graph is not a directed acyclic graph (DAG)."
        assert nx.is_weakly_connected(DAG), "The graph is not weakly connected. Try creating another instance again." \
                                            "A directed graph is weakly connected if, and only if, the graph is " \
                                            "connected when the direction of the edge between nodes is ignored."

        return DAG

    def generate_weights(self):
        """
        Generate weights for each node in the graph.

        For each node in the graph, it finds the number of parents (in-degree) and generates weights
        for these parents using the _generate_weights function. It stores these weights in a dictionary
        with the node as the key.

        Returns:
            dict: A dictionary mapping each node to its weight vector.
        """
        W = {}
        for node in self.model.nodes():
            num_parents = self.model.in_degree(node)  # obtain the number of parents
            W[node] = self._generate_weights(num_parents, self.sem_type)
        return W

    def _generate_weights(self, num_parents, sem_type):
        """
        Generate weights for the parent values given the Structural Equation Model (SEM) type.

        For each sem_type, weights are generated using the function sample_disjoint_uniform
        with different sizes according to the following rules:
            1. If sem_type is one of ['gauss', 'exp', 'uniform', 'gumbel', 'logistic', 'poisson'],
               one set of weights are generated with shape (num_parents, ).
            2. If sem_type is 'mlp' or 'mlp-non-add', two sets of weights are generated with
               shapes (num_parents, 10) and (10, ),
               or (num_parents + 1, 10) and (10, ) respectively.
            3. If sem_type is 'mim' or 'mim-non-add', three sets of weights are generated
               each with shape (num_parents, 1) or (num_parents+ 1, 1) respectively.

        :param parent_values: A 2D numpy array of parent values with shape (n_samples, num_parents).
        :param sem_type: A string indicating the type of the SEM.
                         It should be one of ['gauss', 'exp', 'uniform', 'gumbel', 'logistic', 'gp', 'gp-add',
                                             'poisson', 'mlp', 'mlp-non-add', 'mim', 'mim-non-add'].

        :return: A list of 2D numpy arrays. Each array is a set of weights corresponding to
                 parent_values. The shape of each array depends on sem_type as described above.
        """

        def sample_weights(size):
            return sample_disjoint_uniform(self.w_ranges, size=size, rng=self.rng)

        def mlp_size(sem_type):
            hidden = 10  # Hidden units in MLP
            if sem_type == 'mlp':
                return (num_parents, hidden)
            else:
                return (num_parents + 1, hidden)

        def mim_size(sem_type):
            if sem_type == 'mim':
                return (num_parents,)
            else:
                return (num_parents + 1,)

        if sem_type in ['gauss', 'exp', 'uniform', 'gumbel', 'logistic', 'poisson']:
            return [sample_weights(num_parents, )]

        elif 'mlp' in sem_type:
            return [sample_weights(mlp_size(sem_type)), sample_weights((10, 10)), sample_weights((10,))]

        elif 'mim' in sem_type:
            return [sample_weights(mim_size(sem_type)) for _ in range(3)]

        else:
            return None

    def _simulate_single_equation(self, parent_values, W, n_samples, scale):
        """
        Simulate a single equation given the input matrix, weights, and noise scale.

        :param parent_values: The input matrix with shape (n_samples, num_parents).
        :param W: The weights vector. Its shape depends on sem_type and parent_values' shape.
        :param n_samples: The number of samples to generate.
        :param scale: The noise scale. Scalar or array of length (n_nodes)
        :return: The resulting values after applying the equation with shape (n_samples,).
        """

        def linear(inputs, W):
            return inputs @ W[0]

        def apply_nn(inputs, W):
            h = np.maximum(0, inputs @ W[0])  # ReLU activation
            h2 = np.maximum(0, h @ W[1])  # Additional hidden layer with ReLU activation
            return h2 @ W[2]

        def apply_mim(inputs, W):
            return np.tanh(inputs @ W[0]) + np.cos(inputs @ W[1]) + np.sin(inputs @ W[2])

        def apply_noise(sem_type, scale, size):
            if sem_type in ['gauss', 'mlp', 'mlp-non-add', 'mim', 'mim-non-add', 'gp', 'gp-add']:
                return self.rng.normal(scale=scale, size=size)
            elif sem_type == 'exp':
                return self.rng.exponential(scale=scale, size=size)
            elif sem_type == 'gumbel':
                return self.rng.gumbel(scale=scale, size=size)
            elif sem_type == 'uniform':
                return self.rng.uniform(low=-scale, high=scale, size=size)
            elif sem_type == 'logistic':
                return self.rng.binomial(1, 0.5 * np.ones(size)) * 1.0
            elif sem_type == 'poisson':
                return self.rng.poisson(size=size) * 1.0
            else:
                raise ValueError(f"Unknown SEM type {sem_type}")

        def add_noise(out, z):
            if 'non-add' not in self.sem_type:
                out += z
            return out

        apply_function = {
            'gauss': linear,
            'exp': linear,
            'gumbel': linear,
            'uniform': linear,
            'logistic': lambda x, W: self.rng.binomial(1, sigmoid(linear(x, W))) * 1.0,
            'poisson': lambda x, W: self.rng.poisson(np.exp(linear(x, W))) * 1.0,
            'mlp': apply_nn,
            'mlp-non-add': apply_nn,
            'mim': apply_mim,
            'mim-non-add': apply_mim,
        }

        pa_size = parent_values.shape[1]

        inputs = parent_values

        # Calculate the magnitude of the parent values
        if pa_size > 0:
            magnitude = np.mean(np.abs(parent_values), axis=1)
        else:
            magnitude = np.ones(n_samples)

        z = apply_noise(self.sem_type, scale * magnitude, n_samples)

        if 'non-add' in self.sem_type:
            if pa_size > 0:
                inputs = np.hstack((parent_values, z.reshape(n_samples, -1)))
            else:
                inputs = z.reshape(n_samples, -1)

        if self.sem_type not in ['gp', 'gp-add']:
            out = apply_function[self.sem_type](inputs, W)

        else:  # If sem_type in ['gp', 'gp-add']
            out = np.zeros(n_samples, )  # If no parents, return 0s
            if pa_size > 0:
                random_state_gp = np.random.RandomState(self.rng.integers(0, 2 ** 31 - 1))
                gp = GaussianProcessRegressor(random_state=random_state_gp)
                if self.sem_type == 'gp':
                    out = gp.sample_y(parent_values, random_state=random_state_gp).flatten()
                else:  # sem_type == 'gp-add'
                    out = sum([gp.sample_y(parent_values[:, i, None], random_state=random_state_gp).flatten()
                               for i in range(parent_values.shape[1])])

        out = add_noise(out, z)

        return out

    def simulate_sem(
            self,
            n_samples: int,
            intervened_node: Optional[int] = None,
            intervened_value: Optional[float] = None,
            noise_scale: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate samples from the DAG following linear SEM with a noise variable.
        Can also perform a hard intervention on a given node if intervened_node and intervened_value are provided.

        :param n_samples: The number of samples to generate.
        :param noise_scale: The scale parameter of the additive noise.
        :param intervened_node: Optional. The node to intervene on.
        :param intervened_value: Optional. The value to set the intervened node to.
        :return: A NumPy array containing the samples of shape (n_samples, n_nodes).
        """

        self.rng.bit_generator.state = self.initial_rng_state  # Reset RNG state before each simulation

        if noise_scale is None:
            scale_vec = np.ones(self.n_nodes)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(self.n_nodes)
        else:
            if len(noise_scale) != self.n_nodes:
                raise ValueError('noise scale must be a scalar or has length number of nodes')
            scale_vec = noise_scale

        model = self.model.copy()

        if intervened_node is not None:
            if intervened_node not in model.nodes():
                raise ValueError(f"Intervened node {intervened_node} is not in the set of nodes of the model.")
            model.remove_edges_from(list(model.in_edges(intervened_node)))

        topological_order = list(nx.topological_sort(model))
        X = np.ones([n_samples, self.n_nodes]) * np.nan

        for node in topological_order:
            if node == intervened_node:
                X[:, intervened_node] = intervened_value
            else:
                parents = list(model.predecessors(node))
                parent_values = X[:, parents]

                assert not np.isnan(parent_values).any(), \
                    "Parent values contain NaN. Child node is accessing parent value before initialization."

                W = self.W[node]
                X[:, node] = self._simulate_single_equation(parent_values, W, n_samples, scale_vec[node])
        return X
