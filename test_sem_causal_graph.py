import unittest

import networkx as nx
import numpy as np

from src.synthetic_causal_graphs import CausalGraph
from src.utils import fitness

SEM_TYPES = {'gauss', 'exp', 'gumbel', 'uniform', 'mlp', 'mlp-non-add', 'mim', 'mim-non-add', 'gp', 'gp-add'}
GRAPH_TYPES = {'ER', 'SF'}


class TestCausalGraph(unittest.TestCase):

    def setUp(self):
        """Setup base parameters for the tests. """
        self.n_nodes = 1000  # default number of nodes
        self.p = 0.2  # default probability
        self.n_samples = 100
        self.w_ranges = ((-2.0, -0.5), (0.5, 2.0))  # weight ranges
        self.noise_scale = 1.
        self.seed = 42

    def test_dag_generation(self):
        """
        Test ensures:
            1. generated graphs are DAGs
            2. Have specified number of nodes
            3. Empirical edge probability is close to input parameter.
        """
        for graph_type in GRAPH_TYPES:
            model = CausalGraph(self.n_nodes, self.p, graph_type=graph_type)
            self.assertTrue(nx.is_directed_acyclic_graph(model.model), f'{graph_type} graph is not a DAG')
            self.assertEqual(len(model.model.nodes), self.n_nodes,
                             f'{graph_type} graph does *not* have {self.n_nodes} nodes')

            total_possible_edges = self.n_nodes * (self.n_nodes - 1) / 2
            observed_edge_prob = len(model.model.edges) / total_possible_edges
            expected_prob_delta = 0.01 if graph_type == 'ER' else 0.02
            self.assertAlmostEqual(observed_edge_prob, self.p, None,
                                   f'{graph_type} has too many or too few edges.', delta=expected_prob_delta)

    def check_weights(self, weights, shapes, ranges, sem_type):
        self.assertEqual(len(weights), len(shapes),
                         f'Expected {len(shapes)} weight sets, but got {len(weights)} for {sem_type}')
        for w, shape in zip(weights, shapes):
            self.assertEqual(w.shape, shape, f'Expected shape {shape}, but got {w.shape} for {sem_type}')
            # Ensure each element in W falls within at least one range
            if shapes[0][0] > 0:  # If parents
                for w_ in np.nditer(w):
                    self.assertTrue(any(low <= w_ <= high for low, high in ranges),
                                    f"Value {w_} does not fall within any of the specified ranges for {sem_type}")

    def test_generate_weights(self):
        """
        Checks generated weights have expected shape and fall within w_ranges.
        """
        for sem_type in SEM_TYPES:
            if 'gp' not in sem_type:  # GP Regressor uses no weights
                model = CausalGraph(n_nodes=self.n_nodes, p=self.p, graph_type='ER', w_ranges=self.w_ranges,
                                    sem_type=sem_type)
                for node in model.model.nodes():
                    weights = model.W[node]
                    num_parents = model.model.in_degree(node)
                    expected_shapes = {
                        'gauss': [(num_parents,)],
                        'exp': [(num_parents,)],
                        'uniform': [(num_parents,)],
                        'gumbel': [(num_parents,)],
                        'mlp': [(num_parents, 10), (10, 10), (10,)],
                        'mlp-non-add': [(num_parents + 1, 10), (10, 10), (10,)],
                        'mim': [(num_parents,), (num_parents,), (num_parents,)],
                        'mim-non-add': [(num_parents + 1,), (num_parents + 1,), (num_parents + 1,)],
                    }
                    self.check_weights(weights, expected_shapes[sem_type], self.w_ranges, sem_type)

    def test_simulate_sem_shape(self):
        """
        Tests if shape of generated samples matches expected shape.
        """
        for graph_type in GRAPH_TYPES:
            for sem_type in SEM_TYPES:
                model = CausalGraph(self.n_nodes, self.p, graph_type=graph_type, sem_type=sem_type,
                                    w_ranges=((-2.0, -0.5), (0.5, 2.0)))
                X = model.simulate_sem(self.n_samples, noise_scale=self.noise_scale)
                self.assertEqual(X.shape, (self.n_samples, self.n_nodes))

    def test_simulate_sem(self):
        """
        Checks if function simulate_sem works as expected. We compare the output of simulate_sem against
        the expected output for several SEMs. Since the noise for these SEMs is additive and
        Gaussian, we can check if abs|X - expected_X| falls within 3 standard deviations for 99% of samples.

        Note we do not have explicit checks for non-additive noise models! We rely on the fact the logic is identical
        in simulate_sem for both additive and non-additive models.
        """
        n_nodes = 20  # Fewer nodes to speed up test
        n_samples = 10000  # More samples for statistical significance
        p = 0.4  # Higher edge probability makes sure graph has 1 component.
        threshold = 3  # Checking within three standard deviations

        def linear(inputs, W):
            return inputs @ W[0]

        def apply_nn(inputs, W):
            h = np.maximum(0, inputs @ W[0])  # ReLU activation
            h2 = np.maximum(0, h @ W[1])  # Additional hidden layer with ReLU activation
            return h2 @ W[2]

        def apply_mim(inputs, W):
            return np.tanh(inputs @ W[0]) + np.cos(inputs @ W[1]) + np.sin(inputs @ W[2])

        apply_function = {
            'gauss': linear,
            'uniform': linear,
            'mlp': apply_nn,
            'mim': apply_mim,
        }

        for graph_type in ['ER', 'SF']:
            for sem_type in ['gauss', 'mlp', 'mim', 'uniform']:
                model = CausalGraph(n_nodes, p, graph_type=graph_type, sem_type=sem_type,
                                    w_ranges=self.w_ranges)
                X = model.simulate_sem(n_samples, noise_scale=self.noise_scale)
                expected_X = np.ones([n_samples, model.n_nodes]) * np.nan

                topological_order = list(nx.topological_sort(model.model))

                for node in topological_order:
                    parents = list(model.model.predecessors(node))
                    parent_values = X[:, parents]
                    W = model.W[node]

                    if parents:
                        magnitude = np.mean(np.abs(parent_values), axis=1) * self.noise_scale
                    else:
                        magnitude = np.ones(n_samples) * self.noise_scale

                    expected_X[:, node] = apply_function[sem_type](parent_values, W)

                    differences = np.abs(X[:, node] - expected_X[:, node])

                    # Check if differences fall within the expected Gaussian noise range
                    acceptable_range = threshold * magnitude
                    percent_within_range = np.mean(differences <= acceptable_range) * 100

                    self.assertTrue(percent_within_range >= 99,  # Should be 99.7 but we relax requirement.
                                    msg=f"{percent_within_range}% of values exceed 3σ for node {node} in {graph_type} with {sem_type}. Should be at least 99%.")

    def test_intervention(self):
        """
        Check if value of intervened node matches the intervened_value parameter.
        """
        for graph_type in GRAPH_TYPES:
            model = CausalGraph(self.n_nodes, self.p, graph_type=graph_type, sem_type='mlp-non-add',
                                w_ranges=self.w_ranges)

            n_sampled_nodes = int(self.n_nodes * 0.2)
            sampled_nodes = np.random.choice(self.n_nodes, size=n_sampled_nodes, replace=False)

            for intervened_node in sampled_nodes:
                intervened_value = 0
                X = model.simulate_sem(
                    self.n_samples,
                    noise_scale=self.noise_scale,
                    intervened_node=intervened_node,
                    intervened_value=intervened_value)

                self.assertEqual(X.shape, (self.n_samples, self.n_nodes))

                np.testing.assert_allclose(X[:, intervened_node], np.full(self.n_samples, intervened_value), rtol=0,
                                           atol=0)

    def test_graph_reproducibility(self):
        """Ensure graph structure and weights are consistent across instances with the same seed."""
        graph_type = 'ER'
        sem_type = 'mlp'
        model1 = CausalGraph(self.n_nodes, self.p, graph_type, sem_type, seed=self.seed)
        model2 = CausalGraph(self.n_nodes, self.p, graph_type, sem_type, seed=self.seed)
        # Check graph structure
        self.assertTrue(nx.is_isomorphic(model1.model, model2.model))
        # Check weights
        for node in model1.model.nodes():
            if model1.model.in_degree(node) > 0:  # Check only nodes with parents
                w1 = np.concatenate([w.flatten() for w in model1.W[node]]) if isinstance(model1.W[node], list) else \
                    model1.W[node]
                w2 = np.concatenate([w.flatten() for w in model2.W[node]]) if isinstance(model2.W[node], list) else \
                    model2.W[node]
                self.assertTrue(np.array_equal(w1, w2),
                                f"Weights for node {node} differ between model instances.\n"
                                f"Weights from model 1: {w1}\n"
                                f"Weights from model 2: {w2}")

    def test_sample_reproducibility(self):
        """Consecutive observational samples should be identical for the same model."""
        graph_type = 'ER'
        sem_type = 'mlp'
        model = CausalGraph(self.n_nodes, self.p, graph_type, sem_type, seed=self.seed)
        X1 = model.simulate_sem(self.n_samples, noise_scale=self.noise_scale)
        X2 = model.simulate_sem(self.n_samples, noise_scale=self.noise_scale)
        self.assertTrue((X1 == X2).all(), "Consecutive samples should be identical.")

    def test_intervention_reproducibility(self):
        """Pre-intervention ancestor nodes should maintain identical values for consecutive samples.
        Even if the first sample is from the unmodified (observational) distribution and subsequent samples
        are from an interventional distribution."""
        graph_type = 'ER'
        sem_type = 'mlp'
        model = CausalGraph(self.n_nodes, self.p, graph_type, sem_type, seed=self.seed)
        intervened_node = 50  # Adjust node number as needed
        X_obs = model.simulate_sem(self.n_samples, noise_scale=self.noise_scale)
        X_int = model.simulate_sem(self.n_samples, intervened_node=intervened_node, intervened_value=0,
                                   noise_scale=self.noise_scale)
        ancestors = nx.ancestors(model.model, intervened_node)
        for ancestor in ancestors:
            self.assertTrue((X_obs[:, ancestor] == X_int[:, ancestor]).all(),
                            f"Values for ancestor node {ancestor} should not change after intervention.")

    def test_fitness_reproducibility(self):
        """Fitness function should sample the same indices and weights with the same input seed."""
        graph_type = 'ER'
        sem_type = 'mlp'
        intervened_node = 50

        model = CausalGraph(self.n_nodes, self.p, graph_type, sem_type, seed=self.seed)

        X_obs = model.simulate_sem(self.n_samples, noise_scale=self.noise_scale)

        Y_obs, indices_obs, weights_obs = fitness(X_obs, noise_std=0.1, proportion=0.1, seed=self.seed,
                                                  strategy='last_few')

        X_int = model.simulate_sem(self.n_samples, intervened_node=intervened_node, intervened_value=0,
                                   noise_scale=self.noise_scale)

        Y_int, indices_int, weights_int = fitness(X_int, noise_std=0.1, proportion=0.1, seed=self.seed,
                                                  strategy='last_few')

        # Check if results, indices, and weights are identical
        self.assertTrue((indices_obs == indices_int).all(), "Sampled indices should be identical for fitness "
                                                            "function in observational and interventional "
                                                            "experiments.")

        self.assertTrue((weights_obs == weights_int).all(), "Sampled weights should be identical for fitness "
                                                            "function in observational and interventional "
                                                            "experiments.")

        self.assertTrue((Y_int != Y_obs).all(), "Observational and interventional outcomes should differ.")

        Y_obs_2, indices_obs_2, weights_obs_2 = fitness(X_obs, noise_std=0.1, proportion=0.1, seed=self.seed,
                                                        strategy='last_few')
        self.assertTrue((Y_obs == Y_obs_2).all(), "Fitness outcome should be identical for same observational samples.")


if __name__ == '__main__':
    unittest.main()
