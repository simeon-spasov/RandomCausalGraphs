# README

Welcome to the RandomCausalGraphs repository! This guide will introduce you to the functionalities of the `CausalGraph` class, which is designed for creating synthetic causal graphs and simulating both observational and interventional data using various structural equation models (SEMs).

## Overview

`CausalGraph` is a Python class that allows users to generate random Directed Acyclic Graphs (DAGs) based on specified parameters. It supports various graph generation methods and SEM types, making it versatile for simulating complex causal structures in computational experiments.

### Supported Graph Types

- **Erdos-Renyi (ER)**: Randomly create edges between nodes with a fixed probability.
- **Barabasi-Albert (SF)**: Nodes are added sequentially and connected to existing nodes with preferential attachment. The parameter `m` is calculated based on the edge probability input parameter. See lines 72-78 in `synthetic_causal_graphs.py` for details.

### Supported SEM Types

The class supports a wide range of SEM types with additive and non-additive noise. `sem_type` parameter shown in brackets:

- **Linear SEMs**: A linear model with additive noise. The noise variable can be sampled from Gaussian ('gauss'), exponential ('exp'), Gumbel ('gumbel'), uniform ('uniform') distributions.
- **Non-linear SEMs**: multi-layer perceptron or multiple interaction model. Both have additive noise ('mlp' or 'mim') or non-additive noise versions ('mlp-non-add' or 'mim-non-add') respectively.
- **Discrete and other models**: 'logistic', 'poisson'
- **Gaussian processes**: 'gp' or 'gp-add' depending on whether parent nodes are modelled jointly in a multi-dimensional GP or independently by applying a GP to each parent node and then summing the result.

### Weight Ranges (w_ranges)

The `w_ranges` parameter specifying the range of weights for the causal links between nodes:

- **Default range**: `((-2.0, -0.5), (0.5, 2.0))`
  - Note weights are sampled from disjoint uniform distributions within these specified ranges.

## Usage

### Installation
To install libraries refer to `requirements.txt` file.

### Creating a Synthetic Graph

```python
from src.synthetic_causal_graphs import CausalGraph

# Initialize a CausalGraph with 100 nodes using Erdos-Renyi model and Gaussian SEM type
graph = CausalGraph(n_nodes=100, p=0.1, graph_type='ER', sem_type='gauss', seed=42)

```

### Sampling Data
#### Observational Data
To sample observational data from the graph:

```python
# Simulate observational data from the graph
# Shape of X is (n_samples, n_nodes). The columns follow the topological order of the nodes, that is 0, 1, 2,...n_nodes-1.
X = graph.simulate_sem(n_samples=500)
```

#### Interventional Data
To simulate data under an intervention (e.g., setting a node to a fixed value):

```python
# Simulate data with an intervention on node 10, setting its value to 0
# Shape of X_intervened is (n_samples, n_nodes). The columns follow the topological order of the nodes, that is 0, 1, 2,...n_nodes-1.
X_intervened = graph.simulate_sem(n_samples=500, intervened_node=10, intervened_value=0)
```

#### Fitness function
Allows you to generate an outcome `Y` from the samples. The fitness function applies a sparse mask to the columns of 
the input samples only retaining a `proportion` of the nodes as parents of the outcome `Y`. The sparse mask can be generated
using one of two strategies: `midpoint` or `last_few` depending on whether you want the parents to be around the middle or end of the topological order of nodes.
The outcome is generated as:

#### Fitness Function

The fitness function allows you to calculate an outcome `Y` from the samples by computing the weighted mean of selected variables and adding Gaussian noise. The selection of variables is based on a specified strategy, either `midpoint` or `last_few`. 

Mathematically, the fitness function can be described as:

Y = ( X ⊙ M )⋅θ + noise


where:
- \( X \) is the matrix of samples with shape (n_samples, n_nodes).
- \( M \) is a sparse mask matrix with the same shape as \( X \). It is used to select specific columns (variables) based on the sampling strategy.
- \( ⊙ \) denotes the element-wise multiplication.
- \( θ \) is the vector of weights for the selected variables.
- \( noise \) is Gaussian noise added to the weighted sum.

### Sampling Strategies

The possible strategies used to sample the parent nodes (variables) are:
1. **Midpoint**: Selects variables around the midpoint of the total number of nodes.
2. **Last Few**: Selects variables from the last few nodes.

### Weight Generation

The weights θ are sampled from a disjoint uniform distribution with default values ([-2.0, -0.5]) and ([0.5, 2.0]).

### Example Usage

Here is an example of how to use the `fitness` function:

```python
import numpy as np
from src.synthetic_causal_graphs import CausalGraph
from src.utils import fitness

# Generate samples using CausalGraph
n_nodes = 1000
p = 0.2
n_samples = 100
seed = 42
graph_type = 'ER'
sem_type = 'mlp'
model = CausalGraph(n_nodes, p, graph_type, sem_type, seed=seed)
X = model.simulate_sem(n_samples, noise_scale=1.0)

# Calculate fitness
fitness_values, sampled_indices, theta = fitness(X, noise_std=0.1, proportion=0.1, seed=seed, strategy='last_few')

print("Fitness Values:", fitness_values)
print("Sampled Indices (parent nodes):", sampled_indices)
print("Theta (Weights):", theta)
```

## Reproducibility
Using a fixed seed ensures that the generated graphs, sampled data, and fitness calculations are reproducible. Here's how the seed impacts different aspects of the simulation:

### Graph Generation
With a fixed seed, the structure of the generated graph remains the same across multiple class instantiations with the same input parameters. This includes the same set of nodes and edges.

### Sampling Observational Data
When sampling observational data, using the same seed ensures that the sampled data is identical across different runs of `simulate_sem`.

### Sampling Interventional Data
Similarly, using the same seed for interventional data ensures that the interventional samples are consistent across runs, provided the intervention is applied to the same node with the same value. Note this means the ancestors of the intervened node will have the same values as sampling observational data with the same seed. This is useful for calculating the counterfactual!

### Fitness Function
Using the same seed in the fitness function ensures that the same indices (parent nodes) and weights are sampled, resulting in consistent fitness values across runs. The mask matrix 
𝑀 and the weights 
𝜃 will be the same for repeated runs, ensuring that the computed fitness values are reproducible.

By setting the seed parameter consistently, you can ensure the repeatability of your experiments, making your simulations and results reliable and reproducible.

