import numpy as np


def fitness(samples: np.ndarray, noise_std: float, proportion: float, seed: int = None,
            strategy: str = "midpoint") -> tuple:
    """
    Calculate the fitness of each sample by computing the weighted mean of selected variables and adding Gaussian noise.
    The selection of variables can be based on a specified strategy: 'midpoint' or 'last_few'.

    :param samples: A NumPy array of shape (n_samples, n_nodes) representing the samples.
    :param noise_std: The standard deviation of the Gaussian noise to be added to the fitness values.
    :param proportion: Proportion of nodes to sample (from the middle for 'midpoint' or from the last for 'last_few').
    :param seed: Optional seed value for reproducible random sampling and noise.
    :param strategy: Sampling strategy ('midpoint' or 'last_few').

    :return: A tuple containing:
        - np.ndarray: Fitness values of shape (n_samples,).
        - np.ndarray: Indices of the sampled nodes.
        - np.ndarray: Weights corresponding to the sampled nodes.
    """
    if np.isnan(samples).any():
        raise ValueError("There are NaN values in the samples array.")

    # Validate strategy and proportion inputs
    valid_strategies = ['midpoint', 'last_few']
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy. Expected one of {valid_strategies}, but got '{strategy}'.")

    if strategy == "midpoint":
        if not (0 < proportion <= 0.2):
            raise ValueError("Proportion for 'midpoint' must be in the range (0, 0.2].")
    elif strategy == "last_few":
        if not (0 < proportion <= 1):
            raise ValueError("Proportion for 'last_few' must be in the range (0, 1].")

    n_samples, n_nodes = samples.shape
    rng = np.random.default_rng(seed)

    # Select indices based on the chosen strategy
    if strategy == "midpoint":
        midpoint = n_nodes // 2
        range_width = 0.1 * n_nodes  # 10% on either side of the midpoint
        start = int(midpoint - range_width)
        end = int(midpoint + range_width)
        num_to_sample = int(np.ceil(proportion * (end - start)))
        sampled_indices = rng.choice(np.arange(start, end), num_to_sample, replace=False)
    elif strategy == "last_few":
        last_n_columns = int(np.ceil(proportion * n_nodes))
        sampled_indices = np.arange(n_nodes - last_n_columns, n_nodes)

    # Use the sampled indices to slice the sample array
    samples = samples[:, sampled_indices]

    # Generate weights and noise
    theta = sample_disjoint_uniform([(-2.0, -0.5), (0.5, 2.0)], (len(sampled_indices),), rng=rng)
    noise = rng.normal(loc=0, scale=noise_std, size=n_samples)
    fitness_values = samples @ theta + noise

    return fitness_values, sampled_indices, theta


def sample_disjoint_uniform(w_ranges, size, rng=None):
    """
    Sample disjoint uniform distributions from given ranges and returns a
    numpy array of specified size.

    Parameters
    ----------
    w_ranges : list of tuple
        A list of tuples each containing two elements specifying the lower
        and upper bounds for the uniform distribution to be sampled from.
        Each tuple denotes a distinct range from which values are sampled.
        E.g. ((-2.0, -0.5), (0.5, 2.0))

    size : tuple
        The desired output shape for the numpy array 'W'. E.g. (5, 5)

    rng : numpy.random.Generator, optional
        An instance of numpy's random number generator. If None, a new generator
        will be created for each call, leading to non-reproducible results.
        Defaults to None.

    Returns
    -------
    W : numpy.ndarray
        An array of the specified 'size' containing values sampled from
        uniform distributions corresponding to the ranges specified in
        'w_ranges'. The shape of 'W' is the same as the provided 'size'.

    Notes
    -----
    The function first creates an array 'S' of random integers corresponding
    to indices of 'w_ranges'. For each unique value in 'S', it samples a
    uniform distribution of the same shape as 'size' based on the corresponding
    range in 'w_ranges', and assigns these sampled values into the final
    output array 'W' at the positions where 'S' equals the current index.
    """

    if rng is None:
        rng = np.random.default_rng()  # Create a new generator instance if none provided

    W = np.zeros(size)
    S = rng.integers(len(w_ranges), size=size)  # which range to sample values from
    for i, (low, high) in enumerate(w_ranges):
        U = rng.uniform(low=low, high=high, size=size)
        W += (S == i) * U
    return W
