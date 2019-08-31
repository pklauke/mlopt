# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains classes for maximizing or minimizing a given metric. Currently implemented classes are:

BlendingSwarmTransformer: Uses particle swarm optimization.
BlendingGreedyTransformer: Uses a greedy approach.
"""

import numpy as np

import mlopt.optimization


class BlendingTransformer:
    """Optimizer to minimize or maximize an objective metric using Particle Swarm Optimization.

    :param metric: Callable function to optimize.
    :param maximize: Boolean indicating whether `metric` wants to be maximized or minimized.
    """
    def __init__(self, metric, maximize, optimizer='greedy'):
        self.metric = metric
        self.X = None
        self.y = None

        def _func(weights):
            return self.metric(self.y, np.average(self.X, axis=0, weights=weights))

        if optimizer.lower() == 'pso':
            self.optimizer = mlopt.optimization.ParticleSwarmOptimizer(func=_func, maximize=maximize)
        elif optimizer.lower() == 'greedy':
            self.optimizer = mlopt.optimization.GreedyOptimizer(func=_func, maximize=maximize)
        else:
            if hasattr(optimizer, 'optimizer'):
                self.optimizer = optimizer
            else:
                raise AttributeError('Provided optimizer does not have a optimize method.')

    def fit(self, X, y, iterations=100, random_state=None, params=None):
        """Fit the model on the given predictions.

        :param X: Predictions of different models for the labels.
        :param y: Labels.
        :param iterations: Number of iterations.
        :param random_state: Random state for initializing.
        :param params: Dictionary containing the variable names and their value ranges. Key is expected to be the
                       variable name and value a tuple containing the minimum and maximum value of the variable.
        :return: Optimized weights."""
        self.X = X
        self.y = y
        if params is None:
            params = {'x' + str(i): (0, 1) for i in range(np.shape(X)[0])}
        self.optimizer.optimize(params=params, iterations=iterations, random_state=random_state)

        return self

    def transform(self, X):
        """Transform blended predictions using the trained weights.

        :param X: Predictions.
        :return: Blended predictions.
        """
        return np.average(X, axis=0, weights=self.optimizer.coords)

    def fit_transform(self, X, y, **kwargs):
        """Fit transformer to X, then transform X. See `fit` and `transform` for further explanation."""
        return self.fit(X=X, y=y, **kwargs).transform(X=X)
