# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains unit tests for testing the mlopt.blending.ParticleSwarmOptimizer class."""

import numpy as np

from mlopt.optimization import GreedyOptimizer


def opt_func(*args):
    """Function for minimization testing."""
    return sum(arg**2 for arg in args)


def opt_func_inv(*args):
    """Function for maximization testing."""
    return - opt_func(*args)


def test_init_correct_dimensions_best_coords_glob():
    """Test if the initialized best coordinates of all particles combined have the correct dimensions."""
    optimizer = GreedyOptimizer(func=opt_func, maximize=False)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    optimizer.init(params=params, random_state=1)

    assert optimizer.coords.shape == (2,)


def test_init_correct_dimensions_best_score_glob():
    """Test if the initialized best score has the correct dimension."""
    optimizer = GreedyOptimizer(func=opt_func, maximize=False)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    optimizer.init(params=params, random_state=1)
    print('best score', optimizer.score)
    assert np.shape(optimizer.score) == ()


def test_init_deterministic_random_state():
    """Test if the initialized coordinates are deterministic if random state is fixed."""
    optimizer = GreedyOptimizer(func=opt_func, maximize=False)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    optimizer.init(params=params, random_state=1)
    coords0 = optimizer._coords_all
    optimizer.init(params=params, random_state=1)
    coords1 = optimizer._coords_all

    assert all(val0 == val1 for row0, row1 in zip(coords0, coords1) for val0, val1 in zip(row0, row1))


def test_update_monotonic_best_score_glob_minimize():
    """Test if the greedy optimizer monotonically converges for minimization problems."""
    optimizer = GreedyOptimizer(func=opt_func, maximize=False)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    optimizer.init(params=params, random_state=1)

    scores = [optimizer.score]
    for i in range(100):
        optimizer.update(params)
        scores.append(optimizer.score)

    assert all(scores[i+1] <= scores[i] for i in range(len(scores)-1))


def test_update_monotonic_best_scores_minimize():
    """Test if each particle of the particle swarm optimizer monotonically converges for minimization problems."""
    optimizer = GreedyOptimizer(func=opt_func, maximize=False)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    optimizer.init(params=params, random_state=1)

    scores = {p: [optimizer._score_all[p]] for p in range(20)}
    for i in range(100):
        optimizer.update(params)
        for particle in range(20):
            scores[particle] = scores[particle] + [optimizer._score_all[particle]]

    assert all(all(scores[particle][i+1] <= scores[particle][i] for i in range(len(scores[particle])-1))
               for particle in range(20))


def test_update_monotonic_best_score_glob_maximize():
    """Test if the greedy optimizer monotonically converges for maximization problems."""
    optimizer = GreedyOptimizer(func=opt_func, maximize=False)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    optimizer.init(params=params, random_state=1)

    scores = [optimizer.score]
    for i in range(100):
        optimizer.update(params)
        scores.append(optimizer.score)

    assert all(scores[i+1] >= scores[i] for i in range(len(scores)-1))


def test_coord_history_correct_dimension():
    """Test if the saved particles coordinate history has the correct dimensions."""
    optimizer = GreedyOptimizer(func=opt_func, maximize=False)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    optimizer.optimize(params, iterations=20)

    history = optimizer.coords_history

    assert len(history) == 21
    assert all(history[i].shape == (20, 2) for i in range(20))
