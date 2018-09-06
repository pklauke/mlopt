# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from mlopt.optimization import ParticleSwarmOptimizer


def opt_func(x, y):
    return x**2 + y**2


def opt_func_inv(x, y):
    return - opt_func(x, y)


def test_init_correct_dimensions_velocities():
    """Test if the initialized velocities have the correct dimension."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)

    assert pso.velocities.shape == (20, 2)


def test_init_correct_dimensions_coords():
    """Test if the initialized coordinates have the correct dimension."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)

    assert pso.coords.shape == (20, 2)


def test_init_correct_dimensions_best_coords():
    """Test if the initialized best coordinates of each particle have the correct dimensions."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)

    assert pso.best_coords.shape == (20, 2)


def test_init_correct_dimensions_best_coords_glob():
    """Test if the initialized best coordinates of all particles combined have the correct dimensions."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)

    assert pso.best_coords_glob.shape == (2,)


def test_init_correct_dimensions_best_scores():
    """Test if the initialized best scores of each particle have the correct dimensions."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)

    assert len(pso.best_scores) == 20


def test_init_correct_dimensions_best_score_glob():
    """Test if the initialized best score of all particles have the correct dimension."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)

    assert np.shape(pso.best_score_glob) == ()


def test_init_deterministic_random_state():
    """Test if the initialized coordinates are deterministic if random state is fixed."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)
    coords0 = pso.coords
    pso.init(params=params, random_state=1)
    coords1 = pso.coords

    assert all(val0 == val1 for row0, row1 in zip(coords0, coords1) for val0, val1 in zip(row0, row1))


def test_init_different_random_state():
    """Test if the initialized coordinates are not deterministic if random state is not fixed."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)
    coords0 = pso.coords
    pso.init(params=params, random_state=2)
    coords1 = pso.coords

    assert any(val0 != val1 for row0, row1 in zip(coords0, coords1) for val0, val1 in zip(row0, row1))


def test_update_monotonic_best_score_glob_minimize():
    """Test if the particle swarm optimizer monotonically converges for minimization problems."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)

    scores = [pso.best_score_glob]
    for i in range(100):
        pso.update(params)
        scores.append(pso.best_score_glob)

    assert all(scores[i+1] <= scores[i] for i in range(len(scores)-1))


def test_update_monotonic_best_scores_minimize():
    """Test if each particle of the particle swarm optimizer monotonically converges for minimization problems."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)

    scores = {p: [pso.best_scores[p]] for p in range(20)}
    for i in range(100):
        pso.update(params)
        for p in range(20):
            scores[p] = scores[p] + [pso.best_scores[p]]

    assert all(all(scores[p][i+1] <= scores[p][i] for i in range(len(scores[p])-1)) for p in range(20))


def test_update_monotonic_best_score_glob_maximize():
    """Test if the particle swarm optimizer monotonically converges for maximization problems."""
    pso = ParticleSwarmOptimizer(func=opt_func_inv, maximize=True, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)

    scores = [pso.best_score_glob]
    for i in range(100):
        pso.update(params)
        scores.append(pso.best_score_glob)

    assert all(scores[i+1] >= scores[i] for i in range(len(scores)-1))


def test_update_monotonic_best_scores_maximize():
    """Test if each particle of the particle swarm optimizer monotonically converges for maximization problems."""
    pso = ParticleSwarmOptimizer(func=opt_func_inv, maximize=True, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.init(params=params, random_state=1)

    scores = {p: [pso.best_scores[p]] for p in range(20)}
    for i in range(100):
        pso.update(params)
        for p in range(20):
            scores[p] = scores[p] + [pso.best_scores[p]]

    assert all(all(scores[p][i+1] >= scores[p][i] for i in range(len(scores[p])-1)) for p in range(20))


def test_coord_history_correct_dimension():
    """Test if the saved particles coordinate history has the correct dimensions."""
    pso = ParticleSwarmOptimizer(func=opt_func, maximize=False, particles=20)

    params = {'x': (-1, 1), 'y': (-1, 1)}
    pso.optimize(params, iterations=20)

    history = pso.get_history()

    assert len(history) == 21
    assert all(history[i].shape == (20, 2) for i in range(20))
