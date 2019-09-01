# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from mlopt.optimization import ParticleSwarmOptimizer
from examples.visualization.gif import create_gif


def obj_func(x, y):
    return (0.25*np.sin(3*np.pi * x) + np.sin(3*np.pi * y)) - np.abs(1-x) * np.abs(1-y)


if __name__ == '__main__':

    linspace = np.linspace(0.0, 2, 41)

    bso = ParticleSwarmOptimizer(func=obj_func, maximize=True, particles=20)
    bso.optimize(params={'x': (0, 2), 'y': (0, 2)}, inertia=0.8, c_cog=2, c_soc=2,
                 learning_rate=0.02, random_state=None, iterations=300)
    lst_coords = bso.coords_history

    create_gif(obj_func=obj_func, linspace=linspace, coords=lst_coords, xlim=(0, 2), ylim=(0, 2))
