# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from mlopt.optimization import ParticleSwarmOptimizer
from examples.visualization.gif import create_gif


def ackley_func(x, y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.exp(1)+20


if __name__ == '__main__':

    linspace = np.linspace(-4.0, 4.0, 80)

    bso = ParticleSwarmOptimizer(func=ackley_func, maximize=False, particles=20)
    bso.optimize(params={'x': (-4.0, 4.0), 'y': (-4.0, 4.0)}, inertia=0.8, c_cog=2, c_soc=2,
                 learning_rate=0.01, random_state=None, iterations=300)
    lst_coords = bso.coords_history

    create_gif(obj_func=ackley_func, linspace=linspace, coords=lst_coords, xlim=(-4, 4), ylim=(-4, 4))
