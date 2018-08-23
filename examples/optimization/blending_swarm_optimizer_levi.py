# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from mlopt.optimization import ParticleSwarmOptimizer
from examples.visualization.gif import create_gif


def levi_func(x, y):
    return np.sin(3*np.pi*x)**2+(x-1)**2*(1+np.sin(3*np.pi*y)**2)+(y-1)**2*(1+np.sin(2*np.pi*y)**2)


if __name__ == '__main__':

    linspace = np.linspace(0, 2, 80)

    bso = ParticleSwarmOptimizer(func=levi_func, maximize=False, particles=20)
    bso.optimize(params={'x': (0, 2), 'y': (0, 2)}, inertia=0.8, c_cog=2, c_soc=2,
                 learning_rate=0.01, random_state=None, iterations=300)
    lst_coords = bso.get_history()

    create_gif(obj_func=levi_func, linspace=linspace, coords=lst_coords, xlim=(0, 2), ylim=(0, 2))
