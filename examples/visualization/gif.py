# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from examples import visualization


def create_gif(obj_func, linspace, coords, xlim, ylim, save_filename=None, color='blue', cmap='RdGy'):

    score_grid = visualization.calc_score_grid(obj_func=obj_func, linspace=linspace)

    fig = plt.figure()

    plt.contour(linspace, linspace, score_grid, 50, cmap=cmap)
    plt.colorbar()

    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plot = plt.scatter([], [], color=color, marker='.', zorder=10)

    def update(i):
        particles = []
        for p in range(len(coords[i])):
            particles.append([coords[i][p][0], coords[i][p][1]])
        plot.set_offsets(particles)

        return plot

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(coords)), interval=20, blit=False)
    if save_filename is not None:
        anim.save(save_filename, dpi=80, writer='imagemagick')
    else:
        plt.show()
