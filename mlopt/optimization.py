# !/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, List, Dict, Tuple, Union

import numpy as np


class ParticleSwarmOptimizer(object):

    def __init__(self, func: Callable, maximize: bool, particles: int = 20):
        """Optimizer to minimize or maximize an objective function using Particle Swarm Optimization. The whole
        optimization can either be executed using the method `optimize` at once or iteratively using the `init` and
        `update` methods.

        :param func: Callable function to optimize.
        :param maximize: Boolean indicating whether `func` wants to be maximized or minimized.
        :param particles: Number of particles to use.
        """
        self.func = func
        self.maximize = maximize
        self.particles = particles
        self.velocities = None
        self.coords = None
        self.best_coords = None
        self.best_coords_glob = None
        self.best_scores = None
        self.best_score_glob = None
        self._best_particle_idx = None
        self._coords_history = None

    def _calc_scores(self) -> List[float]:
        """Calculate the value of `func` for each particle."""
        return [self.func(*self.coords[i, :]) for i in range(np.shape(self.coords)[0])]

    @staticmethod
    def __next_velocity(inertia: float, velocities, coords, best_coords, best_coords_glob, c_cog: float, c_soc: float):
        return inertia * velocities \
               + c_cog * np.random.random(coords.shape) * (best_coords - coords) \
               + c_soc * np.random.random(coords.shape) * (best_coords_glob - coords)

    def init(self, params: Dict[str, Tuple[float, float]], random_state: Union[int, None] = None):
        """Initialize the members of the ParticleSwarmOptimizer.

        :param params: Dictionary containing the variable names and their value ranges. Key is expected to be the
                       variable name and value a tuple containing the minimum and maximum value of the variable.
        :param random_state: Random state for initializing.
        :return: None
        """
        np.random.seed(random_state)

        lst_vel_norm = [[end - begin for begin, end in params.values()] for _ in range(self.particles)]
        self.velocities = (np.random.random((self.particles, len(params))) - 0.5) * 2 * lst_vel_norm
        self.coords = np.random.random((self.particles, len(params)))
        for i in range(self.particles):
            for j, (key, value) in enumerate(params.items()):
                self.coords[i, j] = self.coords[i, j] * (value[1] - value[0]) + value[0]
        self.best_coords = self.coords

        self._coords_history = []
        self._coords_history.append(self.coords)

        scores = self._calc_scores()
        self.best_scores = scores
        self._best_particle_idx = np.argmax(self.best_scores) if self.maximize else np.argmin(self.best_scores)
        self.best_score_glob = self.best_scores[self._best_particle_idx]
        self.best_coords_glob = self.best_coords[self._best_particle_idx].copy()

        self._coords_history.append(self.coords)

    def update(self, params: Dict[str, Tuple[float, float]], inertia: float = 0.5, c_cog: float = 2.0,
               c_soc: float = 2.0, learning_rate: float = 0.1):
        """Synchronously updates all particles velocities and positions once.

        The velocity v and position x of particle i are updated using
            v(i, t+1) = v(i, t) * inertia + c_cog * rand(0, 1) * (x_best(i)-x(i)) + c_soc * rand(0, 1) * (x_best-x(i))
            x(i, t+1) = x(i, t) * v(i, t+1) * learning_rate
        where x_best(i) is the best position in particle i's history and x_best is the best position in all particles'
        history.

        :param params: Dictionary containing the variable names and their value ranges. Key is expected to be the
                       variable name and value a tuple containing the minimum and maximum value of the variable.
        :param inertia: Inertia of a particle. Higher values result in smaller velocity changes.
        :param c_cog: Cognitive scaling factor.
        :param c_soc: Social scaling factor.
        :param learning_rate: Rate at which the position of the particles gets updated in respect to their velocity.
        :return: None
        """
        self.velocities = self.__next_velocity(inertia, self.velocities, self.coords, self.best_coords,
                                               self.best_coords_glob, c_cog, c_soc)
        self.coords += self.velocities * learning_rate
        lst_clip_range = [tup for tup in params.values()]
        for dim_idx in range(len(params)):
            self.coords[:, dim_idx] = self.coords[:, dim_idx].clip(lst_clip_range[dim_idx][0],
                                                                   lst_clip_range[dim_idx][1])

        def __better_score(x, y, maximize: bool):
            return x > y if maximize else x < y

        lst_scores = self._calc_scores()
        if self.maximize:
            tmp = np.argmax([self.best_scores, lst_scores], axis=0)
            self.best_coords = np.array([bc if s == 0 else c for bc, c, s in zip(self.best_coords, self.coords, tmp)])
            self.best_scores = np.max([self.best_scores, lst_scores], axis=0)
            self._best_particle_idx = np.argmax(self.best_scores)
        else:
            tmp = np.argmin([self.best_scores, lst_scores], axis=0)
            self.best_coords = np.array([bc if s == 0 else c for bc, c, s in zip(self.best_coords, self.coords, tmp)])
            self.best_scores = np.min([self.best_scores, lst_scores], axis=0)
            self._best_particle_idx = np.argmin(self.best_scores)

        if __better_score(self.best_scores[self._best_particle_idx], self.best_score_glob, maximize=self.maximize):
            self.best_score_glob = self.best_scores[self._best_particle_idx]
            self.best_coords_glob = self.coords[self._best_particle_idx].copy()
        self._coords_history.append(self.coords.copy())

    def optimize(self, params: Dict[str, Tuple[float, float]], inertia: float = 0.8, c_cog: float = 2.0,
                 c_soc: float = 2.0, learning_rate: float = 0.1, iterations: int = 100,
                 random_state: Union[int, None] = None):
        """Optimize the given function `func` using the methods `init` and `update`.

        :param params: Dictionary containing the variable names and their value ranges. Key is expected to be the
                       variable name and value a tuple containing the minimum and maximum value of the variable.
        :param inertia: Inertia of a particle. Higher values result in smaller velocity changes. Good values are in the
                        range (0.4, 0.9).
        :param c_cog: Cognitive scaling factor.
        :param c_soc: Social scaling factor.
        :param learning_rate: Rate at which the position of the particles gets updated in respect to their velocity.
        :param iterations: Number of iterations.
        :param random_state: Random state for initializing.
        :return:
        """
        self.init(params=params, random_state=random_state)

        for i in range(iterations):
            self.update(params=params, inertia=inertia, c_cog=c_cog, c_soc=c_soc, learning_rate=learning_rate)

        return self

    def get_history(self):
        """Return the position history of each particle."""
        assert self._coords_history is not None, 'Coordinate history is not saved. Call `optimize` first.'
        return self._coords_history
