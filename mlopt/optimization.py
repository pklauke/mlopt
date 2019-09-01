# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains classes for solving optimization problems. Currently implemented classes are:

ParticleSwarmOptimizer: Uses particle swarm optimization to maximize or minimize a given function.
"""

from typing import Callable, List, Dict, Tuple, Union

import numpy as np


class ParticleSwarmOptimizer:
    """Optimizer to minimize or maximize an objective function using Particle Swarm Optimization. The whole
    optimization can either be executed using the method `optimize` at once or iteratively using the `init` and
    `update` methods.

    :param func: Callable function to optimize.
    :param maximize: Boolean indicating whether `func` wants to be maximized or minimized.
    :param particles: Number of particles to use.
    """

    def __init__(self, func: Callable, maximize: bool, particles: int = 20):

        self.func = func
        self.maximize = maximize
        self.particles = particles
        self._velocities = None
        self._coords_all = None
        self._best_coords_all = None
        self.coords = None
        self._score_all = None
        self.score = None
        self._best_particle_idx = None
        self.coords_history = None

    def _calc_scores(self) -> List[float]:
        """Calculate the value of `func` for each particle."""
        return [self.func(*self._coords_all[i, :]) for i in range(np.shape(self._coords_all)[0])]

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
        self._velocities = (np.random.random((self.particles, len(params))) - 0.5) * 2 * lst_vel_norm
        self._coords_all = np.random.random((self.particles, len(params)))
        for i in range(self.particles):
            for j, value in enumerate(params.values()):
                self._coords_all[i, j] = self._coords_all[i, j] * (value[1] - value[0]) + value[0]
        self._best_coords_all = self._coords_all

        self.coords_history = []
        self.coords_history.append(self._coords_all)

        scores = self._calc_scores()
        self._score_all = scores
        self._best_particle_idx = np.argmax(self._score_all) if self.maximize else np.argmin(self._score_all)
        self.score = self._score_all[self._best_particle_idx]
        self.coords = self._best_coords_all[self._best_particle_idx].copy()

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
        self._velocities = self.__next_velocity(inertia, self._velocities, self._coords_all,
                                                self._best_coords_all, self.coords, c_cog, c_soc)
        self._coords_all += self._velocities * learning_rate
        lst_clip_range = [tup for tup in params.values()]
        for dim_idx in range(len(params)):
            self._coords_all[:, dim_idx] = self._coords_all[:, dim_idx].clip(lst_clip_range[dim_idx][0],
                                                                             lst_clip_range[dim_idx][1])

        def __better_score(x, y, maximize: bool):
            return x > y if maximize else x < y

        lst_scores = self._calc_scores()
        if self.maximize:
            tmp = np.argmax([self._score_all, lst_scores], axis=0)
            self._best_coords_all = np.array([bc if s == 0 else c for bc, c, s in zip(self._best_coords_all,
                                                                                      self._coords_all, tmp)])
            self._score_all = np.max([self._score_all, lst_scores], axis=0)
            self._best_particle_idx = np.argmax(self._score_all)
        else:
            tmp = np.argmin([self._score_all, lst_scores], axis=0)
            self._best_coords_all = np.array([bc if s == 0 else c for bc, c, s in zip(self._best_coords_all,
                                                                                      self._coords_all, tmp)])
            self._score_all = np.min([self._score_all, lst_scores], axis=0)
            self._best_particle_idx = np.argmin(self._score_all)

        if __better_score(self._score_all[self._best_particle_idx], self.score, maximize=self.maximize):
            self.score = self._score_all[self._best_particle_idx]
            self.coords = self._coords_all[self._best_particle_idx].copy()
        self.coords_history.append(self._coords_all.copy())

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

        for _ in range(iterations):
            self.update(params=params, inertia=inertia, c_cog=c_cog, c_soc=c_soc, learning_rate=learning_rate)

        return self


class GreedyOptimizer:
    """Optimizer to minimize or maximize an objective function using a greedy approach. The whole
    optimization can either be executed using the method `optimize` at once or iteratively using the `init` and
    `update` methods.

    :param func: Callable function to optimize.
    :param maximize: Boolean indicating whether `func` wants to be maximized or minimized.
    """

    def __init__(self, func: Callable, maximize: bool):

        self.func = func
        self.maximize = maximize
        self.coords = None
        self.score = None
        self.coords_history = None
        self._lower_bounds = None
        self._upper_bounds = None

    def init(self, params: Dict[str, Tuple[float, float]], random_state: Union[int, None] = None):
        """Initialize the coordinates of the GreedyOptimizer.

        :param params: Dictionary containing the variable names and their value ranges. Key is expected to be the
                       variable name and value a tuple containing the minimum and maximum value of the variable.
        :param random_state: Random state for initializing.
        :return: None
        """
        self._lower_bounds = [lower for lower, _ in params.values()]
        self._upper_bounds = [upper for _, upper in params.values()]
        self.coords = np.array([(upper+lower) / 2 for lower, upper in zip(self._lower_bounds, self._upper_bounds)])

        self.score = self.func(*self.coords)

        self.coords_history = []
        self.coords_history.append(self.coords)

    def update(self, params: Dict[str, Tuple[float, float]], step_size: float = 0.1):
        """Update all coordinates for one step.

        :param params: Dictionary containing the variable names and their value ranges. Key is expected to be the
                       variable name and value a tuple containing the minimum and maximum value of the variable.
        :param step_size: Size of the step the coordinates will change.
        :return: None
        """
        def __better_score(x, y, maximize: bool):
            return x > y if maximize else x < y

        score = 0
        best_score = self.maximize - 0.5

        while __better_score(best_score, score, self.maximize):
            best_score = self.func(*self.coords)

            score = best_score
            best_index, best_step = -1, 0.0
            for j in range(len(params)):
                delta = np.array([(0 if k != j else step_size) for k in range(len(params))])

                if self.coords[j] + step_size <= self._upper_bounds[j]:
                    curr_score = self.func(*(self.coords+delta))
                    if __better_score(curr_score, best_score, self.maximize):
                        best_index, best_score, best_step = j, curr_score, step_size
                        continue

                if self.coords[j] - step_size >= self._lower_bounds[j]:
                    curr_score = self.func(*(self.coords+delta))
                    if curr_score > best_score:
                        best_index, best_score, best_step = j, curr_score, -step_size
            if __better_score(best_score, score, self.maximize):
                self.coords[best_index] += best_step
                self.score = best_score

        self.coords_history.append(self.coords.copy())

    def optimize(self, params: Dict[str, Tuple[float, float]], step_size=0.1, iterations: int = 100,
                 random_state: Union[int, None] = None):
        """Optimize the given function `func` using the methods `init` and `update`.

        :param params: Dictionary containing the variable names and their value ranges. Key is expected to be the
                       variable name and value a tuple containing the minimum and maximum value of the variable.
        :param step_size: Rate at which the position of the particles gets updated in respect to their velocity.
        :param iterations: Number of iterations.
        :param random_state: Random state for initializing.
        :return:
        """
        self.init(params=params, random_state=random_state)

        for _ in range(iterations):
            self.update(params=params, step_size=step_size)

        return self
