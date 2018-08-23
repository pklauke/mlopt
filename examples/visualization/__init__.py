# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def calc_score_grid(obj_func, linspace):
    l = []
    for x in linspace:
        l_ = []
        for y in linspace:
            l_.append(obj_func(x, y))
        l.append(l_)

    return np.array(l)
