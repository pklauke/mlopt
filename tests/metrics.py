# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def mean_absolute_error(y_true, y_pred, sample_weight=None):
    """Mean absolute error regression loss

    :param y_true: Labels
    :param y_pred: Predictions
    :param sample_weight: Weight to calculate each sample with before taking mean
    """
    return np.average(np.abs(y_pred - y_true), weights=sample_weight, axis=0)


def gini(y_true, y_pred):
    """Gini coefficient

    :param y_true: Labels
    :param y_pred: Predictions
    """

    assert (len(y_true) == len(y_pred))
    all_ = np.asarray(np.c_[y_true, y_pred, np.arange(len(y_true))], dtype=np.float)
    all_ = all_[np.lexsort((all_[:, 2], -1 * all_[:, 1]))]
    total_losses = all_[:, 0].sum()
    gini_sum = all_[:, 0].cumsum().sum() / total_losses

    gini_sum -= (len(y_true) + 1) / 2
    return gini_sum / len(y_true)


def gini_normalized(y_true, y_pred):
    """Normalized gini coefficient

    :param y_true: Labels
    :param y_pred: Predictions
    """
    return gini(y_true, y_pred) / gini(y_true, y_true)


def roc_auc_score(y_true, y_pred):
    """Receiver operating area under curve score

    :param y_true: Labels
    :param y_pred: Predictions
    """
    return gini_normalized(y_true, y_pred) / 2 + 0.5
