# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains unit tests for testing the mlopt.blending.BlendingSwarmTransformer class."""

from sklearn.metrics import mean_absolute_error, roc_auc_score

from mlopt.blending import BlendingSwarmTransformer


def test_blended_predictions_correct_dimensions():
    """Test if the transformer returns predictions with the correct dimension."""
    target = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    predictions_0 = [0.01, 0.09, 0.17, 0.31, 0.39, 0.54, 0.59, 0.69, 0.77, 0.92, 0.99]
    predictions_1 = [0.00, 0.12, 0.19, 0.32, 0.42, 0.51, 0.61, 0.68, 0.81, 0.91, 1.02]

    bst = BlendingSwarmTransformer(metric=mean_absolute_error, maximize=False)
    p_blended = bst.fit_transform([predictions_0, predictions_1], target)

    assert p_blended.shape == (len(target), )


def test_fixed_random_state_deterministic_minimize():
    """Test if the transformer returns deterministic results with fixed random state for minimization problems."""
    target = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    predictions_0 = [0.01, 0.09, 0.17, 0.31, 0.39, 0.54, 0.59, 0.69, 0.77, 0.92, 0.99]
    predictions_1 = [0.00, 0.12, 0.19, 0.32, 0.42, 0.51, 0.61, 0.68, 0.81, 0.91, 1.02]

    bst = BlendingSwarmTransformer(metric=mean_absolute_error, maximize=False)
    p_blended_0 = bst.fit_transform([predictions_0, predictions_1], target, random_state=1)
    p_blended_1 = bst.fit_transform([predictions_0, predictions_1], target, random_state=1)

    assert all(v0 == v1 for v0, v1 in zip(p_blended_0, p_blended_1))


def test_blended_results_minimize():
    """Test if the blended results of the transformer are at least as good as each single predictions for minimization
    problems."""
    target = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    predictions_0 = [0.01, 0.09, 0.17, 0.31, 0.39, 0.54, 0.59, 0.69, 0.77, 0.92, 0.99]
    predictions_1 = [0.00, 0.12, 0.19, 0.32, 0.42, 0.51, 0.61, 0.68, 0.81, 0.91, 1.02]

    bst = BlendingSwarmTransformer(metric=mean_absolute_error, maximize=False)
    p_blended = bst.fit_transform([predictions_0, predictions_1], target, random_state=1)

    mse_0 = mean_absolute_error(target, predictions_0)
    mse_1 = mean_absolute_error(target, predictions_1)
    mse_blended = mean_absolute_error(target, p_blended)

    assert mse_blended <= mse_0 and mse_blended <= mse_1


def test_fixed_random_state_deterministic_maximize():
    """Test if the transformer returns deterministic results with fixed random state for maximization problems."""
    target = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]

    predictions_0 = [0.01, 0.09, 0.17, 0.31, 0.39, 0.54, 0.59, 0.69, 0.77, 0.92, 0.99]
    predictions_1 = [0.00, 0.12, 0.19, 0.32, 0.42, 0.51, 0.61, 0.68, 0.81, 0.91, 1.02]

    bst = BlendingSwarmTransformer(metric=roc_auc_score, maximize=True)
    p_blended_0 = bst.fit_transform([predictions_0, predictions_1], target, random_state=1)
    p_blended_1 = bst.fit_transform([predictions_0, predictions_1], target, random_state=1)

    assert all(v0 == v1 for v0, v1 in zip(p_blended_0, p_blended_1))


def test_blended_results_maximize():
    """Test if the blended results of the transformer are at least as good as each single predictions for maximization
    problems."""
    target = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]

    predictions_0 = [0.01, 0.09, 0.17, 0.31, 0.39, 0.54, 0.59, 0.69, 0.77, 0.92, 0.99]
    predictions_1 = [0.00, 0.12, 0.19, 0.32, 0.42, 0.51, 0.61, 0.68, 0.81, 0.91, 1.02]

    bst = BlendingSwarmTransformer(metric=roc_auc_score, maximize=True)
    p_blended = bst.fit_transform([predictions_0, predictions_1], target, random_state=1)

    auc_0 = roc_auc_score(target, predictions_0)
    auc_1 = roc_auc_score(target, predictions_1)
    auc_blended = roc_auc_score(target, p_blended)

    assert auc_blended >= auc_0 and auc_blended >= auc_1
