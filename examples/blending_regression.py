# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import mean_absolute_error

from mlopt import BlendingTransformer

if __name__ == '__main__':

    labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    predictions_model_1 = [0.11, 0.19, 0.25, 0.37, 0.55, 0.62, 0.78, 0.81, 0.94]
    predictions_model_2 = [0.07, 0.21, 0.29, 0.33, 0.53, 0.54, 0.74, 0.74, 0.91]
    predictions_blended = [predictions_model_1, predictions_model_2]

    blender = BlendingTransformer(metric=mean_absolute_error, maximize=False)
    blender.fit(y=labels, X=predictions_blended)

    weights = blender.weights
    score = blender.score

    print('MAE 1: {:0.3f}'.format(mean_absolute_error(labels, predictions_model_1)))
    print('MAE 2: {:0.3f}'.format(mean_absolute_error(labels, predictions_model_2)))
    print('Optimized blending weights: ', weights)
    print('MAE blended: {:0.3f}'.format(score))
