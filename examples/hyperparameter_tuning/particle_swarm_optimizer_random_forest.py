# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides an example how hyperparameter tuning for a RandomForestClassifier on iris flower data set can be
done using mlopt.optimize.ParticleSwarmOptimizer."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score

from mlopt.optimization import ParticleSwarmOptimizer


if __name__ == '__main__':
    iris = load_iris()

    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df = df.loc[(df.species == 'virginica') | (df.species == 'versicolor'), :]
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

    X_train, X_test = df[df['is_train'] == True], df[df['is_train'] == False]
    features = df.columns[:4]
    y_train, y_test = pd.factorize(X_train['species'])[0], pd.factorize(X_test['species'])[0]

    def get_score(max_depth, min_samples_leaf):
        clf = RandomForestClassifier(random_state=1, max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf))
        clf.fit(X_train[features], y_train)
        preds_test = clf.predict_proba(X_test[features])[:, 1]
        score = roc_auc_score(y_test, preds_test)
        print('AUC: {:0.4f}, max depth: {:0.0f}, min_samples_leaf: {:0.0f}'.format(score, max_depth, min_samples_leaf))
        return score

    bso = ParticleSwarmOptimizer(func=get_score, maximize=True, particles=10)
    params = {'max_depth': (1, 20), 'min_samples_leaf': (1, 20)}
    bso.optimize(params=params, random_state=1, iterations=10)

    print('Best AUC: {:0.4f}, max_depth: {}, min_samples_leaf: {}'.format(bso.best_score_glob,
                                                                          int(bso.best_coords_glob[0]),
                                                                          int(bso.best_coords_glob[1])))
