import numpy as np

import mlopt.optimization


class BlendingSwarmTransformer(mlopt.optimization.ParticleSwarmOptimizer, mlopt.TransformerMixin):
    def __init__(self, metric, maximize, particles=20):
        """Optimizer to minimize or maximize an objective metric using Particle Swarm Optimization.

        :param metric: Callable function to optimize.
        :param maximize: Boolean indicating whether `metric` wants to be maximized or minimized.
        :param particles: Number of particles to use.
        """
        super().__init__(func=lambda X, y, weights: metric(y, self.__weighted_average(X, weights)),
                         maximize=maximize, particles=particles)
        self.X = None
        self.y = None

    def _calc_scores(self):
        """Calculate the score of the objective metric for each particle."""
        scores = [self.func(X=self.X, y=self.y, weights=coord) for coord in self.coords]
        if self.best_score_glob is not None:
            print('id: {}, AUC: {:0.7f}, best AUC: {:0.7f}'.format(self._best_particle_idx, np.max(scores),
                                                                   self.best_score_glob))
        return scores

    @staticmethod
    def __weighted_average(X, weights):
        return np.average(X, axis=0, weights=weights)

    def fit(self, X, y, inertia=0.5, c_cog=2.0, c_soc=2.0, learning_rate=0.1, iterations=100, random_state=None):
        """Fit the model on the given predictions.

        :param X: Predictions of different models for the labels.
        :param y: Labels.
        :param inertia: Inertia of a particle. Higher values result in smaller velocity changes. Good values are in the
                        range (0.4, 0.9).
        :param c_cog: Cognitive scaling factor.
        :param c_soc: Social scaling factor.
        :param learning_rate: Rate at which the position of the particles gets updated in respect to their velocity.
        :param iterations: Number of iterations.
        :param random_state: Random state for initializing.
        :return: Optimized weights."""
        self.X = X
        self.y = y
        params = {'x' + str(i): (0, 2) for i in range(np.shape(X)[0])}
        self.optimize(params=params, inertia=inertia, c_cog=c_cog, c_soc=c_soc,
                      learning_rate=learning_rate, iterations=iterations, random_state=random_state)

    def transform(self, X):
        """Transform blended predictions using the trained weights.

        :param X: Predictions.
        :return: Blended predictions.
        """
        return self.__weighted_average(X=X, weights=self.best_coords_glob)

    def fit_transform(self, X, y, **kwargs):
        """Fit transformer to X, then transform X. See `fit` and `transform` for further explanation."""
        return self.fit(X=X, y=y, **kwargs).transform(X=X)


class BlendingGreedyTransformer(mlopt.TransformerMixin):
    """Class for optimizing the weights in blending of different model predictions.

    :param metric: Callable metric function to optimize.
    :param maximize: Boolean indicating whether the `metric` needs to be maximized or minimized.
    :param power: Power to apply on each models' predictions before blending.

    Example:
        >>> from sklearn.metrics import mean_absolute_error

        >>> y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        >>> p_model_1 = [0.11, 0.19, 0.25, 0.37, 0.55, 0.62, 0.78, 0.81, 0.94]
        >>> p_model_2 = [0.07, 0.21, 0.29, 0.33, 0.53, 0.54, 0.74, 0.74, 0.91]
        >>> p = [p_model_1, p_model_2]

        >>> opt = BlendingTransformer(metric=mean_absolute_error, maximize=False)
        >>> weights = opt.optimize(y=y, X=p)
        >>> print('MAE 1: {:0.3f}'.format(mean_absolute_error(y, p_model_1)))
        >>> print('MAE 2: {:0.3f}'.format(mean_absolute_error(y, p_model_2)))
        >>> print('Optimized blending weights: ', weights)
        >>> print('MAE blended: {:0.3f}'.format(opt._score))
    """
    def __init__(self, metric, maximize=True, power=1):
        self.metric = metric
        self.maximize = maximize
        self._power = power
        self._score = None
        self._weights = None
        
    def fit(self, X, y, step_size=0.1, init_weights=None, warm_start: bool = False):
        """Fit the model on the given predictions.

        :param X: Predictions of different models for the labels.
        :param y: Labels.
        :param step_size: Step size for optimizing the weights. Smaller step sizes most likely improve resulting score but
                     increase training time.
        :param init_weights: Initial weights for training.
        :param warm_start: Continues training. Will only work when `fit` has been called with this object earlier.
                           Ignores `init_weights``
        :return: Optimized weights.
        """
        assert len(np.shape(X)) == 2, 'X must be 2-dimensional, got {}-D instead.'.format(len(np.shape(X)))
        assert np.shape(X)[0] > 1, 'X must contain predictions from at least two models. ' \
                                   'Got {} instead'.format(np.shape(X)[0])

        assert np.shape(X)[1] == len(y), (
            'BlendingOptimizer: Length of predictions and labels does not match: '
            'preds_len={}, y_len={}'.format(np.shape(X)[1], len(y)))

        if (init_weights is not None) and warm_start:
            print('Warning: When warm_start is used init_weights are ignored.')

        def __is_better_score(score_to_test, score):
            return score_to_test > score if self.maximize else not score_to_test > score
        
        if warm_start:
            assert self._weights is not None, 'Optimizer has to be fitted before `warm_start` can be used.'
            weights = self._weights
        elif init_weights is None:
            weights = np.array([1.0] * len(X))
        else:
            assert (len(init_weights) == np.shape(X)[0]), (
                'BlendingOptimizer: Number of models to blend its predictions and weights does not match: '
                'n_models={}, weights_len={}'.format(np.shape(X)[0], len(init_weights)))
            weights = init_weights

        score = 0
        best_score = self.maximize - 0.5
        
        while __is_better_score(best_score, score):
            best_score = self.metric(y, np.average(np.power(X, self._power), weights=weights, axis=0) ** (
                        1.0 / self._power))
            score = best_score
            best_index, best_step = -1, 0.0
            for j in range(len(X)):
                delta = np.array([(0 if k != j else step_size) for k in range(len(X))])
                s = self.metric(y, np.average(np.power(X, self._power), weights=weights + delta, axis=0) ** (
                            1.0 / self._power))
                if __is_better_score(s, best_score):
                    best_index, best_score, best_step = j, s, step_size
                    continue
                if weights[j] - step_size >= 0:
                    s = self.metric(y, np.average(np.power(X, self._power), weights=weights - delta, axis=0) ** (
                                1.0 / self._power))
                    if s > best_score:
                        best_index, best_score, best_step = j, s, -step_size
            if __is_better_score(best_score, score):
                weights[best_index] += best_step
                
        self._weights = weights
        self._score = best_score

        return weights
    
    def transform(self, X):
        """Transform blended predictions using the trained weights.

        :param X: Predictions.
        :return: Blended predictions.
        """
        assert np.shape(X)[0] == len(self._weights), (
               'Length of predictions and weights does not match: {} != {}'.format(np.shape(X)[0], len(self._weights)))
        return np.average(np.power(X, self._power), weights=self._weights, axis=0) ** (1.0 / self._power)

    def fit_transform(self, X, y, step_size=0.1, init_weights=None, warm_start=False):
        """Fit transformer to X, then transform X. See `fit` and `transform` for further explanation."""
        self.fit(X=X, y=y, step_size=step_size, init_weights=init_weights, warm_start=warm_start)

        return self.transform(X=X)
