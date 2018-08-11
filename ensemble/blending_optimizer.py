import numpy as np


class BlendingOptimizer:
    """Class for optimizing the weights in blending of different model predictions.

    :param metric: Callable metric function to optimize.
    :param maximize: Boolean indicating whether the `metric` needs to be maximized or minimized.
    :param power: Power to apply on each models' predictions before blending.

    :Example:
        >>> from sklearn.metrics import mean_absolute_error

        >>> y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        >>> p_model_1 = [0.11, 0.19, 0.25, 0.37, 0.55, 0.62, 0.78, 0.81, 0.94]
        >>> p_model_2 = [0.07, 0.21, 0.29, 0.33, 0.53, 0.54, 0.74, 0.74, 0.91]
        >>> p = [p_model_1, p_model_2]

        >>> opt = BlendingOptimizer(metric=mean_absolute_error, maximize=False)
        >>> weights = opt.fit(y=y, p=p)
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
        
    def fit(self, y, p, step=0.1, init_weights=None, warm_start: bool = False):
        """Fit the model on the given predictions.

        :param y: Labels.
        :param p: Predictions of different models for the labels.
        :param step: Step size for optimizing the weights. Smaller step sizes most likely improve resulting score but
                     increase training time.
        :param init_weights: Initial weights for training.
        :param warm_start: Continues training. Will only work when `fit` has been called with this object earlier.
                           Ignores `init_weights``
        :return: Optimized weights.
        """
        if (init_weights is not None) and warm_start:
            print('Warning: When warm_start is used init_weights are ignored.')
        
        assert np.shape(p)[1] == len(y), (
               'Length of predictions and labels does not match: {} != {}'.format(np.shape(p)[1], len(y)))

        def __is_better_score(score_to_test, score):
            return score_to_test > score if self.maximize else not score_to_test > score
        
        if warm_start:
            assert self._weights is not None, 'Optimizer has to be fitted before `warm_start` can be used.'
            weights = self._weights
        elif init_weights is None:
            weights = np.array([1.0] * len(p))
        else:
            assert(len(init_weights) == np.shape(p)[0]), (
                'Length of predictions and weights does not match: {} != {}'.format(np.shape(p)[0], len(init_weights)))
            weights = init_weights

        score = 0
        best_score = self.maximize - 0.5
        
        while __is_better_score(best_score, score):
            best_score = self.metric(y, np.average(np.power(p, self._power), weights=weights, axis=0) ** (
                        1.0 / self._power))
            score = best_score
            best_index, best_step = -1, 0.0
            for j in range(len(p)):
                delta = np.array([(0 if k != j else step) for k in range(len(p))])
                s = self.metric(y, np.average(np.power(p, self._power), weights=weights + delta, axis=0) ** (
                            1.0 / self._power))
                if __is_better_score(s, best_score):
                    best_index, best_score, best_step = j, s, step
                    continue
                if weights[j] - step >= 0:
                    s = self.metric(y, np.average(np.power(p, self._power), weights=weights - delta, axis=0) ** (
                                1.0 / self._power))
                    if s > best_score:
                        best_index, best_score, best_step = j, s, -step
            if __is_better_score(best_score, score):
                weights[best_index] += best_step
                
        self._weights = weights
        self._score = best_score

        return weights
    
    def predict(self, p):
        """Predict blended predictions using the trained weights.

        :param p: Predictions.
        :return: Blended predictions.
        """
        assert np.shape(p)[0] == len(self._weights), (
               'Length of predictions and weights does not match: {} != {}'.format(np.shape(p)[0], len(self._weights)))
        return np.average(np.power(p, self._power), weights=self._weights, axis=0) ** (1.0 / self._power)
