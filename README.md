# Ensemble

Python class for optimizing the weights in model ensemble blending. 

# Parameters


<b>metric</b> - Function for the metric that wants to be optimized. The function should have 2 array-like parameters. The first are the actual values, the second the predicted values.<br>
<b>maximize</b> - Boolean parameter that indicates whether the metric wants to be maximized or minimized. Default is True.<br>
<b>power</b> - Float parameter for the power of the predicted values before they are averaged. Default is 1. Higher values put more value on confident models.<br>

# Methods

## fit(predictions, real, step = 0.1, init_weights = None, warm_start = False)

Optimizes the weights for the blending of different models. Returns the weights.<br>

<b>predictions</b> - 2d array-like parameter for the model predictions.<br>
<b>real</b> - Array-like parameters for the labels.<br>
<b>step</b> - Float parameter for the size of a step used while optimizing. Default is 0.1. Higher values will speed up the process. Lower values may lead to overfitting<br>
<b>init_weights</b> - Array-like parameter for the initialization weights in the optimizing process. If not set 1.0 is used as initialization weight for each model. Blending is a non-convex optimization problem and therefore different weights may lead to a different score.<br>
<b>warm_start</b> - Boolean parameter that indicates if the currently set weights should be used as initialization weights.<br>

## predict(predictions)

Blends the weights using the optimized weights. Returns the blended predictions.<br>

<b>predictions</b> - 2d array-like parameter for the model predictions.<br>
