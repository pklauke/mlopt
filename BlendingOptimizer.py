import numpy as np

class BlendingOptimizer:
    
    def __init__(self, metric, maximize = True, power = 1):    
        self.metric = metric
        self.maximize = maximize
        self.power = power
        self.score = -1
        self.power = 1

    def is_better_score(self, score_to_test, score):
        cond = score_to_test > score
        return cond if self.maximize else not cond
        
    def fit(self, predictions, real, step = 0.1, init_weights = None, warm_start = False):
        if (init_weights != None) and warm_start:
            print('Warning: When warm_start is used init_weights are ignored.')
        
        assert(np.shape(predictions)[1] == len(real))
        
        le = len(predictions)
        
        if warm_start:
            weights = self.weights
        elif init_weights == None:
            weights = np.array([1.0] * le)
        else:
            assert(len(init_weights) == le)
            weights = init_weights

        score = 0
        best_score = score + self.maximize -0.5
        
        while self.is_better_score(best_score, score):
            best_score = self.metric(real, np.average(np.power(predictions, self.power), weights=weights, axis=0)**(1.0/self.power))
            score = best_score
            best_index, best_step = -1, 0.0
            for j in range(le):
                delta = np.array([(0 if k != j else step) for k in range(le)])
                s = self.metric(real, np.average(np.power(predictions, self.power), weights=weights+delta, axis=0)**(1.0/self.power))
                if self.is_better_score(s, best_score):
                    best_index, best_score, best_step = j, s, step
                    continue
                if weights[j] - step >= 0:
                    s = self.metric(real, np.average(np.power(predictions, self.power), weights=weights-delta, axis=0)**(1.0/self.power))
                    if(s > best_score):
                        best_index, best_score, best_step = j, s, -step
            if self.is_better_score(best_score, score):
                weights[best_index] += best_step
                
        self.weights = weights
        self.score = best_score
        return weights
    
    def predict(self, predictions):
        assert(np.shape(predictions)[0] == len(self.weights))
        return np.average(np.power(predictions, self.power), weights = self.weights, axis = 0)**(1.0/self.power)

