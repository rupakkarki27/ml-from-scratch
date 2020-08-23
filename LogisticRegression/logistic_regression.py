import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.1):
        self.weights = []
        self.lr = lr
        self.cost_history = []

    def sigmoid(self, X):
        return 1.0 / (1 + np.exp(-X))
    
    def _prediction(self, X, weights):
        pred = np.dot(X, weights)

        return self.sigmoid(pred)
    
    def cost_function(self, X, y, weights):
        N = len(y)

        pred = self._prediction(X, weights)

        cost = (-y.T * np.log(pred)) - ((1 - y).T * np.log(1-pred)) 

        cost = cost.sum()

        return (cost) / N

    def gradient_descent(self, X, y, weight):
        pred = self._prediction(X, weight)
        error = pred - y

        gradient = np.dot(X.T, error)
        gradient *= self.lr

        weight -= gradient

        return weight
    
    def fit(self, X, y, epochs=50):
        X = self._pre_process_data(X)
        weights = np.zeros((X.shape[1], 1))

        for i in range(epochs):
            weights = self.gradient_descent(X, y, weights)
            cost = self.cost_function(X, y, weights)
            self.cost_history.append(cost)

            #Log
            print("Iter {}/{}   Cost:{:.4f}".format(
                i+1, epochs, cost))
        
        self.weights = weights
    
    def _pre_process_data(self, X):
        # normalize the input variable
        fmean = X.mean()
        frange = np.amax(X) - np.amin(X)

        normalized = (X - fmean) / frange
        bias = np.ones((len(X), 1))
        X = np.append(bias, normalized, axis=1)

        return X


    def predict(self, X):
        X = self._pre_process_data(X)
        pred = self._prediction(X, self.weights)

        return 1 if pred >= 0.5 else 0
    