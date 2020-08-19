# implementing the linear regression
# algorithm from scratch in python
import random

class LinearRegression1V():
    def __init__(self, lr=0.1):
        self.w = 0
        self.b = 0
        self.lr = lr
    
    def prediction(self, X, weight, bias):
        return weight * X + bias
    
    def gradient_descent(self, X, y):
        # w_d = 0
        # b_d = 0
        N = len(X)
        for i in range(N):
            self.w -= (self.lr * 2 * (self.prediction(X[i], self.w, self.b) - y[i]) * X[i]) / N
            self.b -= (self.lr * 2 * (self.prediction(X[i], self.w, self.b) - y[i])) / N
        
        return self.w, self.b

    def cost_function(self, X, y, weight, bias):
        N = len(X)
        cost = 0.0
        for i in range(N):
            cost += (y[i] - self.prediction(X[i], weight, bias))**2
        
        return cost / N

    def fit(self, X, y, epochs=5):
        cost_hist = []
        for i in range(epochs):
            weight, bias = self.gradient_descent(X, y)
            cost = self.cost_function(X, y, weight, bias)
            cost_hist.append(cost)

            # Log loop
            print("Iter {0}/{1}  weight={2:.3f}   bias={3:.4f}  cost={4:.4f}".format(
                i+1, epochs, weight, bias, cost
            ))
        
        return weight, bias
    
    def predict(self, X):
        pred = []
        for i in range(len(X)):
            pred.append(round((self.w * X[i]) + self.b, 2))

        return pred
    
    def score(self, y, y_pred):
        N = len(y)
        diff = []
        for i in range(N):
            diff.append(abs(y[i] - y_pred[i]))
        
        print("Accuracy: {:.2f}".format(1.0 - sum(diff) / N))
