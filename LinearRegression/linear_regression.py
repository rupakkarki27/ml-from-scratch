# implementing the linear regression
# algorithm from scratch in python
import random
import numpy as np

class LinearRegression1V():
    """
    This class is for Single Variable Linear Regression
    """
    def __init__(self, lr=0.1):
        """
        Initialize the class with hyper-parameters
        """
        self.cost_history = []
        self.w = 0
        self.b = 0
        self.lr = lr
    
    def prediction(self, X, weight, bias):
        return weight * X + bias
    
    def gradient_descent(self, X, y):
        """
        Update the gradient/weights
        """
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
        """
        Function to train the model
        Inputs:
            X: Input set of features
            y: Set of labels/outputs
        """
        for i in range(epochs):
            weight, bias = self.gradient_descent(X, y)
            cost = self.cost_function(X, y, weight, bias)
            self.cost_history.append(cost)

            # Log loop
            print("Iter {0}/{1}  weight={2:.3f}   bias={3:.4f}  cost={4:.4f}".format(
                i+1, epochs, weight, bias, cost
            ))
        
        return weight, bias
    
    def predict(self, X):
        """
        Function to predict the output of the value
        Input:
            X: Array of features to predict
        """
        pred = []
        for i in range(len(X)):
            pred.append(round((self.w * X[i]) + self.b, 2))

        return pred
    
    def score(self, y, y_pred):
        """
        Returns the score of the model
        """
        N = len(y)
        diff = []
        for i in range(N):
            diff.append(abs(y[i] - y_pred[i]))
        
        print("Accuracy: {:.2f}".format(1.0 - sum(diff) / N))


class MVLinearRegression():
    """
    This class is for multivariate linear regression
    """
    def __init__(self, lr=0.1):
        self.lr = lr
        self.weights = []
        self.cost_history = []
    
    def prediction(self, X, weight):
        return np.dot(X, weight)
    
    def cost_function(self, X, y, weight):
        N = len(y)
        predictions = self.prediction(X, weight)
        # print("COST FN -> pred {}  y {}".format(predictions.shape, y.shape))
        error = np.square(predictions - y)

        return (1.0 * error.sum()) / (2 * N)
    
    def gradient_descent(self, X, y, weight):
        predictions = self.prediction(X, weight)
        # print("In gred pred shape {} and weight shape {}".format(predictions.shape, weight.shape))
        error = y - predictions
        # print("y shape: {} error shape: {}".format(y.shape, error.shape))
        # print("X shape: {} error shape: {}".format(X.shape, error.shape))
        gradient = np.dot(-X.T, error)
        gradient /= len(y)
        gradient *= self.lr
        # print("Gradient Shape {}".format(gradient.shape))
        weight -= gradient
        return weight
    
    def preprocess_data(self, X):
        # normalize the input variable
        fmean = X.mean()
        frange = np.amax(X) - np.amin(X)

        normalized = (X - fmean) / frange
        bias = np.ones((len(X), 1))
        X = np.append(bias, normalized, axis=1)

        return X
    
    def fit(self, X, y, epochs=5):
        """
        Function to train the model
        Input:
            X: Input features [Array]
            y: Label/Output    [Array]
            epochs: No. of iterations to run the training   [Int]
        """
        # print("FIT Y SHAPE: {}".format(y.shape))
        # print(X.shape)
        weight = np.zeros((X.shape[1], 1))
        # print(weight.shape)
        for i in range(epochs):
            weight = self.gradient_descent(X, y, weight)
            cost = self.cost_function(X, y, weight)
            self.cost_history.append(cost)

            # Log loop
            print("Iter {0}/{1} cost={2:.4f}".format(
                i+1, epochs, cost
            ))
        self.weights = weight
        return self.weights
    
    def predict(self, X):
        """
        Weights and Input Variable are transposed
        to match matrix multiplication shape
        Inputs:
            X: Array to be predicted
        """
        return np.dot(self.weights.T, X.T)