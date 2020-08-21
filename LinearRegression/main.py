from linear_regression import MVLinearRegression
import numpy as np
lr = MVLinearRegression()

X = np.matrix([
             [230.1, 37.8, 69.1],
             [44.5, 39.3, 23.1],
             [17.2, 45.9, 34.7],
             [151.5, 41.3, 13.2]
             ])

y = np.array([[22.1],
            [10.4],
            [18.3],
            [18.5]])
# print("Y SHAPE {}".format(y.shape))
X = lr.preprocess_data(X)

lr.fit(X, y, epochs=100)