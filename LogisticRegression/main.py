from logistic_regression import LogisticRegression
import numpy as np
log_reg = LogisticRegression(lr=0.3)

X = np.matrix([[8.2, 8.5],
    [2.4, 8.0],
    [1.5, 7.5],
    [4.5, 8.0]])

y = np.matrix([[1],
    [0],
    [0],
    [1]])

X_test = np.matrix([8.0, 7.0])

log_reg.fit(X, y, epochs=100)

print(log_reg.predict(X_test))