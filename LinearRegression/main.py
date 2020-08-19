from linear_regression import LinearRegression1V

lr = LinearRegression1V()

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
X_test = [5, 6, 7]

lr.fit(X, y, epochs=50)

# print(lr.predict(X_test))

# lr.score([13, 15, 17], lr.predict(X_test))