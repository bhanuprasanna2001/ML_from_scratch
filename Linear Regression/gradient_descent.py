# It took me a long time to perfectly understand gradient descent.
# But in short it is just the chain rule underneath.

# The algorithm doing all this is Backpropagation calculating gradients 
# in this process allowing the gradient descent.

# So, for linear regression, the equation looks like this:
# y = mx + b
# where if we push it to matrix notation then
# y = X @ theta + B or
# y = X_bar @ theta where the intercept (bias) is included in X_bar
# X_bar is the design matrix

# So, we are trying to find the theta and B (intercept or bias) where 
# the X gets as close as to Y, when we pluggin X and the optimized theta and B.

# It is all partial derivatives underneath, that is more terms (or multiple variable 
# not just one like in ordinary differential equations or normal differential equations) 
# to derivate with.

# We have a loss function l(y[i], y_hat[i]) = 1/2 * (y_hat[i] - y[i])^2 
# where y_hat is the prediction y_hat = X_bar @ theta
# where y is the true prediction

# So, our goal is to get y_hat as close as possible to y by minimzing the loss.
# which we do by minimizing the loss and the total cost
# the cost J = 1 / m * sigma (i = 0 to m) l(y[i], y_hat[i])
# sometimes it is also 1/2m rather than just 1/m for expanded loss.
# where m is the number of data samples or the number of input data.

# Then we do delta (referred to as d throughout for brevity - concise notation) 
# dl/dw = dl/dy_hat * dy_hat/dw = (y_hat - y) * x for just one sample
# dl/db = dl/dy_hat * dy_hat/db = (y_hat - y)
# dj/dw = (1/m) * sigma dl/dw
# dj/db = (1/m) * sigma dl/db
# This is the chain rule if you clearly observe it, so this applies for more complex
# architectures like FFN with multiple layers, which goes on. But the final goal is to 
# minimize the loss.
# But as all these are matrix operations it is much more easier for us to do, Lord NumPy.

# At final after solving these, we get:
# dj/dw = (1/m) * X.T @ (y_hat - y)
# dj/db = (1/m) * sigma (y_hat[i] - y[i])

# So, we just found the gradients but haven't yet updated the w and b to get closer, we need
# another term weird_n which is the learning rate that we have to use.
# w = w - weird_n * dj/dw
# b = b - weird_n * dj/db

# this happens over a lot of iterations which allows us to minimize the loss as much as possible.

# We will look at the example by starting to write the code.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42) # type: ignore

# Printing shape
print(f"X_shape: {X.shape}, y_shape: {y.shape}\n")

# Now we create the Design matrix of X to include the Intercept (Bias)
X_bar = np.column_stack((np.ones(X.shape[0]), X))

# Printing shape and few rows of the data
print(f"X_shape: {X_bar.shape}, y_shape: {y.shape}")
print(np.column_stack((X, y)), "\n")

# Let's divide the data to train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")
print(f"y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")

# So, now we have X and y in place, we want to establish:
# - Loss function - MSE = 1/2 (y_hat - y)^2
# - Learning Rate - weird_n

def linear_regression(X: np.ndarray, y: np.ndarray, weird_n=0.01, iterations=10):
    # Linear Regression function
    
    m = X.shape[0] # Number of samples
    
    w = np.random.rand(X.shape[1])
    bias = np.random.rand(1)
    
    for i in range(iterations):
        y_hat = X @ w + bias
        dl_dw = X.T @ (y_hat - y)
        dl_db = (y_hat - y)
        
        dj_dw = (1/m) * dl_dw
        dj_db = (1/m) * np.sum(dl_db)
        
        w = w - weird_n * dj_dw
        bias = bias - weird_n * dj_db
        
        if i % 100 == 0 or i == iterations - 1:
            cost = (1/(2*m)) * np.sum((y_hat - y)**2)
            print(f"Iteration {i}, Cost: {cost}")
    
    return w, bias

w, b = linear_regression(X_train, y_train, iterations=1000)

print(f"\nWeights: {w}")
print(f"Bias: {b}")

# Now let's evaluate the Linear Regression model using the test data
def evaluate(X, y, w, b):
    # Evaluating Linear Regression model
    
    m = X.shape[0]
    
    y_hat = X @ w + b
    
    test_cost = (1/(2*m)) * np.sum((y_hat - y)**2)
    
    return np.vstack([["        y       ", "        y_hat      "], np.column_stack((y, y_hat))]), test_cost

tested_y, test_cost = evaluate(X_test, y_test, w, b)

print("Tested y (Brevity - 5): \n", tested_y[:5])

print(f"Total cost of test data: {test_cost}")