# So, it is the same as Linear Regression Gradient Descent.
# The important thing is that now we have an activation function sigmoid.
# In linear regression we did not have a activation function (Identity Activation Function),
# But it is not considered as an activation function.

# Now, let's see, the equation remains the same y = Xw + b.
# But now for y_hat we have to do the following:
# y_hat = sigmoid(z)
# z = Xw + b

# So, the optimization looks like this:
# We are again trying to find dl/dw and dl/db.
# where the loss function l has changed to Binary Cross Entropy loss taken from
# the negative log likelihood of bernoulli.
# l(y_hat, y) = - (y * log y_hat + (1 - y) * log (1 - y_hat))

# dl/dw = dl/dy_hat * dy_hat/dz * dz/dw = (y_hat - y) * X
# dl/db = dl/dy_hat * dy_hat/dz * dz/db = (y_hat - y)

# dj/dw = 1/m * X.T @ (y_hat - y)
# dj/db = 1/m * (y_hat - y)

# First Let's create data

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000,
                           n_features=5,
                           n_informative=3,
                           n_redundant=1,
                           n_classes=2,
                           random_state=42)

# Printing shapes of data
print(f"X_shape: {X.shape}, y_shape: {y.shape}")

# Now let's split the data into train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Let's start Logistic Regression implementation

import os
import numpy as np
import matplotlib.pyplot as plt

def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    # Logistic Regression from Scratch
    
    w = np.random.rand(X.shape[1])
    b = np.random.rand(1)
    m = X.shape[0]
    
    cost_list = []
    
    for i in range(iterations):
        z = (X @ w + b)
        y_hat = 1 / (1 + np.power(np.e, -z))
        
        dj_dw = (1/m) * (X.T @ (y_hat - y))
        dj_db = (1/m) * np.sum(y_hat - y)
        
        w = w - (learning_rate * dj_dw)
        b = b - (learning_rate * dj_db)
        
        cost_test = -1 * ((y * np.log(y_hat + 1e-15)) + ((1 - y) * np.log(1 - y_hat + 1e-15)))
        cost_test = (1 / m) * np.sum(cost_test)
        cost_list.append(cost_test)
        
        if i % 100 == 0 or i == iterations - 1:
            print(f"Iteration: {i} -> Cost: {cost_test}")
            
    return w, b, cost_list

w, b, cost_history = logistic_regression(X_train, y_train)

print(f"Weights: {w}")
print(f"Bias: {b}")

# Now evaluation phase

def evaluate (X, y, w, b):
    z = X @ w + b
    y_hat = 1 / (1 + np.power(np.e, -z))
    
    m = X.shape[0]
    
    test_cost = -1 * ((y * np.log(y_hat + 1e-15)) + ((1 - y) * np.log(1 - y_hat + 1e-15)))
    test_cost = 1 / m * np.sum(test_cost)
    
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0
    
    return np.vstack([["  y", "y_hat"], np.column_stack((y, y_hat))]), test_cost


tested_y, test_cost = evaluate(X_test, y_test, w, b)

print(f"Y and Y_hat comparison:\n{tested_y[:6]}")
print(f"Test cost: {test_cost}")


def plot_cost_history(cost_history, filename='cost_history.png', output_dir='output_figs/gradient_descent'):
    """
    Plot training cost over iterations.
    
    Parameters:
    -----------
    cost_history : list
        List of cost values at each iteration
    filename : str
        Output filename
    output_dir : str
        Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(cost_history)), cost_history, linewidth=2, color='blue')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
    ax.set_title('Training Cost over Iterations', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Cost history plot saved to: {filepath}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png', output_dir='output_figs/gradient_descent'):
    """
    Plot confusion matrix for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred : array-like
        Predicted binary labels (0 or 1)
    filename : str
        Output filename
    output_dir : str
        Directory to save the plot
    """
    # Compute confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=20)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix\nAcc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}', 
                 fontsize=14)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nClassification Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion matrix plot saved to: {filepath}")
    plt.show()
    
# Plot results
# Remove header row and extract columns
y_true = tested_y[1:, 0].astype(float).astype(int)
y_pred = tested_y[1:, 1].astype(float).astype(int)
plot_cost_history(cost_history)
plot_confusion_matrix(y_true, y_pred)