import os
import numpy as np
import matplotlib.pyplot as plt


def plot_linear_regression(X, y, X_bar, theta, title=None, filename=None, output_dir="output_figs/closed_form"):
    """
    Plot linear regression fit using your existing variables.
    
    Parameters:
    -----------
    X : array-like, shape (n,) or (n,1)
        Feature vector (original, without intercept column)
    y : array-like, shape (n,)
        Target values
    X_bar : array-like, shape (n, 2)
        Design matrix with intercept column [1, X]
    theta : array-like, shape (2,) or (2,1)
        Optimized parameters [intercept, slope]
    title : str, optional
        Plot title
    filename : str, optional
        Output filename (e.g., 'plot1.png'). If None, plot is shown but not saved.
    output_dir : str, default='output_figs/closed_form'
        Directory to save the plot
    """
    # Ensure 1-D arrays for plotting
    X_plot = np.asarray(X).ravel()
    y_plot = np.asarray(y).ravel()
    theta_vec = np.asarray(theta).ravel()
    
    # Compute predictions
    y_pred = (X_bar @ theta_vec.reshape(-1, 1)).ravel()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_plot, y_plot, alpha=0.6, label="Data", s=30)
    
    # Sort for clean line plot
    idx = np.argsort(X_plot)
    ax.plot(X_plot[idx], y_pred[idx], color="red", linewidth=2, label="Fitted Line")
    
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")
    
    plt.show()


# Let's generate random data
# Circle data would be great

X = np.arange(1,101)
y = np.square(X) * np.pi

# Printing Data
print(np.column_stack((X, y))[:5])

# Now as we now that Linear Regression Closed Form
# Theta = ((X)^T * X)^-1 * X^T * y
# Where Theta is the optimal parameters
# We need to add another column of 1:
	# - Intercept (bias)
	# - Without the Intercept the fitted line is forced to pass through the origin.
	# - So we form a design matrix X = [1 X] and use the Theta to compute the closed form solution
 
X_bar = np.column_stack((np.ones(X.shape[0]), X))

# Printing Data
print(np.column_stack((X_bar, y))[:5])

# Printing Shape
print(f"X_bar_shape: {X_bar.shape}, y_shape: {y.shape}")

try:
	l_h_t = np.linalg.inv(np.transpose(X_bar) @ (X_bar))
	r_h_t = X_bar.T @ y
	theta = l_h_t @ r_h_t
except Exception as e:
    print(e)
    
print(f"Theta Shape: {theta.shape}")

print(f"Optimized parameters: {theta}")

plot_linear_regression(X, y, X_bar, theta, title="Handmade Dataset", filename="handmade_fit.png")

print("\n")

# So, let's do sklearn.datasets and make_regression

from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42) # type: ignore

# Printing Shape
print(f"X_shape: {X.shape}, y_shape: {y.shape}")

# Printing Data
print(np.column_stack((X, y))[:5])

X_bar = np.column_stack((np.ones(X.shape[0]), X))

# Printing Data
print(np.column_stack((X_bar, y))[:5])

# Printing Shape
print(f"X_bar_shape: {X_bar.shape}, y_shape: {y.shape}")

try:
    l_h_t = np.transpose(X_bar) @ (X_bar)
    l_h_t = np.linalg.inv(l_h_t)
    r_h_t = X_bar.T @ y
    theta = l_h_t @ r_h_t
except Exception as e:
    print(e)
    
print(f"Theta Shape: {theta.shape}")

print(f"Optimized parameters: {theta}")

plot_linear_regression(X, y, X_bar, theta, title="make_regression Dataset", filename="sklearn_fit.png")