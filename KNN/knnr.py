# I have already explained the regression in the knn.py.
# The knn.py has the full implementaiton of the classification both weighted and unweighted.

# This script will contain the implementaiton of the KNN for Regression both the weighted and unweighted.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

np.random.seed(42)

X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42) # type: ignore

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

class knn_unweighted_regression:
    
    def __init__(self, k=5):
        self.k = k
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_test):
        
        y_hat = []
        for i in range(X_test.shape[0]):
            # Step 1: Calculate the distance from all training points
            dis_tt = np.sqrt(np.sum((self.X_train - X_test[i]) ** 2, axis=1))
            
            # Step 2: Select k nearest neighbours
            ks_idx = np.argpartition(dis_tt, self.k)[:self.k]
            
            # Step 3: Get the values from y_train for the k idx
            values_idx = self.y_train[ks_idx]
            
            # Step 4: Append the mean of each row containing the values of k nearest neighbours 
            y_hat.append(np.mean(values_idx))
            
        return y_hat
    

class knn_weighted_regression:
    
    def __init__(self, k=5, p=2, epsilon=1e-6):
        self.k = k
        self.p = p
        self.epsilon = epsilon
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_test):
        
        y_hat = []
        for i in range(X_test.shape[0]):
            # Step 1: Compute the distance from all training points
            dis_tt = np.sqrt(np.sum((self.X_train - X_test[i]) ** 2, axis=1))
            
            # Step 2: Find the indexes of k nearest neighbours
            ks_idx = np.argpartition(dis_tt, self.k)[:self.k]
            dis_tt = dis_tt[ks_idx]
            values_idx = self.y_train[ks_idx]
            
            # Step 3: Calculate the weight
            w = 1 / (((dis_tt) ** self.p) + self.epsilon)
            
            # Step 4: Calculate the weighted average of their target values
            wa = np.average(values_idx, weights=w)
            
            # Step 5: Append the wa to y_hat
            y_hat.append(wa)
            
        return y_hat
            
            
            
# Train both models
knnr_unweighted = knn_unweighted_regression(k=5)
knnr_unweighted.fit(X_train, y_train)
y_pred_unweighted = knnr_unweighted.predict(X_test)

knnr_weighted = knn_weighted_regression(k=5, p=2)
knnr_weighted.fit(X_train, y_train)
y_pred_weighted = knnr_weighted.predict(X_test)


def plot_regression_results(y_true, y_pred, model_name='KNN Regression', output_dir='output_figs/knn_regression'):
    """
    Plot regression results with predicted vs actual and residual plot.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model for plot title
    output_dir : str
        Directory to save plots
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    residuals = y_true - y_pred
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Predicted vs Actual
    ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('True Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.set_title(f'{model_name}\nPredicted vs Actual', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Residuals
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'regression_results.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    
    # Print metrics
    print(f"\n{model_name} Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"\nPlot saved to: {filepath}")
    
    plt.show()
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}


# Get metrics for both
print("\n" + "="*60)
print("KNN REGRESSION RESULTS")
print("="*60)

metrics_unweighted = plot_regression_results(y_test, y_pred_unweighted, 
                                            'KNN Unweighted Regression (k=5)',
                                            'output_figs/knn_unweighted_regression')

metrics_weighted = plot_regression_results(y_test, y_pred_weighted,
                                          'KNN Weighted Regression (k=5, p=2)',
                                          'output_figs/knn_weighted_regression')

# Quick comparison
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"{'Metric':<10} {'Unweighted':<15} {'Weighted':<15} {'Difference':<15}")
print("-"*60)
print(f"{'MAE':<10} {metrics_unweighted['mae']:<15.4f} {metrics_weighted['mae']:<15.4f} "
      f"{metrics_weighted['mae'] - metrics_unweighted['mae']:>+14.4f}")
print(f"{'RMSE':<10} {metrics_unweighted['rmse']:<15.4f} {metrics_weighted['rmse']:<15.4f} "
      f"{metrics_weighted['rmse'] - metrics_unweighted['rmse']:>+14.4f}")
print(f"{'R²':<10} {metrics_unweighted['r2']:<15.4f} {metrics_weighted['r2']:<15.4f} "
      f"{metrics_weighted['r2'] - metrics_unweighted['r2']:>+14.4f}")
print("="*60 + "\n")
