# Let's implement SVR from scratch using the sub gradient descent method.

# For this we will be using the Epsilon Insensitive Loss. Which works like tube.

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
print(f"y_train range: [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"y_train std: {y_train.std():.2f}\n")

class linear_svr:
    
    def __init__(self, C=0.1, epsilon=10.0, learning_rate=0.01):
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.loss_history = []
        
    def fit(self, X_train, y_train, iterations=1000, threshold=1e-6):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        
        w = np.zeros(self.X_train.shape[1])
        b = 0
        
        total_loss_old = -np.inf
        
        for epoch in range(iterations):
            epoch_loss = 0
            for i in range(self.X_train.shape[0]):
                f_i = np.dot(w, self.X_train[i]) + b
                residual = self.y_train[i] - f_i
                abs_residual = np.abs(residual)
                
                if abs_residual <= self.epsilon:
                    # Inside tube: only regularization
                    dw = w
                    db = 0
                    loss_i = 0
                else:
                    # Outside tube: check which side
                    if residual > self.epsilon:
                        # Prediction too low: y - f(x) > epsilon
                        dw = w - self.C * self.X_train[i]
                        db = -self.C
                    else:
                        # Prediction too high: f(x) - y > epsilon
                        dw = w + self.C * self.X_train[i]
                        db = self.C
                    loss_i = abs_residual - self.epsilon
                    
                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db
                
                epoch_loss += loss_i
                
            total_loss = 0.5 * np.dot(w, w) + self.C * epoch_loss
            self.loss_history.append(total_loss)
            
            if abs(total_loss_old - total_loss) < threshold:
                print(f"Converged at epoch {epoch}")
                break
            
            if epoch % 100 == 0 or epoch == iterations - 1:
                print(f"Epoch {epoch}/{iterations} - Loss: {total_loss:.4f}")
                
            total_loss_old = total_loss
        
        self.w = w
        self.b = b
        
        return w, b
    
    def predict(self, X_test):
        return np.dot(X_test, self.w) + self.b
    

svr = linear_svr(C=1.0, epsilon=10.0, learning_rate=0.01)
w, b = svr.fit(X_train, y_train, iterations=1000)

# Make predictions
y_pred_train = svr.predict(X_train)
y_pred_test = svr.predict(X_test)

# Calculate metrics
train_mse = np.mean((y_train - y_pred_train) ** 2)
test_mse = np.mean((y_test - y_pred_test) ** 2)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

ss_res_test = np.sum((y_test - y_pred_test) ** 2)
ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
r2_test = 1 - (ss_res_test / ss_tot_test)

print(f"\nFinal weights: {w}")
print(f"Final bias: {b}")
print(f"\nTest RMSE: {test_rmse:.4f}")
print(f"Test R²: {r2_test:.4f}")

# Identify support vectors (points outside epsilon tube)
residuals_train = np.abs(y_train - y_pred_train)
sv_mask = residuals_train > svr.epsilon
n_sv = np.sum(sv_mask)
print(f"Support vectors: {n_sv}/{len(y_train)} ({n_sv/len(y_train)*100:.1f}%)")


# Plot 1: Loss History
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(len(svr.loss_history)), svr.loss_history, linewidth=2, color='blue')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (Regularization + ε-insensitive)', fontsize=12)
ax.set_title('SVR Training Loss over Epochs', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('output_figs/svr_subgradient', exist_ok=True)
plt.savefig('output_figs/svr_subgradient/loss_history.png', dpi=150, bbox_inches='tight')
print(f"\nLoss history plot saved to: output_figs/svr_subgradient/loss_history.png")
plt.show()


# Plot 2: Epsilon-Insensitive Tube (single comprehensive plot)
# Sort by predictions for better visualization
sort_idx = np.argsort(y_pred_train)
y_train_sorted = y_train[sort_idx]
y_pred_sorted = y_pred_train[sort_idx]
residuals_sorted = residuals_train[sort_idx]

fig, ax = plt.subplots(figsize=(12, 6))

# Plot predictions and actual values
ax.plot(range(len(y_pred_sorted)), y_pred_sorted, 'b-', lw=2, label='Predicted f(x)', zorder=3)

# Plot epsilon tube
ax.fill_between(range(len(y_pred_sorted)), 
                y_pred_sorted - svr.epsilon,
                y_pred_sorted + svr.epsilon,
                alpha=0.3, color='yellow', label=f'ε-tube (ε={svr.epsilon})', zorder=1)

# Plot actual points
inside_tube = residuals_sorted <= svr.epsilon
indices = np.arange(len(y_train_sorted))

ax.scatter(indices[inside_tube], y_train_sorted[inside_tube], 
          c='green', s=20, alpha=0.5, label=f'Inside tube ({np.sum(inside_tube)} pts)', zorder=2)
ax.scatter(indices[~inside_tube], y_train_sorted[~inside_tube], 
          c='red', s=40, alpha=0.7, marker='^', label=f'Support vectors ({np.sum(~inside_tube)} pts)', zorder=4)

ax.set_xlabel('Sample Index (sorted by prediction)', fontsize=12)
ax.set_ylabel('Target Value', fontsize=12)
ax.set_title(f'ε-Insensitive Loss Tube\nTest RMSE: {test_rmse:.2f} | R²: {r2_test:.3f} | SVs: {n_sv}/{len(y_train)} ({n_sv/len(y_train)*100:.1f}%)', 
            fontsize=14)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('output_figs/svr_subgradient/epsilon_tube.png', dpi=150, bbox_inches='tight')
print(f"Epsilon tube plot saved to: output_figs/svr_subgradient/epsilon_tube.png")
plt.show()