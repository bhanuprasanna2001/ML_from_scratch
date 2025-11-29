import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

np.random.seed(42)

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


class svc_cvxopt:
    
    def __init__(self, C=1.0, gamma=0.5, threshold=1e-7):
        self.C = C
        self.gamma = gamma
        self.threshold = threshold
        
    def rbf_kernel(self, X1, X2, gamma=1.0):
        squared_distance = np.sum((X1 - X2) ** 2)
        return np.exp(-self.gamma * squared_distance)
        
    def fit(self, X_train, y_train):
        
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        
        # Convert labels to {-1, +1} for SVM dual formulation
        self.y_train = np.where(self.y_train == 0, -1, 1)
        
        K = np.zeros((self.X_train.shape[0], self.X_train.shape[0]))
        
        # Compute Kernel Matrix
        for i in range(self.X_train.shape[0]):
            for j in range(self.X_train.shape[0]):
                K[i,j] = self.rbf_kernel(self.X_train[i], self.X_train[j])
                
        from cvxopt import matrix, solvers
        
        # Hide cvxopt progress output
        solvers.options['show_progress'] = False
        
        # For cvxopt we need P, q, G, h, A, b
        P = matrix(np.outer(self.y_train, self.y_train) * K, tc="d")
        q = matrix(-1 * np.ones(self.X_train.shape[0]))
        
        # Box Constraints 0 <= alpha <= C
        G = matrix(np.vstack((-1 * np.eye(self.X_train.shape[0]), np.eye(self.X_train.shape[0]))))
        h = matrix(np.hstack((np.zeros(self.X_train.shape[0]), self.C * np.ones(self.X_train.shape[0]))))
        
        # Sum constrain: y.T * alpha = 0
        A = matrix(self.y_train.reshape(1, -1), tc="d")
        b = matrix(0.0, tc="d")
        
        print(f"Training SVM with CVXOPT...")
        print(f"Samples: {self.X_train.shape[0]}, Features: {self.X_train.shape[1]}")
        
        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])
        
        sv_indices = alpha > self.threshold
        
        self.lagr_multipliers = alpha[sv_indices]
        self.support_vectors = self.X_train[sv_indices]
        self.support_vector_labels = self.y_train[sv_indices]
        
        # Calculate intercept using average of support vectors on margin
        margin_sv_indices = (alpha > self.threshold) & (alpha < self.C - self.threshold)
        if np.sum(margin_sv_indices) > 0:
            b_values = []
            for idx in np.where(margin_sv_indices)[0]:
                b_val = self.y_train[idx]
                for i in range(len(self.lagr_multipliers)):
                    b_val -= self.lagr_multipliers[i] * self.support_vector_labels[i] * \
                            self.rbf_kernel(self.support_vectors[i], self.X_train[idx])
                b_values.append(b_val)
            self.intercept = np.mean(b_values)
        else:
            # Fallback: use first support vector
            self.intercept = self.support_vector_labels[0]
            for i in range(len(self.lagr_multipliers)):
                self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[i] * \
                                 self.rbf_kernel(self.support_vectors[i], self.support_vectors[0])
        
        print(f"Support vectors: {len(self.lagr_multipliers)}/{self.X_train.shape[0]} ({len(self.lagr_multipliers)/self.X_train.shape[0]*100:.1f}%)")
        print(f"Training complete!")
        
        return self
            
            
    def predict(self, X_test):
        y_pred = []
        for sample in X_test:
            prediction = 0
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[i] * self.rbf_kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
        
svc = svc_cvxopt(C=1.0, gamma=0.1)
svc.fit(X_train, y_train)
y_pred_test = svc.predict(X_test)

# Convert predictions back to {0, 1} for metrics
y_pred_test_binary = np.where(y_pred_test == -1, 0, 1)

# Calculate metrics
accuracy = np.mean(y_pred_test_binary == y_test)
tp = np.sum((y_test == 1) & (y_pred_test_binary == 1))
tn = np.sum((y_test == 0) & (y_pred_test_binary == 0))
fp = np.sum((y_test == 0) & (y_pred_test_binary == 1))
fn = np.sum((y_test == 1) & (y_pred_test_binary == 0))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'='*60}")
print("TEST METRICS")
print(f"{'='*60}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")


# Plot: Support Vector Distribution (single comprehensive plot)
# Get all alphas including zeros
all_alphas = np.zeros(len(X_train))
sv_indices = svc.lagr_multipliers > 0
# Map support vector alphas back to original training indices
for i, (alpha, sv) in enumerate(zip(svc.lagr_multipliers, svc.support_vectors)):
    # Find original index
    for j in range(len(X_train)):
        if np.allclose(X_train[j], sv):
            all_alphas[j] = alpha
            break

# Categorize support vectors
margin_sv = np.sum((all_alphas > svc.threshold) & (all_alphas < svc.C - svc.threshold))
bound_sv = np.sum(np.abs(all_alphas - svc.C) < svc.threshold)
n_sv = len(svc.lagr_multipliers)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot alpha values
ax.scatter(np.arange(len(all_alphas)), all_alphas, c='blue', alpha=0.6, s=50, edgecolors='black')
ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='α = 0 (Non-SV)')
ax.axhline(y=svc.C, color='red', linestyle='--', linewidth=2, label=f'α = C (Bound SV)')

ax.set_xlabel('Training Sample Index', fontsize=12)
ax.set_ylabel('Alpha (Lagrange Multiplier)', fontsize=12)
ax.set_title(f'Support Vector Distribution (CVXOPT)\nTest Acc: {accuracy:.3f} | SVs: {n_sv}/{len(X_train)} ({n_sv/len(X_train)*100:.1f}%) | On Margin: {margin_sv} | At Bound: {bound_sv}', 
             fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('output_figs/svc_cvxopt', exist_ok=True)
plt.savefig('output_figs/svc_cvxopt/support_vectors.png', dpi=150, bbox_inches='tight')
print(f"\nSupport vectors plot saved to: output_figs/svc_cvxopt/support_vectors.png")
plt.show()

print(f"\n{'='*60}")

