# SVM for classification using Sequential Minimal Optimization to bring Kernel Trick to life
# References: https://cs229.stanford.edu/materials/smo.pdf

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

np.random.seed(42)

X, y = make_classification(n_samples=500,
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


class smo_svc:
    
    def __init__(self, C=0.1, kernel="rbf", gamma=10.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        
    def rbf_kernel(self, X1, X2, gamma=1.0):
        squared_distance = np.sum((X1 - X2) ** 2)
        return np.exp(-self.gamma * squared_distance)
        
    
    def fit(self, X_train, y_train, iterations=1, tolerance=1e-5, max_passes=5):
        X_train = X_train.copy()
        y_train = y_train.copy()
        
        self.X_train = X_train
        self.y_train = y_train
        
        # Step 1: Convert labels to -1, +1
        y_train[y_train == 0] = -1
        y_train[y_train == 1] = +1
        
        # Step 2: Intialize the parameters alpha and b
        alpha = np.zeros(X_train.shape[0])
        b = 0.0
        
        # Just checking shapes
        print(f"Alpha shape: {alpha.shape}, y_train shape: {y_train.shape}, X_train shape: {X_train.shape}")
        
        # Full range possibilities for j
        j_range = np.arange(X_train.shape[0])
        
        # Step 3: Outer loop
        passes = 0
        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(X_train.shape[0]):
                # Step 4: Calculate Error for i
                K_i = np.array([self.rbf_kernel(X_train[k], X_train[i], self.gamma) 
                                for k in range(len(X_train))])
                E_i = np.sum(alpha * y_train * K_i) + b - y_train[i]
                
                # Step 5: KKT Violation Check
                if ((y_train[i] * E_i < -tolerance and alpha[i] < self.C) or (y_train[i] * E_i > tolerance and alpha[i] > 0)):
                    # Step 6: Choose j != i randomly
                    j = np.random.choice(j_range[j_range != i])
                    
                    # Step 7: Calculate Error for j
                    K_j = np.array([self.rbf_kernel(X_train[k], X_train[j], self.gamma) 
                                    for k in range(len(X_train))])
                    E_j = np.sum(alpha * y_train * K_j) + b - y_train[j]
                    
                    # Step 8: Save old alphas both i and j
                    old_alpha_i = alpha[i]
                    old_alpha_j = alpha[j]
                    
                    # Step 9: Compute the bounds L and H
                    L, H = 0, 0
                    if y_train[i] != y_train[j]:
                        L = np.max([0, alpha[j] - alpha[i]])
                        H = np.min([self.C, self.C + alpha[j] - alpha[i]])
                    else:
                        L = np.max([0, alpha[i] + alpha[j] - self.C])
                        H = np.min([self.C, alpha[i] + alpha[j]])
                    
                    # Step 10: Check if bounds are equal or not    
                    if L == H:
                        continue
                    
                    # Step 11: Calculate mu
                    mu = 2 * self.rbf_kernel(X_train[i], X_train[j]) - self.rbf_kernel(X_train[i], X_train[i]) - self.rbf_kernel(X_train[j], X_train[j])
                    
                    # Step 12: Check mu >= 0
                    if mu >= 0:
                        continue
                    
                    # Step 13: Now compute and clip new value for alpha_j
                    alpha_j_new_unclipped = alpha[j] - ((y_train[j] * (E_i - E_j)) / mu)
                    
                    if alpha_j_new_unclipped > H:
                        alpha[j] = H
                    elif alpha_j_new_unclipped < L:
                        alpha[j] = L
                    else:
                        alpha[j] = alpha_j_new_unclipped
                    
                    # Step 14: Check if abs of old and new alpha_j is within tolerance
                    if (np.abs(alpha[j] - old_alpha_j) < 1e-5):
                        continue
                    
                    # Step 15: Determine value for alpha_i
                    alpha[i] = alpha[i] + (y_train[i] * y_train[j] * (old_alpha_j - alpha[j]))
                    
                    # Step 16: Compute b1 and b2
                    b_1 = b - E_i - (y_train[i] * (alpha[i] - old_alpha_i) * self.rbf_kernel(X_train[i], X_train[i])) - (y_train[j] * (alpha[j] - old_alpha_j) * self.rbf_kernel(X_train[i], X_train[j]))
                    b_2 = b - E_j - (y_train[i] * (alpha[i] - old_alpha_i) * self.rbf_kernel(X_train[i], X_train[j])) - (y_train[j] * (alpha[j] - old_alpha_j) * self.rbf_kernel(X_train[j], X_train[j]))
                    
                    # Step 17: Compute b
                    if alpha[i] > 0 and alpha[i] < self.C:
                        b = b_1
                    elif alpha[j] > 0 and alpha[j] < self.C:
                        b = b_2
                    else:
                        b = (b_1 + b_2) / 2
                        
                    # Step 18: Update num_changed_alphas
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
                
        # Step 19: Save alpha and b
        self.alpha = alpha
        self.b = b
                
        return self
    
    def predict(self, X_test):
        sv_indices = np.where(self.alpha > 1e-5)[0]
        
        print(f"Number of support vectors: {len(sv_indices) / len(self.alpha)}")
        
        alpha_sv = self.alpha[sv_indices]
        y_sv = self.y_train[sv_indices]
        X_sv = self.X_train[sv_indices]
        
        n_test = X_test.shape[0]
        predictions = np.zeros(n_test)
        
        for i in range(n_test):
            K_test = np.array([self.rbf_kernel(X_sv[k], X_test[i], self.gamma) 
                          for k in range(len(X_sv))])
            f_x = np.sum(alpha_sv * y_sv * K_test) + self.b
            predictions[i] = np.sign(f_x)
        
        return predictions
            
        
svc = smo_svc(C=1.0, gamma=0.1)
svc.fit(X_train, y_train, max_passes=10)

# Get predictions on test set
y_pred_test = svc.predict(X_test)

# Convert predictions back to 0, 1 for metrics
y_pred_test_binary = np.where(y_pred_test == -1, 0, 1)

print(f"\nFinal bias: {svc.b}")
print(f"Total support vectors: {np.sum(svc.alpha > 1e-5)} / {len(svc.alpha)}")

# Plotting and evaluation functions

def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png', output_dir='output_figs/svc_smo'):
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
    ax.set_title(f'Confusion Matrix (SMO-SVM with RBF Kernel)\nAcc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}', 
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

# Support vector visualization
y_train_signed = np.where(y_train == 0, -1, 1)
sv_mask = svc.alpha > 1e-5
n_sv = np.sum(sv_mask)

# Categorize support vectors
margin_sv = np.sum((svc.alpha > 1e-5) & (svc.alpha < svc.C - 1e-5))
bound_sv = np.sum(np.abs(svc.alpha - svc.C) < 1e-5)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot alpha values
ax.scatter(np.arange(len(svc.alpha)), svc.alpha, c='blue', alpha=0.6, s=50, edgecolors='black')
ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='α = 0 (Non-SV)')
ax.axhline(y=svc.C, color='red', linestyle='--', linewidth=2, label=f'α = C (Bound SV)')

ax.set_xlabel('Training Sample Index', fontsize=12)
ax.set_ylabel('Alpha (Lagrange Multiplier)', fontsize=12)
ax.set_title(f'Support Vector Distribution\nTotal SVs: {n_sv}/{len(svc.alpha)} ({n_sv/len(svc.alpha)*100:.1f}%) | On Margin: {margin_sv} | At Bound: {bound_sv}', 
             fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('output_figs/svc_smo', exist_ok=True)
plt.savefig('output_figs/svc_smo/support_vectors.png', dpi=150, bbox_inches='tight')
print(f"\nSupport vectors plot saved to: output_figs/svc_smo/support_vectors.png")
plt.show()

plot_confusion_matrix(y_test, y_pred_test_binary, filename='confusion_matrix_smo.png')