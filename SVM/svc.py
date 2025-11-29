# You thought SGD is Stochastic Gradient Descent
# No it is Sub Gradient Descent.

# So, let's implement SVM - Support Vector Machine.
# First let's tackle the Classification side.

# So, the main idea of the SVM is to find a optimal hyperplane that seperates classes
# with the maximum margin.
# The margin is nothing but from the hyperplane the data points of both classes
# must be at a wider distance.
# SVM goal is to Maximizes the margin distance to nearest points from each class.
# Margin: Distance between hyperplane and nearest points.

# Why Maximum margin is due to it's robustness to noise and proven to reduce overfitting.
# Also, it generalizes better to unseen data.
# In 2d it is a line, 3d it is a plane, n-D it is w.T @ x + b = 0

# Support vectors: The data points closest to the hyperplane (that define the margin)

# So, the classifier looks like this:
# f(x) = sign(w.T @ x + b)
# Where the sign is the Signum function where if f(x) < 0 it is -1, f(x) == 0 it is 0, f(x) > 0 it is 1

# I can't explain the whole lagrangian multiplier, the dual concept here because it is too long.
# For now, we go from Hard Margin to Soft margin where we see the dual concept come to life which
# extends us to the Kernel trick which is the core of SVM.

# The hard margin concept is for linear svm that we can't allow any misclassifications. The division or
# the decision boundary or the hyperplane must perfeclty divide all the data points.
# The margin or the distance to hyperplane is gamma = 1 / ||w||.
# So the hard margin primal optimization is that:
# min_w,b 1/2 * ||w||^2
# such that y_i * (w.T @ x + b) >= 1 for all i
# This is a convex quadratic optimization with linear constraints.
# As we have constraints we can use the langrangian multipliers concept to build a Dual Formulation for the SVM.
# Now like the dual problem forumation for optimization looks like:
# max_alpha W(alpha) = sigma (i = 1 to m) alpha_i - ((1/2) * sigma(i = 1 to m) sigma(j = 1 to m) alpha_i * alpha_j * y_i * y_j * x_i.T * x_j
# such that alpha_i >= 0 for all i
# where sigma(i = 1 to m) alpha_i * y_i which we get from dl/db, there is also dl/dw.
# What is alpha_i - It is a lagrange multiplier that we get when we rewrite the constraints in the form of g_i(w,b) <= 0 => g_i(w,b) = 1 - y_i * (w.T @ x_i + b) <= 0
# Because of the inequality constrain that we have y_i * (w.T @ x_i + b) >= 1
# So, we introduce lagrange multiplier for each constrain g_i.

# The main problem with hard margin is the inequality constraint which are conditions that demand every single
# training data point must b classified correctly and must lie outside the margin or exactly on it's boundary.
# There is no tolerance for misclassification or points failing within the margin.

# That is why we move from hard margin to soft margin, where we actually allow some tolerance to the classes to be misclassified.
# We have this term C which is the regularization term. Also, with the langrange multipliers we also introduce slack variables to allow violations.
# For the Soft margin SVM, we have the Primal optimization as:
# min_w,b,{slack_i} 1/2 ||w||^2 + C * sigma(i = 1 to m) slack_i
# such that y_i * (w.T @ x_i + b) >= 1 - slack_i
# slack_i >= 0

# Then, we do the same lagrange multiplier with g_i(w,b) => g_i(w,b,slack_i) for soft margin case with slack variables.
# When we do the dl/dw and dl/db, we get the same as hard margin case, but we have another term, dl/dslack_i which provides us that
# the alpha_i must be between 0 <= alpha_i <= C. this is a important because in hard margin we have only alpha_i >= 0.
# With C, we can actually control trade-off between margin width and violations.
# C is a hyperparameter.
# The support vectors are those with alpha_i > 0. The misclassified or margin-violating ones often have alpha_i = C.
# So, the dual problem formulation for soft margin is:
# w = sigma(i = 1 to m) alpha_i * y_i * x_i
# also we know that sigma(i = 1 to m) alpha_i * y_i = 0
# So the dual looks like the following:
# max_alpha W(alpha) = sigma(i = 1 to m) alpha_i - 1/2 sigma(i = 1 to m) sigma(j = 1 to m) alpha_i * alpha_j * y_i * y_j * x_i.T * x_j
# such that 0 <= alpha_i <= C,
# sigma(i = 1 to m) alpha_i * y_i = 0
# The decision function is the same for hard margin and soft margin:
# f(x) = sigma(i = 1 to m) alpha_i * y_i * x_i.T * x + b

# Then, we introduce the kernel trick which is just that it maps our inner products to n dimension to divide the data better.
# x_i.T * X => K(x_i, x_j), this is the kernel trick. The dual becomes:
# W(alpha) = sigma(i = 1 to m) alpha_i - 1/2 sigma(i = 1 to m) sigma(j = 1 to m) alpha_i * alpha_j * y_i * y_j * K(x_i, x_j)
# the classifier: f(x) = sign(sigma(i = 1 to m) alpha_i * y_i * K(x_i, x) + b)

# The concept for SVR is the same, but we use 2 slack variables and a new loss function epsilon insensitive loss which acts as a tube seperating the values.
# The primal version:
# min_w,b,slack,slack_star (1/2) ||w||^2 + C sigma(i = 1 to m) (slack_i + slack_i_star)
# such that y_i - (w.T * x_i + b) <= epsilon + slack_i
#           (w.T * x_i + b) - y_i <= epsilon + slack_i_star
#           slack_i, slack_i_star >= 0

# The dual form looks like this:
# max_alpha,alpha_star (-1/2) sigma(i = 1 to m) sigma(j = 1 to m) (alpha_i - alpha_i_star) * (alpha_j - alpha_j_star) * K(x_i, x_j) - epsilon * sigma(i = 1 to m) (alpha_i + alpha_i_star) + sigma(i = 1 to m) y_i * (alpha_i - alpha_i_star)
# such that sigma(i = 1 to m) (alpha_i - alpha_i_star) = 0, 0 <= alpha_i, alpha_i_star <= C

# The predictor or the decision function looks like:
# f(x) = sigma(i = 1 to m) (alpha_i - alpha_i_star) * K(x_i, x) + b

# Let's talk about C:
# C is a regularization hyperparameter that controls the trade off between:
# 1. margin width (||w||^2)
# 2. training error (slack penality via sigma(i = 1 to m) slack_i)

# Small C (eg. 0.01): Wider margin, more regularization, better generalization (less overfitting), more violations allowed, lower training accuracy.
# Only use when Noisy data, many outliers, prefer generaliztion
# Large C (eg. 1000): Higher training accuracy, tighter margin, less regularization, risk of overfitting, sensitive to outliers.
# Only use when you have clean data, confident in labels, need high accuracy.

# Let's start Linear SVM through SGD (Sub Gradient Descent) implementation:

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

class linear_svc:
    
    def __init__(self, C=0.01, learning_rate=0.01):
        self.C = C
        self.learning_rate = learning_rate
        self.loss_history = []
        
    def fit(self, X_train, y_train, iterations=10000, threshold=1e-6):
        X_train = X_train.copy()
        y_train = y_train.copy()
        
        # Step 1: Convert labels to -1, +1
        y_train[y_train == 0] = -1
        y_train[y_train == 1] = +1
        
        # Step 2: Initialize w and b parameters
        w = np.random.randn(X_train.shape[1])
        b = 0
        
        total_loss_old = -np.inf
        
        # Training loop
        for epoch in range(iterations):
            epoch_loss = 0
            
            # Step 3: For each iterations go over each data point
            for i in range(X_train.shape[0]):
                # Step 4: Compute margin for current sample with current w, b
                m_i = y_train[i] * (np.dot(w, X_train[i]) + b)
                
                # Step 5: 2 conditions m_i >= 1 and m_i < 1
                if m_i >= 1:
                    # No violation: only regularization gradient
                    gradient_w = w
                    gradient_b = 0
                    loss_i = 0  # No hinge loss
                else:
                    # Margin violation: regularization + hinge loss gradient
                    gradient_w = w - (self.C * (X_train[i] * y_train[i]))
                    gradient_b = -1 * self.C * y_train[i]
                    loss_i = 1 - m_i  # Hinge loss
                
                # Step 6: Update parameters w and b
                w = w - (self.learning_rate * gradient_w)
                b = b - (self.learning_rate * gradient_b)
                
                # Accumulate loss
                epoch_loss += loss_i
            
            # Compute total loss: regularization + hinge loss
            total_loss = 0.5 * np.dot(w, w) + self.C * epoch_loss
            self.loss_history.append(total_loss)
            
            # Step 7: Convergence checking
            if abs(total_loss_old - total_loss) < threshold:
                print(f"Converged at epoch {epoch}")
                break
            
            # Print progress every 1000 epochs or at the end
            if epoch % 100 == 0 or epoch == iterations - 1:
                print(f"Epoch {epoch}/{iterations} - Loss: {total_loss:.4f}")
                
            total_loss_old = total_loss
        
        self.w = w
        self.b = b
        
        return w, b
    
    def predict(self, X_test):
        """Predict class labels (-1 or +1)"""
        s = self.decision_function(X_test)
        return np.sign(s)
    
    def decision_function(self, X_test):
        """Compute decision function values (distance to hyperplane)"""
        return np.dot(X_test, self.w) + self.b
        
        
svc = linear_svc(C=10.0, learning_rate=0.01)
w, b = svc.fit(X_train, y_train, iterations=1000)

# Get predictions on test set
y_pred_test = svc.predict(X_test)

# Convert predictions back to 0, 1 for metrics
y_pred_test_binary = np.where(y_pred_test == -1, 0, 1)

print(f"\nFinal weights: {w}")
print(f"Final bias: {b}")

# Plotting and evaluation functions

def plot_loss_history(loss_history, filename='loss_history.png', output_dir='output_figs/svc_subgradient'):
    """
    Plot the loss history over epochs.
    
    Parameters:
    -----------
    loss_history : list
        List of loss values per epoch
    filename : str
        Output filename
    output_dir : str
        Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(loss_history)), loss_history, linewidth=2, color='blue')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (Hinge + Regularization)', fontsize=12)
    ax.set_title('Training Loss over Epochs', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Loss history plot saved to: {filepath}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png', output_dir='output_figs/svc_subgradient'):
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

# Generate plots
plot_loss_history(svc.loss_history)

# Compute and visualize support vectors on training data
y_train_signed = np.where(y_train == 0, -1, 1)
margins = y_train_signed * (X_train @ svc.w + svc.b)
n_support_vectors = np.sum(margins <= 1.0)

# SVM-specific plot: Margin analysis
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(margins, bins=50, edgecolor='black', alpha=0.7, color='purple')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Decision Boundary')
ax.axvline(x=1, color='orange', linestyle='--', linewidth=2, label='Margin (+1)')
ax.axvline(x=-1, color='orange', linestyle='--', linewidth=2, label='Margin (-1)')
ax.set_xlabel('Margin: y × (w·x + b)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'SVM Margin Distribution\nSupport Vectors (margin ≤ 1): {n_support_vectors} / {len(y_train_signed)} samples', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('output_figs/svc_subgradient', exist_ok=True)
plt.savefig('output_figs/svc_subgradient/svm_margins.png', dpi=150, bbox_inches='tight')
print(f"\nSVM margin plot saved to: output_figs/svc_subgradient/svm_margins.png")
plt.show()

plot_confusion_matrix(y_test, y_pred_test_binary)