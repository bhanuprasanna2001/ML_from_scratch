# So, I want to implement Naive Bayes from scratch.
# I will be using the random dataset created from sklearn for this.

# For suppose we currently have 2 classes namely Spam vs Not Spam.
# So for this, we first compute the probabilities:
# P(Y = Spam | X) = some value and P(Y = Not Spam | X) = some value,
# but the sum of these 2 probabilities is 1, because there are only 2 classes.

# So, the thing is that, we have Bayes theorem here where it has the structure:
# P(y = k | x) = (P(x | y = k) * P(y = k)) / P(x)
# But we ignore P(x) because it doesn't change at all, so, we approximate like:
# P(y = k | x) ∝ P(x | y = k) * P(y = k)

# But let's break down the terms:
# 1. Posterior: P(y = k | x) = Probability of class k given features x (what we want)
# 2. Likelihood: P(x | y = k) = Probability of seeing features x in class k
# 3. Prior: P(y = k) = probability of class k (before seeing any features)
# 4. Evidence: P(x) = probability of features x (across all classes)

# P(x) is the same for all classes, so for comparison we can ignore it: leading us to:
# P(y = k | x) ∝ P(x | y = k) * P(y = k)

# So, calculating the P(x | y = k) is slightly hard

# The P(x | y = k) = product (i = 1 to n) weird_N(x_i, mu_ik, sigma_i)
# so, the log(P(y = k | x)) = argmax(log(P(y = k)) + sigma (i = 1 to n) log(weird_N(x_i, mu_ik, sigma_i)))
# so y_hat = log(P(y = k | x))

# There is this Naive assumption where we assume features are conditionally independent given the class.
# But usually the Naive assumption is almost always wrong.
# In real data, features are almost never conditionally independent.
# The reason for which Naive Bayes still works:
# 1. We only need ranking, not exact probabilities:
#   y_hat = argmax_k P(y = k | x)
# 2. Errors can cancel out - Two features are +vely correlated in both classes.
#   If class 0 and class 1 joint probability is overestimated by similar amounts, the ratio
#   stays approximately correct.
# 3. High-Dimensional Spaces and Strong signal vs Weak correlation.

# So, at the end there are only 3 parameters that is sent out of the gaussian naive bayes.
# pi_c = P(y = c) where c is the class
# mu_cj = mean of class c for each feature j
# sigma^2_cj = variance of class c for each feature j

from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(42)

X, y = make_classification(n_samples=1000,
                           n_features=5,
                           n_informative=3,
                           n_redundant=1,
                           n_classes=2,
                           random_state=42)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

def naive_bayes(X, y):
    # Gaussian Naive Bayes
    
    X = X.copy()
    y = y.copy()
    
    X_shape = X.shape
    
    class_storage = dict()
    
    unique_y = np.unique(y)
    
    for c in unique_y:
        subset_y_c = X[y == c]
        m_c = subset_y_c.shape[0]
        pi_c = m_c / X_shape[0]
        u_c = np.mean(subset_y_c, axis=0)
        var_c = np.var(subset_y_c, axis=0)
        
        class_storage[c] = {}
        
        class_storage[c]['pi'] = pi_c
        class_storage[c]['mu'] = u_c
        class_storage[c]['var'] = var_c
        
    return class_storage
    

class_info_params = naive_bayes(X_train, y_train)

# Now we have got all the params that we require

def evaluate(X, y, params):
    # Evaluate Gaussian Naive Bayes
    
    X = X.copy()
    X_shape = X.shape
    
    s_c_dict = dict()
    
    for c in params.keys():
        pi_c = params[c]['pi']
        u_c = params[c]['mu']
        var_c = params[c]['var']
        
        s_c = np.log((1 / np.sqrt(2 * np.pi * var_c)) * np.power(np.e, (-1 / 2) * (((X - u_c) ** 2) / var_c)))
        s_c = np.log(params[c]['pi']) + np.sum(s_c, axis=1)
        
        s_c_dict[c] = s_c
        
    y_hat = np.column_stack([s_c_dict[i] for i in s_c_dict.keys()])
    y_hat = np.vstack(y_hat.argmax(axis=1))
    
    return np.vstack([["y", "y_hat"], np.column_stack((y, y_hat))])
    
tested_y = evaluate(X_test, y_test, class_info_params)

print(f"Y and Y_hat comparison:\n{tested_y[:6]}")

def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png', output_dir='output_figs/naive_bayes'):
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
plot_confusion_matrix(y_true, y_pred)
        
# But we can extend this to Bernoulli and Multinomial cases as well.
# Currently we have data like where the features have numerical values like size, length.
# But bernoulli, has only 0 or 1 for every feature column. (Binary labels for features itself), the classes or output is different it can be many right.
# But multinomial, has not only 0 or 1 but also 2 or 3 or 4 or ... n as well for every feature column. The output is different and can n classes.