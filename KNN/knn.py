# So, KNN can work both as a classification and as a regression algorithm for the
# supervised tasks.

# So, what is KNN - K-Nearest Neighbours.
# There is no specific training that is required, because KNN just looks at the:
# close points in the input space should have similar outputs.

# For this we want a Distance equation and there are many choices for this:
# 1. Euclidean - Most commonly used
# 2. Manhattan
# 3. Minkowski
# 4. and a lot more

# So, the algorithm looks like this:
# 1. For each training point i = 1, ..., m, compute the distance:
#   d_i = D(x_query, x^(i))
# 2. Sort the training points by distance d_i.
# 3. Take indices of the k nearest neighbours:
#   N_k(x_query) = {i_1, i_2, ..., i_k}
#   Such that, d_i1 <= d_i2 <= .... <= d_ik
# 4. Do classification or regression using their y^(i) values.

# This is the core algorithm:
# Now let's see classification and regression and how it works:
# 1. Classification
#   We have y_i = {1, 2, ...., C}
#   For a query x we find its k neighbours:
#   N_k(x) = {i_1, ..., i_k}
# 1.1. Unweighted majority vote (classic KNN)
#   1. For each class C
#       1. Count how many of the k neighbours have label C:
#           n_c(x) = sigma (i belongs to N_k(x)) 1{y_i = c}
#           1 is the indicator (1 is true, 0 is false)
#       2. Choose the class with the highest count:
#           y_hat = argmax (c belongs to {1, ..., C}) n_c(x)
#       So, the prediction is simply the most frequent class among the k-neighbours.
#       Also, can be intercepted as:
#           P_hat(y = c | x) = n_c(x) / k
# 1.2 Distane Weighted KNN classification
#   1. The same happens except now, we want closer neighbours to have more influence.
#   2. The weight for each neighbour for example might look like this:
#       w_i = 1 / (d(x, x_i)^p + e), where p is just a power, e is just to save 0 division.
#   3. You just multiply the w_i where we have the indicator function, to count k neighbours having label c:
#       n_c(x) = sigma (i belongs to N_k(x)) w_i * 1{y_i = c}
#   4. The y_hat remains the same.
#   5. The prbability estimate changes:
#       P_hat(y = c  | x) = n_c(x) / sigma (c_dash) n_c_dash(x)
#   c is just one class, but c_dash represents all the classes. c_dash in the sigma runs over all the classes.
# 2. Regression
#   Now, we have target values as y_i belongs to R.
#   Same neighbour set: N_k(x) = {i_1, ..., i_k}
# 1.1 Unweighted KNN for regression
#   KNN regression
#       y_hat(x) = (1 / k) * sigma(i belongs to N_k(x))
#       It is to find the K Nearest target values, and average them.
# 1.2 Distance Weighted KNN for regression
#   You weigh in nearer points more.
#   Given weights w_i >= 0, for i belongs N_k(x).
#       w_i = 1 / (d(x, x_i)^p + e), p is some power, e to save us from 0 divsion.
#   Then, we normalize weights,
#       w_i_bar = w_i / (sigma (j belongs to N_k(x)) w_j)
#   Predicted value
#       y_hat(x) = sigma (i belongs to N_k(x)) w_i_bar * y_i
#   This is the weighted average of neighbour targets.

# What happens when k = 1, weighted and unweighted versions become identical.
# k is a hyperparameter.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(42)

X, y = make_blobs(n_samples=1000, n_features=5, centers=3, random_state=42) # type: ignore

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

class knn_unweighted_classification:
    # K - Nearest Neighbours for Unweighted Classification
    
    def __init__(self, k):
        self.k = k
        
    def fit(self, X_train, y_train):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        return self
    
    def predict(self, X_test):
        
        y_hat = []
        for i in range(X_test.shape[0]):
            # Step 1: Compute distance to all training points
            dis_tt = np.sqrt(np.sum((self.X_train - X_test[i]) ** 2, axis=1))
            
            # Step 2: Compute K Nearest Neighbours indexes
            k_idx = np.argpartition(dis_tt, self.k, axis=0)[:self.k]
            
            # Step 3: Find the labels for the respective indexes in y_train
            labels_idx = self.y_train[k_idx]
            
            # Step 4: Pick the highest count from labels_idx
            count_highest_label = np.bincount(labels_idx)
            count_highest_label = np.argmax(count_highest_label)
            
            # Step 5: Append the highest count label to y_hat
            y_hat.append(count_highest_label)
            
        return y_hat
    
class knn_weighted_classification:
    
    def __init__(self, k, p=2, epsilon=1e-6):
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
            # Step 1: Calculate the distance to all training points
            dis_tt = np.sqrt(np.sum((self.X_train - X_test[i]) ** 2, axis=1))
            
            # Step 2: Find the indexes k nearest neighbours
            k_idx = np.argpartition(dis_tt, self.k, axis=0)[:self.k]
            
            # Step 3: Extract the distances of only those k neighbours
            dis_tt = dis_tt[k_idx]
            
            # Step 4: Calculate the weights only for those k neighbours
            w = 1 / (np.power(dis_tt, self.p) + self.epsilon)
            
            # Step 5: Weighted voting
            labels_idx = self.y_train[k_idx]
            wv = np.bincount(labels_idx, weights=w) # I didn't know this, awesom.
            
            # Step 6: Argmax
            label = np.argmax(wv)
            
            y_hat.append(label)
            
        return y_hat
        
knn_uc = knn_unweighted_classification(5)
output_dir = "output_figs/knn_weighted_classification" if type(knn_uc) == knn_weighted_classification else "output_figs/knn_unweighted_classification"
knn_uc.fit(X_train, y_train)
y_hat = knn_uc.predict(X_test)
tested_y = np.vstack([["y", "y_hat"], np.column_stack((y_test, y_hat))])

print("Y and Y_hat Comparison: \n", tested_y[:6])

def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png', output_dir='output_figs/knn_unweighted_classification'):
    """
    Plot confusion matrix for multi-class classification.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    filename : str
        Output filename
    output_dir : str
        Directory to save the plot
    """
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    # Calculate overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Calculate per-class metrics
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # Macro-averaged metrics
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
    
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels([f'Pred {c}' for c in classes])
    ax.set_yticklabels([f'True {c}' for c in classes])
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix ({n_classes} classes)\nAcc: {accuracy:.3f} | Prec: {avg_precision:.3f} | Rec: {avg_recall:.3f} | F1: {avg_f1:.3f}', 
                 fontsize=14)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nClassification Metrics (Macro-averaged):")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1-Score:  {avg_f1:.4f}")
    print(f"\nPer-class metrics:")
    for i, c in enumerate(classes):
        print(f"  Class {c}: Prec={precisions[i]:.4f}, Rec={recalls[i]:.4f}, F1={f1s[i]:.4f}")
    print(f"\nConfusion matrix plot saved to: {filepath}")
    plt.show()
    
# Plot results
# Remove header row and extract columns
y_true = tested_y[1:, 0].astype(float).astype(int)
y_pred = tested_y[1:, 1].astype(float).astype(int)
plot_confusion_matrix(y_true, y_pred, output_dir=output_dir)

