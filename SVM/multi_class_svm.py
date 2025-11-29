import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

np.random.seed(42)

X, y = make_classification(n_samples=1000,
                           n_features=5,
                           n_informative=4,
                           n_redundant=1,
                           n_classes=5,
                           random_state=42)

# Printing shapes of data
print(f"X_shape: {X.shape}, y_shape: {y.shape}")

# Now let's split the data into train and test data

from sklearn.model_selection import train_test_split

from svc_cvxopt import svc_cvxopt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

class multiclass_svm_0vr:
    
    def __init__(self, C = 1.0, gamma = 0.1):
        self.C = C
        self.gamma = gamma
        self.classifiers = []
        
    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        
        self.classes = np.unique(y)
        
        for cls in self.classes:
            print(f"\nTraining classifier for class {cls}...")
            y_binary = np.where(self.y == cls, 1, -1)
            
            # Debug: check label distribution
            n_positive = np.sum(y_binary == 1)
            n_negative = np.sum(y_binary == -1)
            print(f"  Class {cls}: {n_positive} positive, {n_negative} negative samples")
            
            # Train binary SVM with CVXOPT
            svm = svc_cvxopt(C=self.C, gamma=self.gamma)
            svm.fit(self.X, y_binary)
            self.classifiers.append(svm)
            
    def predict(self, X):
        # Get scores from all classifiers
        scores = np.array([clf.decision_function(X) for clf in self.classifiers])
        # scores shape: (n_classes, n_samples)
        # Pick class with max score for each sample
        return self.classes[np.argmax(scores, axis=0)]
            
        
        
svc_multiclass = multiclass_svm_0vr(C=1.0, gamma=0.1)
svc_multiclass.fit(X_train, y_train)

# Test predictions
y_pred = svc_multiclass.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)

# Confusion matrix
n_classes = len(svc_multiclass.classes)
confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
for true_label, pred_label in zip(y_test, y_pred):
    confusion_matrix[true_label, pred_label] += 1

# Per-class metrics
print(f"\n{'='*60}")
print("MULTI-CLASS TEST METRICS (One-vs-Rest)")
print(f"{'='*60}")
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Number of test samples: {len(y_test)}")
print(f"\nPer-class metrics:")
for cls in svc_multiclass.classes:
    tp = confusion_matrix[cls, cls]
    fp = np.sum(confusion_matrix[:, cls]) - tp
    fn = np.sum(confusion_matrix[cls, :]) - tp
    tn = np.sum(confusion_matrix) - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Class {cls}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

print(f"\nConfusion Matrix:")
print(f"     Predicted →")
print(f"True ↓  ", end="")
for cls in svc_multiclass.classes:
    print(f"{cls:4d}", end="")
print()
for i, cls in enumerate(svc_multiclass.classes):
    print(f"  {cls}    ", end="")
    for j in range(n_classes):
        print(f"{confusion_matrix[i,j]:4d}", end="")
    print()

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')

# Add colorbar
plt.colorbar(im, ax=ax)

# Labels and ticks
ax.set_xticks(np.arange(n_classes))
ax.set_yticks(np.arange(n_classes))
ax.set_xticklabels(svc_multiclass.classes)
ax.set_yticklabels(svc_multiclass.classes)

# Add text annotations
for i in range(n_classes):
    for j in range(n_classes):
        text = ax.text(j, i, confusion_matrix[i, j],
                      ha="center", va="center",
                      color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black",
                      fontsize=14)

ax.set_xlabel('Predicted Class', fontsize=12)
ax.set_ylabel('True Class', fontsize=12)
ax.set_title(f'Multi-class SVM Confusion Matrix (One-vs-Rest)\nAccuracy: {accuracy:.3f} | {n_classes} Classes | CVXOPT', 
             fontsize=14)
plt.tight_layout()

os.makedirs('output_figs/multiclass_svm_ovr', exist_ok=True)
plt.savefig('output_figs/multiclass_svm_ovr/confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix plot saved to: output_figs/multiclass_svm_ovr/confusion_matrix.png")
plt.show()

print(f"\n{'='*60}")