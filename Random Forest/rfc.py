# The random forest just builds on top of the decision tree.
# So, we build multiple decision tree and do a voting to see which class is highest.

# So, how are we building the trees.
# The thing is that, we have a dataset, the tree is built using the dataset. Even slight
# changes to the dataset can lead to a completely different tree, which is what we are gonna do.

# We are gonna sample data from our training data with replacement to get multiple training datasets,
# which we then use to build the decision trees. The part where we sample data from our training data
# from our training data with replacement is called bootstrapping. Bootstrap sampling creates diversity,
# and averaging reduces variance.

# Why are we performing random feature selection at each split - It helps prevent strong predictors from 
# dominating all trees. Without it, trees would be too similar (highly correlated), reducing ensemble benefit.

# The pros are robust to overfitting, handles non-linear relationships, feature importance is built-in, 
# works with missing data, and minimal hyperparameter tuning.
# The cons are Less interpretable than single tree, slower prediction (must query B trees), larger memory footprint,
# and can struggle with extrapolation (that is the model would not perfectly comprehend a new case in future outside of the
# current data that we have).

# The problem with decision tree is the chance of overfitting: Low bias, high variance (overfits).
# So, the random forest averages and reduces variance without increasing the bias much.

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

# Now let's split the data into train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Printing shapes of data
print(f"X_train_shape: {X_train.shape}, y_train_shape: {y_train.shape}")
print(f"X_test_shape: {X_test.shape}, y_test_shape: {y_test.shape}\n\n")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Decision Tree'))
from dtc import DecisionTreeClassification

class RandomForestClassification:
    
    def __init__(self, n_trees = 50, max_depth = 50, max_features = None, min_sample_split = 5, impurity = "gini"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.impurity = impurity
        
        self.trees = []
        for i in range(self.n_trees):
            self.trees.append(
                DecisionTreeClassification(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_sample_split,
                    impurity_type=self.impurity
                )
            )
    
    def _get_random_subsets(self, X, y, n_subsets, replacement=True):
        n_samples = X.shape[0]
        
        X_y = np.column_stack((X, y))
        np.random.shuffle(X_y)
        subsets = []
        
        subsample_size = (n_samples) // 2
        if replacement:
            subsample_size = n_samples
            
        for i in range(n_subsets):
            idx = np.random.choice(
                range(n_samples),
                size=np.shape(range(subsample_size)),
                replace=replacement
            )
            
            X = X_y[idx][:, :-1]
            y = X_y[idx][:, -1]
            
            subsets.append([X, y])
        
        return subsets
            
    
    def fit(self, X_train, y_train):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        
        n_features = self.X_train.shape[1]
        
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))
        
        # Step 1: Get n_trees random subsets of data
        subsets = self._get_random_subsets(self.X_train, self.y_train, n_subsets=self.n_trees)
        
        for i in range(self.n_trees):
            X_subset, y_subset = subsets[i]
            
            # Step 2: Feature bagging (select random subsets of featurez)
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            
            # Step 3: Save the indices for future prediction
            self.trees[i].feature_indices = idx
            
            # Step 4: Choose the features correspoinding to the indices
            X_subset = X_subset[:, idx]
            
            # Step 5: Fit the tree
            self.trees[i].fit(X_subset, y_subset)
            
    
    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        
        for i, tree in enumerate(self.trees):
            idx = tree.feature_indices
            prediction = tree.predict(X[:, idx])
            y_preds[:, i] = prediction
            
        y_pred = []
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        
        return np.array(y_pred)
            

def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test,
                                     n_trees=50, max_depth=50, max_features=None,
                                     min_sample_split=5, impurity="gini",
                                     save_plots=True):
    """
    Train and evaluate a random forest classifier.
    
    Parameters:
    -----------
    X_train, y_train : Training data and labels
    X_test, y_test : Test data and labels
    n_trees : Number of trees in the forest
    max_depth : Maximum depth of each tree
    max_features : Number of features to consider at each split (default: sqrt(n_features))
    min_sample_split : Minimum samples required to split a node
    impurity : "gini" or "entropy"
    save_plots : Whether to save visualization plots
    
    Returns:
    --------
    random_forest : Trained RandomForestClassification object
    y_pred : Predictions on test set
    metrics : Dictionary containing accuracy, precision, recall, f1
    """
    
    # Train the random forest
    random_forest = RandomForestClassification(
        n_trees=n_trees,
        max_depth=max_depth,
        max_features=max_features,
        min_sample_split=min_sample_split,
        impurity=impurity
    )
    random_forest.fit(X_train, y_train)
    
    # Make predictions
    y_pred = random_forest.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    # Print metrics
    print("="*60)
    print("Random Forest Performance Metrics")
    print("="*60)
    print(f"Number of Trees: {n_trees}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60 + "\n")
    
    # Save plots if requested
    if save_plots:
        output_dir = "output_figs/random_forest_classifier"
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=np.unique(y_train), yticklabels=np.unique(y_train),
               xlabel='Predicted Label',
               ylabel='True Label',
               title='Confusion Matrix - Random Forest Classifier')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Compare with Single Decision Tree
        from dtc import DecisionTreeClassification
        single_tree = DecisionTreeClassification(max_depth=max_depth, 
                                                 min_samples_split=min_sample_split,
                                                 impurity_type=impurity)
        single_tree.fit(X_train, y_train)
        y_pred_tree = single_tree.predict(X_test)
        
        tree_accuracy = accuracy_score(y_test, y_pred_tree)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        models = ['Single Decision Tree', 'Random Forest']
        accuracies = [tree_accuracy, accuracy]
        colors = ['#ff7f0e', '#2ca02c']
        
        bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Random Forest vs Single Decision Tree Performance', fontsize=14)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rf_vs_tree_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Number of Trees vs Accuracy
        tree_counts = [1, 5, 10, 20, 30, 40, 50]
        accuracies_by_trees = []
        
        for n in tree_counts:
            rf_temp = RandomForestClassification(n_trees=n, max_depth=max_depth,
                                                 max_features=max_features,
                                                 min_sample_split=min_sample_split,
                                                 impurity=impurity)
            rf_temp.fit(X_train, y_train)
            y_pred_temp = rf_temp.predict(X_test)
            acc_temp = accuracy_score(y_test, y_pred_temp)
            accuracies_by_trees.append(acc_temp)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(tree_counts, accuracies_by_trees, marker='o', linewidth=2, 
                markersize=8, color='#2ca02c')
        ax.set_xlabel('Number of Trees', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Random Forest Accuracy vs Number of Trees', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/trees_vs_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved in '{output_dir}/' directory:")
        print(f"  - confusion_matrix.png")
        print(f"  - rf_vs_tree_comparison.png")
        print(f"  - trees_vs_accuracy.png\n")
    
    return random_forest, y_pred, metrics


# Main execution
if __name__ == "__main__":
    rf_model, predictions, metrics = train_and_evaluate_random_forest(
        X_train, y_train, X_test, y_test,
        n_trees=50,
        max_depth=10,
        max_features=None,  # Will default to sqrt(n_features)
        min_sample_split=5,
        impurity="gini",
        save_plots=True
    )
            