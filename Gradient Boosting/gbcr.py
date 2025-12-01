# The Gradient Boosting algorithm is a iterative algorithm, where
# you first start with the weakest learner, then learn gradually to become
# a good learner.

# Actually the Gradient Boosting algorithm with classification and regression
# is the same thing, just the loss function varies, that's it, so we will build both the
# regression and classification version here itself.

# As the name suggests, we are going to be dealing with Gradients. So, whatever
# loss function that we are going to be dealing with must be differentiable.

# Mainly, we have a loss function L(y, y_hat). Where y is true predicitons and y_hat 
# are the predicted values.

# So, we intialize the y_hat to have the average of all targets and be same size as y. Then take 
# gradient wrt to our loss function. Then, we fit our good old Decision Tree Regression and then 
# predict, to replace the y_hat with the new y_hat. Now, we want to get as close as possible to 
# the y so, we have the learning rate, we do this iteratively until it converges or reaches the
# number of iterations.

# We use the Decision Tree Regression for both the classification and regression task.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression

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
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Decision Tree"))
from dtr import DecisionTreeRegressor

# Loss functions - For Regression SSE and for Classification - CrossEntropy

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()
    
    def gradient(self, y_true, y_pred):
        return NotImplementedError()
    
    def accuracy(self, y_true, y_pred):
        return 0
    
class SquaredError(Loss):
    def loss(self, y_true, y_pred):
        return (0.5) * (np.sum(np.power((y_true - y_pred), 2)))
    
    def gradient(self, y_true, y_pred):
        return -1 * (y_true - y_pred)
    
class CrossEntropy(Loss):
    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (-1 * y_true * np.log(y_pred)) - ((1 - y_true) * np.log(1 - y_pred))
    
    def gradient(self, y_true, y_pred):
        p = 1 / (1 + np.exp(-y_pred))
        return p - y_true

class GradientBoosting:
    
    def __init__(self, n_trees = 50, max_depth = 50, min_samples_split = 5, learning_rate = 0.01, regression = True, impurity = "variance"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.impurity = impurity
        self.regression = regression
        
        self.loss_func = SquaredError()
        if not self.regression:
            self.loss_func = CrossEntropy()
            
        self.trees = []
        for i in range(self.n_trees):
            self.trees.append(
                DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    impurity_type=self.impurity
                )
            )
            
    
    def fit(self, X_train, y_train):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        
        pos_ratio = np.mean(self.y_train)
        self.init_logit = np.log(pos_ratio / (1 - pos_ratio))
        y_pred = np.full(self.y_train.shape, self.init_logit)
        if self.regression:
            self.init_logit = pos_ratio
            y_pred = np.full(self.y_train.shape, pos_ratio)
        
        for i in range(self.n_trees):
            gradient = self.loss_func.gradient(self.y_train, y_pred)
            self.trees[i].fit(self.X_train, gradient)
            update = self.trees[i].predict(self.X_train)
            y_pred = y_pred - np.multiply(self.learning_rate, update)
            
            
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_logit)
        
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = y_pred - update
            
        if not self.regression:
            probabilities = 1 / (1 + np.exp(-y_pred))
            y_pred = (probabilities >= 0.5).astype(int)
            
        return y_pred
            


def train_and_evaluate_gradient_boosting_classifier(X_train, y_train, X_test, y_test,
                                                     n_trees=50, max_depth=5, min_samples_split=5,
                                                     learning_rate=0.1, impurity="variance",
                                                     save_plots=True):
    """
    Train and evaluate a gradient boosting classifier.
    
    Parameters:
    -----------
    X_train, y_train : Training data and labels
    X_test, y_test : Test data and labels
    n_trees : Number of boosting iterations
    max_depth : Maximum depth of each tree (keep shallow for boosting!)
    min_samples_split : Minimum samples required to split a node
    learning_rate : Learning rate (0.01 - 0.3 typical)
    impurity : "variance" or "sse" (for regression trees)
    save_plots : Whether to save visualization plots
    
    Returns:
    --------
    gradient_boost : Trained GradientBoosting object
    y_pred : Predictions on test set
    metrics : Dictionary containing accuracy, precision, recall, f1
    """
    
    # Train gradient boosting classifier
    gradient_boost = GradientBoosting(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        learning_rate=learning_rate,
        regression=False,
        impurity=impurity
    )
    gradient_boost.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gradient_boost.predict(X_test)
    
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
    print("Gradient Boosting Classifier Performance Metrics")
    print("="*60)
    print(f"Number of Trees: {n_trees}")
    print(f"Learning Rate:   {learning_rate}")
    print(f"Max Depth:       {max_depth}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60 + "\n")
    
    # Save plots if requested
    if save_plots:
        output_dir = "output_figs/gradient_boosting_classifier"
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
               title='Confusion Matrix - Gradient Boosting Classifier')
        
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
        
        # Plot 2: Compare with single Decision Tree
        from dtc import DecisionTreeClassification
        single_tree = DecisionTreeClassification(max_depth=max_depth,
                                                 min_samples_split=min_samples_split,
                                                 impurity_type="gini")
        single_tree.fit(X_train, y_train)
        y_pred_tree = single_tree.predict(X_test)
        tree_accuracy = accuracy_score(y_test, y_pred_tree)
        
        # Compare with Random Forest
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Random Forest'))
        from rfc import RandomForestClassification
        rf = RandomForestClassification(n_trees=n_trees, max_depth=max_depth,
                                       min_sample_split=min_samples_split)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        models = ['Single Tree', 'Random Forest', 'Gradient Boosting']
        accuracies = [tree_accuracy, rf_accuracy, accuracy]
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        
        bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Comparison: Gradient Boosting vs Ensemble Methods', fontsize=14)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Learning Curve (Accuracy vs Number of Trees)
        tree_counts = [1, 5, 10, 20, 30, 40, 50]
        train_accuracies = []
        test_accuracies = []
        
        for n in tree_counts:
            gb_temp = GradientBoosting(n_trees=n, max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       learning_rate=learning_rate,
                                       regression=False, impurity=impurity)
            gb_temp.fit(X_train, y_train)
            
            y_pred_train = gb_temp.predict(X_train)
            y_pred_test = gb_temp.predict(X_test)
            
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(tree_counts, train_accuracies, marker='o', linewidth=2,
                markersize=8, label='Training Accuracy', color='#2ca02c')
        ax.plot(tree_counts, test_accuracies, marker='s', linewidth=2,
                markersize=8, label='Test Accuracy', color='#d62728')
        ax.set_xlabel('Number of Trees (Boosting Iterations)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Learning Curve: Accuracy vs Number of Trees', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/learning_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Learning Rate Impact
        learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
        lr_accuracies = []
        
        for lr in learning_rates:
            gb_temp = GradientBoosting(n_trees=50, max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       learning_rate=lr,
                                       regression=False, impurity=impurity)
            gb_temp.fit(X_train, y_train)
            y_pred_temp = gb_temp.predict(X_test)
            lr_accuracies.append(accuracy_score(y_test, y_pred_temp))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(learning_rates, lr_accuracies, marker='D', linewidth=2,
                markersize=10, color='#9467bd')
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Impact of Learning Rate on Performance', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/learning_rate_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved in '{output_dir}/' directory:")
        print(f"  - confusion_matrix.png")
        print(f"  - model_comparison.png")
        print(f"  - learning_curve.png")
        print(f"  - learning_rate_impact.png\n")
    
    return gradient_boost, y_pred, metrics


def train_and_evaluate_gradient_boosting_regressor(X_train, y_train, X_test, y_test,
                                                   n_trees=50, max_depth=5, min_samples_split=5,
                                                   learning_rate=0.1, impurity="variance",
                                                   save_plots=True):
    """
    Train and evaluate a gradient boosting regressor.
    
    Parameters:
    -----------
    X_train, y_train : Training data and target values
    X_test, y_test : Test data and target values
    n_trees : Number of boosting iterations
    max_depth : Maximum depth of each tree (keep shallow for boosting!)
    min_samples_split : Minimum samples required to split a node
    learning_rate : Learning rate (0.01 - 0.3 typical)
    impurity : "variance" or "sse"
    save_plots : Whether to save visualization plots
    
    Returns:
    --------
    gradient_boost : Trained GradientBoosting object
    y_pred : Predictions on test set
    metrics : Dictionary containing R², RMSE, MAE
    """
    
    # Train gradient boosting regressor
    gradient_boost = GradientBoosting(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        learning_rate=learning_rate,
        regression=True,
        impurity=impurity
    )
    gradient_boost.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gradient_boost.predict(X_test)
    
    # Calculate regression metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }
    
    # Print metrics
    print("="*60)
    print("Gradient Boosting Regressor Performance Metrics")
    print("="*60)
    print(f"Number of Trees: {n_trees}")
    print(f"Learning Rate:   {learning_rate}")
    print(f"Max Depth:       {max_depth}")
    print(f"R² Score:  {r2:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAE:       {mae:.4f}")
    print("="*60 + "\n")
    
    # Save plots if requested
    if save_plots:
        output_dir = "output_figs/gradient_boosting_regressor"
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Predicted vs Actual
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Gradient Boosting Regressor: Predicted vs Actual', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/predicted_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Residuals
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/residuals.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Learning Curve
        tree_counts = [1, 5, 10, 20, 30, 40, 50]
        r2_by_trees = []
        rmse_by_trees = []
        
        for n in tree_counts:
            gb_temp = GradientBoosting(n_trees=n, max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       learning_rate=learning_rate,
                                       regression=True, impurity=impurity)
            gb_temp.fit(X_train, y_train)
            y_pred_temp = gb_temp.predict(X_test)
            r2_by_trees.append(r2_score(y_test, y_pred_temp))
            rmse_by_trees.append(np.sqrt(mean_squared_error(y_test, y_pred_temp)))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # R² vs Trees
        axes[0].plot(tree_counts, r2_by_trees, marker='o', linewidth=2,
                    markersize=8, color='#2ca02c')
        axes[0].set_xlabel('Number of Trees', fontsize=12)
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('R² Score vs Number of Trees', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # RMSE vs Trees
        axes[1].plot(tree_counts, rmse_by_trees, marker='s', linewidth=2,
                    markersize=8, color='#d62728')
        axes[1].set_xlabel('Number of Trees', fontsize=12)
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_title('RMSE vs Number of Trees', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/learning_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved in '{output_dir}/' directory:")
        print(f"  - predicted_vs_actual.png")
        print(f"  - residuals.png")
        print(f"  - learning_curve.png\n")
    
    return gradient_boost, y_pred, metrics


# Main execution
if __name__ == "__main__":
    # Classification example
    gb_classifier, predictions, metrics = train_and_evaluate_gradient_boosting_classifier(
        X_train, y_train, X_test, y_test,
        n_trees=50,
        max_depth=3,  # Shallow trees for boosting!
        min_samples_split=5,
        learning_rate=0.1,
        impurity="variance",
        save_plots=True
    )
    
    np.random.seed(42)

    X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42) # type: ignore
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    gb_classifier, predictions, metrics = train_and_evaluate_gradient_boosting_regressor(
        X_train, y_train, X_test, y_test,
        n_trees=50,
        max_depth=3,
        min_samples_split=5,
        learning_rate=0.1,
        impurity="variance",
        save_plots=True
    )
            
        
        