# Now, let's implement random forest regression

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

np.random.seed(42)

X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42) # type: ignore

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Printing shapes of data
print(f"X_train_shape: {X_train.shape}, y_train_shape: {y_train.shape}")
print(f"X_test_shape: {X_test.shape}, y_test_shape: {y_test.shape}\n\n")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Decision Tree"))
from dtr import DecisionTreeRegressor

class RandomForestRegressor:
    
    def __init__(self, n_trees = 50, max_depth = 50, max_features = None, min_samples_split = 5, impurity = "variance"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.impurity = impurity
        
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
        
        n_features = self.X_train.shape[1]
        
        if self.max_features == None:
            if n_features <= 10:
                self.max_features = int(np.sqrt(n_features))
            else:
                self.max_features = n_features // 3
        
        from sklearn.utils import resample
        for i in range(self.n_trees):
            X_subset, y_subset = resample(self.X_train, self.y_train, n_samples=self.X_train.shape[0], stratify=None, replace=True)
            
            idx = np.random.choice(range(n_features), size=self.max_features, replace=False)
            
            self.trees[i].feature_indices = idx
            
            X_subset = np.array(X_subset[:, idx])
            y_subset = np.array(y_subset)
            
            self.trees[i].fit(X_subset, y_subset)
            
    
    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        
        for i, tree in enumerate(self.trees):
            idx = tree.feature_indices
            prediction = tree.predict(X[:, idx])
            y_preds[:, i] = prediction
        
        y_pred = []
        for sample_predictions in y_preds:
            y_pred.append(np.mean(sample_predictions))
            
        return np.array(y_pred)
        


def train_and_evaluate_random_forest_regressor(X_train, y_train, X_test, y_test,
                                               n_trees=50, max_depth=50, max_features=None,
                                               min_samples_split=5, impurity="variance",
                                               save_plots=True):
    """
    Train and evaluate a random forest regressor.
    
    Parameters:
    -----------
    X_train, y_train : Training data and target values
    X_test, y_test : Test data and target values
    n_trees : Number of trees in the forest
    max_depth : Maximum depth of each tree
    max_features : Number of features to consider at each split (default: sqrt(n_features))
    min_samples_split : Minimum samples required to split a node
    impurity : "variance" or "sse"
    save_plots : Whether to save visualization plots
    
    Returns:
    --------
    random_forest : Trained RandomForestRegressor object
    y_pred : Predictions on test set
    metrics : Dictionary containing R², RMSE, MAE
    """
    
    # Train the random forest regressor
    random_forest = RandomForestRegressor(
        n_trees=n_trees,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        impurity=impurity
    )
    random_forest.fit(X_train, y_train)
    
    # Make predictions
    y_pred = random_forest.predict(X_test)
    
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
    print("Random Forest Regressor Performance Metrics")
    print("="*60)
    print(f"Number of Trees: {n_trees}")
    print(f"R² Score:  {r2:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAE:       {mae:.4f}")
    print("="*60 + "\n")
    
    # Save plots if requested
    if save_plots:
        output_dir = "output_figs/random_forest_regressor"
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
        ax.set_title('Random Forest Regressor: Predicted vs Actual', fontsize=14)
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
        
        # Plot 3: Compare with Single Decision Tree
        from dtr import DecisionTreeRegressor
        single_tree = DecisionTreeRegressor(max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           impurity_type=impurity)
        single_tree.fit(X_train, y_train)
        y_pred_tree = single_tree.predict(X_test)
        
        tree_r2 = r2_score(y_test, y_pred_tree)
        tree_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tree))
        
        # R² comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # R² Score comparison
        models = ['Single Decision Tree', 'Random Forest']
        r2_scores = [tree_r2, r2]
        colors = ['#ff7f0e', '#2ca02c']
        
        bars1 = axes[0].bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('R² Score Comparison', fontsize=14)
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # RMSE comparison
        rmse_values = [tree_rmse, rmse]
        bars2 = axes[1].bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_title('RMSE Comparison (Lower is Better)', fontsize=14)
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rf_vs_tree_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Number of Trees vs R² Score
        tree_counts = [1, 5, 10, 20, 30, 40, 50]
        r2_by_trees = []
        rmse_by_trees = []
        
        for n in tree_counts:
            rf_temp = RandomForestRegressor(n_trees=n, max_depth=max_depth,
                                           max_features=max_features,
                                           min_samples_split=min_samples_split,
                                           impurity=impurity)
            rf_temp.fit(X_train, y_train)
            y_pred_temp = rf_temp.predict(X_test)
            r2_temp = r2_score(y_test, y_pred_temp)
            rmse_temp = np.sqrt(mean_squared_error(y_test, y_pred_temp))
            r2_by_trees.append(r2_temp)
            rmse_by_trees.append(rmse_temp)
        
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
        plt.savefig(f"{output_dir}/trees_vs_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved in '{output_dir}/' directory:")
        print(f"  - predicted_vs_actual.png")
        print(f"  - residuals.png")
        print(f"  - rf_vs_tree_comparison.png")
        print(f"  - trees_vs_performance.png\n")
    
    return random_forest, y_pred, metrics


# Main execution
if __name__ == "__main__":
    rf_model, predictions, metrics = train_and_evaluate_random_forest_regressor(
        X_train, y_train, X_test, y_test,
        n_trees=50,
        max_depth=10,
        max_features=None,  # Will default to sqrt(n_features)
        min_samples_split=10,
        impurity="variance",
        save_plots=True
    )
        