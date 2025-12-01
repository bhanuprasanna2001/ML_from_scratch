# Now I will implement decision tree regression. There is already the classification
# version of decision is implemented.

# There is not a lot of difference between the classification and regression version.

# Instead of class labels we have values.
# Instead of gini/entropy we have Variance or Sum of Squared Errors (SSE) both use Mean.
# At leaf node you store the mean of y_i, not class.

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


class Node:
    
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, info_gain = None, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value
        
        
class DecisionTreeRegressor:
    
    def __init__(self, max_depth=50, min_samples_split=5, impurity_type="variance"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_type = impurity_type
        
        
    def _build_tree(self, S, depth):
        
        n, d = len(S), self.X_train.shape[1]
        u_s = np.mean(self.y_train[S])
        
        if self.impurity_type == "variance":
            I = np.var(self.y_train[S])
        elif self.impurity_type == "sse":
            I = np.var(self.y_train[S]) * n
            
        if I == 0:
            leaf_node = Node(value=u_s)
            return leaf_node
        
        if depth >= self.max_depth or n < self.min_samples_split:
            leaf_node = Node(value=u_s)
            return leaf_node
        
        best_info_gain = -np.inf
        best_feature_index = None
        best_threshold = None
        best_left_subset = None
        best_right_subset = None
        
        for j in range(d):
            
            feature_values = self.X_train[list(S), j]
            
            unique_values = np.unique(feature_values)
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i+1]) / 2
                
                left_mask = feature_values <= threshold
                
                S_left = S[left_mask]
                S_right = S[~left_mask]
                
                if self.impurity_type == "variance":
                    var_left = (np.var(self.y_train[S_left]) * (S_left.size / n))
                    var_right = (np.var(self.y_train[S_right]) * (S_right.size / n))
                    IG = I - var_left - var_right
                elif self.impurity_type == "sse":
                    sse_left = np.var(self.y_train[S_left]) * S_left.size
                    sse_right = np.var(self.y_train[S_right]) * S_right.size
                    IG = I - sse_left - sse_right
                    
                if IG > best_info_gain:
                    best_info_gain = IG
                    best_feature_index = j
                    best_threshold = threshold
                    best_left_subset = S_left
                    best_right_subset = S_right    
                    
        if best_info_gain <= 0:
            leaf_node = Node(value=u_s)
            return leaf_node
        
        if best_info_gain > 0:
            left_child = self._build_tree(best_left_subset, depth+1)
            right_child = self._build_tree(best_right_subset, depth+1)
            
            internal_node = Node(
                feature_index=best_feature_index,
                threshold=best_threshold,
                left=left_child,
                right=right_child,
                info_gain=best_info_gain,
                value=None
            )
        
        return internal_node
        
        
    def fit(self, X_train, y_train):
        
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        
        S = np.arange(self.X_train.shape[0])
        self.root_node = self._build_tree(S, 0)
        
    
    def predict(self, X):
        
        test_samples = X.shape[0]
        y_pred = []
        
        for i in range(test_samples):
            
            current_node = self.root_node
            
            while current_node.value == None: # type: ignore
                j = current_node.feature_index
                t = current_node.threshold
                
                if X[i][j] <= t:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            
            y_pred.append(current_node.value) # type: ignore
            
        return np.array(y_pred)
    
    
    def print_tree(self, tree=None, prefix="", is_root_call=True):
        """ Recursively print the decision tree in an elegant format """
        if not tree:
            tree = self.root_node
        
        # Print header only on first call
        if is_root_call:
            print("\n" + "="*60)
            print("Decision Tree Structure")
            print("="*60 + "\n")

        # If we're at leaf => print the prediction
        if tree.value is not None:
            print(f"{prefix}└── Predict: Class {tree.value}")
        # Internal node with split
        else:
            if prefix == "":
                # Root node
                print(f"Feature {tree.feature_index} <= {tree.threshold:.4f} (IG={tree.info_gain:.4f})")
            else:
                print(f"{prefix}Feature {tree.feature_index} <= {tree.threshold:.4f} (IG={tree.info_gain:.4f})")
            
            # Print left subtree (True branch - values <= threshold)
            if tree.left:
                left_prefix = prefix + "├── [True]  "
                child_prefix = prefix + "│           "
                print(left_prefix)
                self.print_tree(tree.left, child_prefix, is_root_call=False)
            
            # Print right subtree (False branch - values > threshold)
            if tree.right:
                right_prefix = prefix + "└── [False] "
                child_prefix = prefix + "            "
                print(right_prefix)
                self.print_tree(tree.right, child_prefix, is_root_call=False)
                
        if is_root_call:
            print("\n\n")

def train_and_evaluate_decision_tree_regressor(X_train, y_train, X_test, y_test, 
                                               max_depth=50, min_samples_split=5, 
                                               impurity_type="variance", 
                                               print_tree=True, 
                                               save_plots=True):
    """
    Train and evaluate a decision tree regressor.
    
    Parameters:
    -----------
    X_train, y_train : Training data and target values
    X_test, y_test : Test data and target values
    max_depth : Maximum depth of the tree
    min_samples_split : Minimum samples required to split a node
    impurity_type : "variance" or "sse"
    print_tree : Whether to print the tree structure
    save_plots : Whether to save visualization plots
    
    Returns:
    --------
    decision_tree : Trained DecisionTreeRegressor object
    y_pred : Predictions on test set
    metrics : Dictionary containing R², RMSE, MAE
    """
    
    # Train the decision tree regressor
    decision_tree = DecisionTreeRegressor(max_depth=max_depth, 
                                          min_samples_split=min_samples_split, 
                                          impurity_type=impurity_type)
    decision_tree.fit(X_train, y_train)
    
    # Print tree structure if requested
    if print_tree:
        decision_tree.print_tree()
    
    # Make predictions
    y_pred = decision_tree.predict(X_test)
    
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
    print("Performance Metrics")
    print("="*60)
    print(f"R² Score:  {r2:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAE:       {mae:.4f}")
    print("="*60 + "\n")
    
    # Save plots if requested
    if save_plots:
        output_dir = "output_figs/decision_tree_regressor"
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
        ax.set_title('Decision Tree Regressor: Predicted vs Actual', fontsize=14)
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
        
        # Plot 3: Decision Tree Visualization using graphviz
        import graphviz
        
        def add_nodes_edges(tree, dot, node_id=0, parent_id=None, edge_label="", S=None):
            """
            Recursively add nodes and edges to graphviz graph for regression tree.
            Returns the next available node_id.
            """
            if tree is None:
                return node_id
            
            current_id = node_id
            
            # Get node statistics from the training data
            if S is not None:
                y_subset = decision_tree.y_train[list(S)]
                n_samples = len(S)
                mean_val = np.mean(y_subset)
                
                # Calculate impurity (variance or SSE)
                if decision_tree.impurity_type == "variance":
                    impurity = np.var(y_subset)
                else:  # sse
                    impurity = np.var(y_subset) * n_samples
            else:
                n_samples = "?"
                mean_val = "?"
                impurity = "?"
            
            if tree.value is not None:
                # Leaf node
                label = f"{decision_tree.impurity_type} = {impurity:.3f}\\nsamples = {n_samples}\\nmean = {mean_val:.3f}\\nPrediction: {tree.value:.3f}"
                dot.node(str(current_id), label, shape='box', style='filled', fillcolor='lightblue')
            else:
                # Internal node
                label = f"Feature {tree.feature_index} <= {tree.threshold:.3f}\\n{decision_tree.impurity_type} = {impurity:.3f}\\nsamples = {n_samples}\\nmean = {mean_val:.3f}"
                dot.node(str(current_id), label, shape='box', style='filled', fillcolor='lightsalmon')
            
            # Add edge from parent if exists
            if parent_id is not None:
                dot.edge(str(parent_id), str(current_id), label=edge_label)
            
            # Recursively add children with updated index sets
            next_id = current_id + 1
            if tree.left is not None:
                # Calculate left subset indices
                if S is not None and tree.feature_index is not None:
                    feature_values = decision_tree.X_train[list(S), tree.feature_index]
                    left_mask = feature_values <= tree.threshold
                    S_left = S[left_mask]
                else:
                    S_left = None
                next_id = add_nodes_edges(tree.left, dot, next_id, current_id, "True", S_left)
            
            if tree.right is not None:
                # Calculate right subset indices
                if S is not None and tree.feature_index is not None:
                    feature_values = decision_tree.X_train[list(S), tree.feature_index]
                    right_mask = feature_values > tree.threshold
                    S_right = S[right_mask]
                else:
                    S_right = None
                next_id = add_nodes_edges(tree.right, dot, next_id, current_id, "False", S_right)
            
            return next_id
        
        # Create graphviz graph
        dot = graphviz.Digraph(comment='Custom Decision Tree Regressor')
        dot.attr(rankdir='TB')  # Top to Bottom layout
        dot.attr('node', fontname='Arial', fontsize='10')
        dot.attr('edge', fontname='Arial', fontsize='9')
        
        # Build the graph from YOUR custom tree, starting with all training indices
        S_root = np.arange(len(decision_tree.X_train))
        add_nodes_edges(decision_tree.root_node, dot, S=S_root)
        
        # Save as PNG
        dot.format = 'png'
        dot.render(f"{output_dir}/tree_visualization", cleanup=True)
        
        print(f"Plots saved in '{output_dir}/' directory:")
        print(f"  - predicted_vs_actual.png")
        print(f"  - residuals.png")
        print(f"  - tree_visualization.png\n")
    
    return decision_tree, y_pred, metrics


# Main execution
if __name__ == "__main__":
    tree, predictions, metrics = train_and_evaluate_decision_tree_regressor(
        X_train, y_train, X_test, y_test,
        max_depth=10,
        min_samples_split=10,
        impurity_type="variance",
        print_tree=False,
        save_plots=True
    )

