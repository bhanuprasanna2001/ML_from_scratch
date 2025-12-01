# This script implements the decision tree classification from scratch.
# The main concept for decision tree is to split the tree into binary cases.
# Each node has 2 branches. There is only 1 root node.
# The final nodes are called the leaf nodes which is our classification output.

# The main concepts here are Information gain, Entropy, and Gini index.
# Gini Index, Entropy, and Information Gain are all measures of impurity or disorder in
# decision tree algorithms to decide how to split a node.

# The entropy is a measure of uncertainity or randomness in a node. A perfectly pure node
# (all data points belong to one class) has an entropy of 0, while a node with an even distribution
# of all classes has the highest entropy.
# Entropy(S) = - sigma(i = 1 to c) p_i log_2 (p_i), 
# where c is the number of classes, and p_i is the proportion of examples in class i.

# The gini index is another measure of impurity or disorder. It calculate the probability of 
# incorrectly picking a random element if it were randomly labeled according to the distribution
# of labels in the subset. A gini index of 0 means perfect purity, while gini index of 1 means
# all elements are impure.
# Gini(S) = 1 - sigma(i = 1 to c) p_i^2,
# where c is the number of classes, and p_i is the proportion of examples in class i.

# Information gain - the reduction in impurity achieved by splitting a dataset on a particular
# attribute. It is the difference between the impurity of the parent node and the weighted
# average of the impurities of the child nodes.
# Information_Gain(S) = Entropy(S) - sigma(j = 1 to v) (|S_j|/|S|) * Entropy(S_j)
# Information_Gain(S) = Gini(S) - sigma(j = 1 to v) (|S_j|/|S|) * Gini(S_j)
# Decision tree algorithms choose the attribute that yields the highes information gain for the
# next split, as this is the most effective at separating the data into pure subsets.

# v is specific to a feature, not all features.
# Let's take an example, I have a dataset for house prices, we have the 5 features:
# Area, Location, number of bedroom, washrooms, and living room area.
# The goal is to predict the price would be for this 5 features. Now, v is not 5 because
# v is specific to a feature. For example, the Area (SQFT), so the v can be >2000 and <=2000 (representing v = 2).
# The decision tree splits it binary. This is the CART (Classification and Regression Trees). But there seems to
# be others like C4.5, ID3, which allow multi-way splits. The sklearn only implements the CART with different criterons
# entropy, gini, and log-loss. gini is faster because no computation of log.
# Gini(S) means root node gini, Gini(S_j) means gini is specific to a feature.

# The nodes are just simple logical gates. They contain only the information necessary to sort a data point into the correct 
# child node and to record the statistics about the data subset that ended up there. 

# What is |S_j| and |S|, it is just the cardinality.

# |S| (pronounced "the cardinality of S"): This is the total count of all data points (rows) in the current parent node before the split.
# |S_j| (pronounced "the cardinality of S sub j"): This is the count of data points that go down the \(j\)-th branch after the split. 
# The ratio |S_{j}| / |S|} is a simple fraction or probability, always between 0 and 1.

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


class Node:
    
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, info_gain = None, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

    
class DecisionTreeClassification:
    
    def __init__(self, max_depth=50, min_samples_split=5, impurity_type="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_type = impurity_type
        
    def _entropy(self, p_k):
        H_S = -1 * np.sum(p_k * np.log2(p_k))
        return H_S
    
    def _gini(self, p_k):
        G_S = 1 - np.sum(p_k ** 2)
        return G_S
    
    def _build_tree(self, S, depth):
        
        n, d = len(S), self.X_train.shape[1]
        unique_classes, n_k = np.unique_counts(self.y_train[S])
        p_k = n_k / n
        majority_class = unique_classes[np.argmax(n_k)]
        
        if self.impurity_type == "gini":
            I = self._gini(p_k)
        elif self.impurity_type == "entropy":
            I = self._entropy(p_k)
            
        if I == 0:
            leaf_node = Node(value=majority_class)
            return leaf_node
        
        if depth >= self.max_depth or n < self.min_samples_split:
            leaf_node = Node(value=majority_class)
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
                
                n_l = S_left.size
                n_r = S_right.size
                
                if n_l == 0 or n_r == 0:
                    continue
                
                n_j_left = np.unique_counts(self.y_train[S_left])[1]
                n_j_right = np.unique_counts(self.y_train[S_right])[1]
                
                p_j_left = n_j_left / n_l
                p_j_right = n_j_right / n_r
                
                if self.impurity_type == "gini":
                    I_left = self._gini(p_j_left)
                    I_right = self._gini(p_j_right)
                elif self.impurity_type == "entropy":
                    I_left = self._entropy(p_j_left)
                    I_right = self._entropy(p_j_right)
                    
                IG = I - ((n_l / n) * I_left) - ((n_r / n) * I_right)
                
                if IG > best_info_gain:
                    best_info_gain = IG
                    best_feature_index = j
                    best_threshold = threshold
                    best_left_subset = S_left
                    best_right_subset = S_right
        
        
        if best_info_gain <= 0:
            leaf_node = Node(value=majority_class)
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


def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test, 
                                     max_depth=50, min_samples_split=5, 
                                     impurity_type="gini", 
                                     print_tree=True, 
                                     save_plots=True):
    """
    Train and evaluate a decision tree classifier.
    
    Parameters:
    -----------
    X_train, y_train : Training data and labels
    X_test, y_test : Test data and labels
    max_depth : Maximum depth of the tree
    min_samples_split : Minimum samples required to split a node
    impurity_type : "gini" or "entropy"
    print_tree : Whether to print the tree structure
    save_plots : Whether to save visualization plots
    
    Returns:
    --------
    decision_tree : Trained DecisionTree object
    y_pred : Predictions on test set
    metrics : Dictionary containing accuracy, precision, recall, f1
    """
    
    # Train the decision tree
    decision_tree = DecisionTreeClassification(max_depth=max_depth, 
                                 min_samples_split=min_samples_split, 
                                 impurity_type=impurity_type)
    decision_tree.fit(X_train, y_train)
    
    # Print tree structure if requested
    if print_tree:
        decision_tree.print_tree()
    
    # Make predictions
    y_pred = decision_tree.predict(X_test)
    
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
    print("Performance Metrics")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60 + "\n")
    
    # Save plots if requested
    if save_plots:
        output_dir = "output_figs/decision_tree_classifier"
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
               title='Confusion Matrix - Decision Tree Classifier')
        
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
        
        # Plot 2: Decision Tree Visualization using graphviz for YOUR custom tree
        import graphviz
        
        def add_nodes_edges(tree, dot, node_id=0, parent_id=None, edge_label="", S=None):
            """
            Recursively add nodes and edges to graphviz graph for custom tree.
            Returns the next available node_id.
            """
            if tree is None:
                return node_id
            
            current_id = node_id
            
            # Get node statistics from the training data
            if S is not None:
                y_subset = decision_tree.y_train[list(S)]
                unique_classes, counts = np.unique(y_subset, return_counts=True)
                n_samples = len(S)
                
                # Calculate class distribution
                class_dist = [0] * len(np.unique(decision_tree.y_train))
                for cls, cnt in zip(unique_classes, counts):
                    class_dist[int(cls)] = cnt
                
                # Calculate impurity
                p_k = counts / n_samples
                if decision_tree.impurity_type == "gini":
                    impurity = 1 - np.sum(p_k ** 2)
                else:  # entropy
                    impurity = -np.sum(p_k * np.log2(p_k + 1e-10))
            else:
                n_samples = "?"
                class_dist = "?"
                impurity = "?"
            
            if tree.value is not None:
                # Leaf node
                label = f"gini = {impurity:.3f}\\nsamples = {n_samples}\\nvalue = {class_dist}\\nclass = Class {tree.value}"
                dot.node(str(current_id), label, shape='box', style='filled', fillcolor='lightblue')
            else:
                # Internal node
                label = f"Feature {tree.feature_index} <= {tree.threshold:.3f}\\ngini = {impurity:.3f}\\nsamples = {n_samples}\\nvalue = {class_dist}\\nclass = Class {np.argmax(class_dist) if isinstance(class_dist, list) else '?'}"
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
        dot = graphviz.Digraph(comment='Custom Decision Tree')
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
        print(f"  - confusion_matrix.png")
        print(f"  - tree_visualization.png\n")
    
    return decision_tree, y_pred, metrics


# Main execution
if __name__ == "__main__":
    tree, predictions, metrics = train_and_evaluate_decision_tree(
        X_train, y_train, X_test, y_test,
        max_depth=50,
        min_samples_split=5,
        impurity_type="gini",
        print_tree=False,
        save_plots=True
    )

