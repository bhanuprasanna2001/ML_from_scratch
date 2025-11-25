# Now let's learn K - Means from scratch.

# So, K-Means is a clustering algorithm which comes under Unsupervised rather than Supervised.
# So, we only have the Data X which has m data points where each data point is represented as x_i.

# So, the main objective of K-Means is to form K clusters where the points inside it have the lowest Cost J.
# The cost is nothing but the sum of Squared Euclidean Distance of the all the data points in a cluster with respect to the center of the cluster.

# So, there can be more clusters, so we have to consider how are we actually implementing this.
# So, mainly we have 4 steps in this process:
# 1. First we randomly take K points from the Data X to be our Center points for our K Clusters.
# 2. Then, Assignment operation starts, where we assign each data point x_i to it's respective kth cluster based on the calculated distance.
# 3. This completes the assignment, now we have best assignments, now the center can be different,
# so, we actually compute the mean of all the points inside the cluster to find the new center for the cluster k.
# 4. Then, if we calculate the cost J, if it converges and is less than our threshold then we can say the model has successfully been fit.
# 5. The steps 1, 2, 3, 4 are repeated again and again for a certain number of iterations or until it converges.

# So, let's look at how the math looks like:
# So, we have data points x_i where i = 1 to m, and u_k where k = 1 to K.
# We also have r_ik which is just an indicator function that tell if the ith data point is present in kth cluster, yes 1 or no 0.
# Also we have u_k which is just the cluster centers.
# The square euclidean distance = (x_i - u_k) ^ 2
# The cost function J({r_ik}, {u_k}) = sigma(i = 1 to m) sigma(k = 1 to K) r_ik * (x_i - u_k) ^ 2

# There are mainly 2 steps that we need to know about:
# 1. Assignment Step (Given Centers u_k, Find best assignments r_ik)
#   We have to choose r_ik for each i,
#       r_ik belongs to {0,1}
#       sigma(k = 1 to K) r_ik = 1
#   Repeat until convergence (or for a fixed number of iterations)
#       # Assignment Step 
#       For each point x_i:
#           Compute distances to each centroid: (x_i - u_k) ^ 2 for all k.
#           Assign the point to the closest centroid
#               r_ik = 1 if argmin_j (x_i - u_j) ^ 2, r_ij = 0 for j != k
#       # Update Step
#       For each cluster k:
#           Recompute centroid as the mean of its assigned points
#           u_k = (sigma(i = 1 to m) r_ik * x_i) / (sigma(i = 1 to m) r_ik) = (1 / n_k) * sigma(i = 1 to m, r_ik = 1) x_i
#       # Convergence
#       Stop when the assignments no longer change (or centroids move less than a certain tiny threshold
#       or after some max iterations). The algorithm converges to a local minimum of J.

# So, let's start the implementation

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(42)

X, y = make_blobs(n_samples=1000, n_features=5, centers=3, random_state=42) # type: ignore

# So, we only want X, because this is a Unsupervised learning.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

class kmeans:
    
    def __init__(self, k=3, iterations=1000, threshold=1e-3):
        self.k = k
        self.iterations = iterations
        self.threshold = threshold
        
    def fit(self, X_train):
        
        self.X_train = X_train.copy()
        X_shape = self.X_train.shape
        
        # Step 1: Choose k random points to be center points for k clusters
        k_points = np.random.choice(X_shape[0], self.k)
        k_points = self.X_train[k_points]
        
        # Cost Intialization
        J_old = np.inf
        
        for i in range(self.iterations):
            # Step 2: Compute distances from each data point to the k center points
            distance_list = np.array([])
            for k in range(self.k):
                dis_dc = np.vstack(np.sum(np.power((self.X_train - k_points[k]), 2), axis=1))
                if distance_list.size == 0:
                    distance_list = dis_dc
                else:
                    distance_list = np.column_stack((distance_list, dis_dc))
            
            # Step 3: Assign each data point to it's closest centroid
            closest_point = np.argmin(distance_list, axis=1)
            
            # Step 4: Compute the center for each cluster
            k_points_new = np.empty((0, X_shape[1]))
            for k in range(self.k):
                k_count = np.count_nonzero(closest_point == k)

                if k_count == 0:
                    u_k = self.X_train[np.random.choice(X_shape[0])]
                else:
                    idxs_k = np.where(closest_point == k)[0]
                    u_k = np.mean(self.X_train[idxs_k], axis=0)
                
                k_points_new = np.vstack([k_points_new, u_k])
            
            k_points = k_points_new
            
            # Step 5: Compute cost and check Convergence
            J_new = 0
            for k in range(self.k):
                idxs_k = np.where(closest_point == k)[0]
                dis_dc = np.sum(np.power((self.X_train[idxs_k] - k_points[k]), 2), axis=1)
                J_new += np.sum(dis_dc)
            
            if np.abs((J_new - J_old)) < self.threshold:
                print(f"Converged at Iteration {i}.")
                break
            
            if i == self.iterations - 1:
                print("Algorithm execution completed before converging!")
            
            J_old = J_new
                
        self.k_points = k_points
        return self.k_points
            
            
    def predict(self, X_test):
        
        X_test = np.array(X_test)
        distance_list = np.array([])
        for k in range(self.k):
            dis_dc = np.vstack(np.sum(np.power((X_test - self.k_points[k]), 2), axis=1))
            if distance_list.size == 0:
                distance_list = dis_dc
            else:
                distance_list = np.column_stack((distance_list, dis_dc))
        
        cluster_labels = np.argmin(distance_list, axis=1)
        
        return cluster_labels


def calculate_clustering_metrics(X, labels, centroids):
    """Calculate clustering metrics: inertia and silhouette score."""
    from sklearn.metrics import silhouette_score
    
    # Inertia (within-cluster sum of squares)
    inertia = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroids[k]) ** 2)
    
    # Silhouette score (only if we have more than 1 cluster and less than n_samples)
    if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(X):
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = 0.0
    
    return inertia, silhouette


def plot_clustering_results(X_train, X_test, train_labels, test_labels, centroids, 
                           true_labels_test=None, output_dir='output_figs/kmeans'):
    """
    Plot clustering results with 2D visualization using first 2 features.
    """
    
    # Calculate metrics
    inertia_train, silhouette_train = calculate_clustering_metrics(X_train, train_labels, centroids)
    inertia_test, silhouette_test = calculate_clustering_metrics(X_test, test_labels, centroids)
    
    # For visualization, use first 2 features
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training data clusters
    ax1 = axes[0]
    scatter1 = ax1.scatter(X_train[:, 0], X_train[:, 1], c=train_labels, 
                          cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', 
               s=200, edgecolors='black', linewidth=2, label='Centroids')
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title('Training Data Clusters', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Cluster')
    
    # Plot 2: Test data clusters
    ax2 = axes[1]
    scatter2 = ax2.scatter(X_test[:, 0], X_test[:, 1], c=test_labels, 
                          cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', 
               s=200, edgecolors='black', linewidth=2, label='Centroids')
    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.set_title('Test Data Clusters', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'clustering_results.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    
    # Print metrics
    print(f"\nK-Means Clustering Metrics:")
    print(f"Number of clusters: {len(centroids)}")
    print(f"\nTraining Set:")
    print(f"  Inertia:          {inertia_train:.4f}")
    print(f"  Silhouette Score: {silhouette_train:.4f}")
    print(f"\nTest Set:")
    print(f"  Inertia:          {inertia_test:.4f}")
    print(f"  Silhouette Score: {silhouette_test:.4f}")
    
    # If true labels available, calculate accuracy
    if true_labels_test is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(true_labels_test, test_labels)
        nmi = normalized_mutual_info_score(true_labels_test, test_labels)
        print(f"\nClustering Quality (vs True Labels):")
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"  Normalized Mutual Info: {nmi:.4f}")
    
    print(f"\nPlot saved to: {filepath}")
    plt.show()
    
    return {'inertia_train': inertia_train, 'silhouette_train': silhouette_train,
            'inertia_test': inertia_test, 'silhouette_test': silhouette_test}


# Run K-Means
km = kmeans(k=3)
k_points = km.fit(X_train)
train_labels = km.predict(X_train)
test_labels = km.predict(X_test)

# Plot results
metrics = plot_clustering_results(X_train, X_test, train_labels, test_labels, 
                                  k_points, true_labels_test=y_test)