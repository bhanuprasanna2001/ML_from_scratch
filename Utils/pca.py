# Now it's time to build PCA from scratch to understand,
# I have built and have completed the Jacobi implementation of calculating
# the eigen values and eigen vectors.

# It seems the following steps:
# 1. We calculate the center or mean of the data set that we have, like
# for each column which is a feature, we will find the mean.
# 2. Then, we will subtract the mean of each feature by itself, so that we
# get a mean of 0, which is a requirement for PCA, but why:
#   So, the thing is that subtracting ensures that no single feature dominates
#   the analysis due to it's scale, making all features contribute equally.
#   Also, we don't care about the position (where the data is) of the data in the space, but only
#   care about it's shape or how it spreads.
# 3. Then, we find the Covariance matrix, it allows to find the how do the features
# spread out and relate to each other.
# 4. Then, we find the eigen values and the eigen vectors for the covariance matrix.
# 5. Then, order the eigen vectors according to the eigen values.
# 6. We select the top k eigen vectors and then transform the data.
# 7. Z = X_centered_0 @ V_k
# 8. The k is the n_components and it cannot be larger the number of eigen vectors.

# The difference between SVD and PCA:
# 1. The SVD tries to find the best rank-k approximaiton that minimized reconstruction error.
# 2. The PCA tries to find k directions that maximize captured variance.

from jacobi_eigvv import *
from sklearn.datasets import make_blobs
import numpy as np
np.random.seed(42)

X, y = make_blobs(n_samples=1000, n_features=5, centers=3, random_state=42) # type: ignore
print(X.shape)

def pca(X, n_components=5):
    # Principal Component Analysis
    
    # Check if n_components is greater than the number of eigen vectors
    if n_components > X.shape[1]:
        print(f"PCA Not possible the n_components must be less than {X.shape[1]}.")
        return np.array([])
    
    # Copying X so that we don't change global X
    X = X.copy()
    
    # Step 1: Computing Shape and Mean
    X_shape = X.shape
    X_mean = X.mean(axis=0)
    
    # Step 2: Subtracting the mean from the data
    X = (X - X_mean)
    
    # Step 3: Find the covariance matrix of the data
    X_cov = np.cov(X.T)
    
    # Step 4: Use our pre-built jacobi Eigen values funciton to 
    # compute the eigen values and eigen vectors of the cov matrix
    eig_values, eig_vectors = jacobi_eigenvalue(X_cov)
    
    # Step 5: Order the eigen values properly
    idx_eigval = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx_eigval]
    eig_vectors = eig_vectors[:, idx_eigval]
    
    print(f"Eigen Values: \n{eig_values}\n")
    print(f"Eigen Vectors: \n{eig_vectors}\n")
    
    # Step 6: Select top k eigen vectors based on n_components
    V_k = eig_vectors[:, :n_components]
    
    # Step 7: Compute Z = X_centered_0 @ V_k
    Z = X @ V_k
    
    return Z
    
Z = pca(X, n_components=2)

print(Z.shape)