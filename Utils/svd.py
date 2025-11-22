# Now I want to learn SVD of a matrix and use it to compress Images.

# So, I have coded this, but I want to write what I have learnt about
# implementing SVD. I know I have used numpy, but SVD and Spectral Decomposition
# are 2 special algorithms which mainly moves around Eigen Values and Eigen Vectors.
# Just knowing that eigen vectors are vectors when a linear transformation is 
# applied it just scales the eigen vectors than doing anything else, like
# v is a eigen vector, A @ v = lambda * v, so the lambda here is the eigen value.

# Then according to traditional we have to do a lot:
# 1. Find u = A @ A.T and then do u - lambda @ I = 0 to find lambda which are
# our eigen values for u.
# 2. Then, find eigen vectors for u through gaussian elimination for each lambda like
# (u - lambda_i @ I) @ eig_v_i = 0, we solve for this and find eigen vectors for U.
# 3. Find eigen vectors for V the same way, but we have to A.T @ A for this.
# 4. After finding the eigen values and U and V, the singular matrix sigma in b/w
# looks like a diagonal matrix of size m x n, where U = m x m, and V = n x n
# both are symmetric matrices, use np.eye(m, n) and then ingest for every i b/w
# 0, m, inject the one eigen value after the other, then we get the sigma as well.
# 5. Then, we have the 3 matrices which make up the SVD, A = U @ Sigma @ V.T.

# That is SVD, but in code, just using NumPy, we can do through to first finding
# Eigen values and Eigen vectors for both A @ A.T and A.T @ A, and then sort by indexes
# and then properly build build the singular matrix as well.

# But it seems there are a lot of mathematical checks that must be done to have a robust working,
# because once I implemented I got all negative A, rather than just A.

# So, we have to do a special thing for U where column vectors of U is U_i = 1 / sigma_i (A @ V_i),
# where V_i is the i th column vector of V.

# For images, it is just the Rank-k Approximation to perform Image Compression.

import numpy as np

np.random.seed(42)

A = np.random.randint(low=0, high=5, size=(3,4))

print(A, "\n")

eig_a_at = np.linalg.eig(A @ A.T) # Represents the U
eig_at_a = np.linalg.eig(A.T @ A) # Represents the V

idx_u = np.argsort(eig_a_at.eigenvalues)[::-1]
idx_v = np.argsort(eig_at_a.eigenvalues)[::-1]

print(idx_u, "\n", idx_v, "\n")

sorted_eigenvalues_u = eig_a_at.eigenvalues[idx_u]
sorted_eigenvalues_v = eig_at_a.eigenvalues[idx_v]

print(sorted_eigenvalues_u, "\n", sorted_eigenvalues_v, "\n")

sorted_eigenvectors_u = eig_a_at.eigenvectors[:, idx_u]
sorted_eigenvectors_v = eig_at_a.eigenvectors[:, idx_v]

sigma_mat = np.eye(3,4)

for i in range(3):
    sigma_mat[i][i] = sorted_eigenvalues_u[i] ** 0.5

V = sorted_eigenvectors_v

U = np.zeros_like(sorted_eigenvectors_u)
for i in range(3):
    U[:, i] = (A @ V[:, i]) / sigma_mat[i, i]

print(U)
print(V)
print(sigma_mat, "\n")

sigma_0 = sigma_mat[0,0]
u_0 = U[:, 0]
v_0 = V[:, 0]

rank1_approx = sigma_0 * np.outer(u_0, v_0)

print("Rank - 1 approximation: \n", rank1_approx)
print("Original A: \n", A, "\n")

print((U @ sigma_mat @ V.T).astype(np.int32))


# So why does the Rank-k approximation actually work in Image Compression.
# Full SVD = sigma_1 * u_1 @ v_1.T + sigma_2 * u_2 @ v_2.T + .... + sigma_r * u_r @ v_r.T
# So, for Rank-k approximation: A_k = sigma_1 * u_1 @ v_1.T + .... + sigma_k * u_k @ v_k.T
# Keeps only the first k terms.

# Why this works:
# 1. Singular values are sorted descending: sigma_1 >= sigma_2 >= sigma_3 >= ....
# 2. The first few singular values are much larger than later ones.
# 3. They capture the most important patterns in the data.
# 4. Later terms add tiny details.

# For images:
# 1. Keeps just k = 50 singular values instead of 1000.
# 2. Storage: (mxk + k + k*n) instead of (mxn) - A = U[:, :k] @ Sigma[:k, :k] @ V[:, :k].T
# 3. Compression ratio: huge for large images.

# Now let's do image compression using SVD:

from sklearn.datasets import load_sample_image
img = load_sample_image('flower.jpg')
gray_img = np.mean(img, axis=2).astype(np.uint8)

print(f"\n\nShape: {gray_img.shape}")

U, sigma, V = np.linalg.svd(gray_img)

# The max k that we can reach is the min(m, n).

print(f"U: {U.shape}, V: {V.shape}, Sigma: {sigma.shape}\n")

# So, here the V is actually the transposed V (V.T) rather than just V.

k = np.min(A.shape)

k_vals = [2, 5, 10, 50, 100, 150, 300, 400, 427]

from PIL import Image

for i in k_vals:
    sigma_i = np.diag(sigma[:i])
    A_k = U[:, :i] @ sigma_i @ V[:i, :]
    A_k_uint8 = np.clip(A_k, 0, 255).astype(np.uint8)
    img = Image.fromarray(A_k_uint8)
    img_save_str = f"output_figs/svd_image_compression/Rank_{i}.png"
    img.save(img_save_str)
    print(f"Image saved as Rank_{i}.png")

print("\n")

# Now we have just done on 2D without any channels like RGB,
# I will try to do the RGB version as well.

img_rgb = load_sample_image('flower.jpg')


for k in k_vals:
    channels_compressed = []

    for c in range(3):
        channel = img_rgb[:, :, c]
        U, S, V = np.linalg.svd(channel)
        
        channel_k = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
        channels_compressed.append(channel_k)
        
    img_compressed = np.stack(channels_compressed, axis=2)
    img_compressed = np.clip(img_compressed, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_compressed)
    img_save_str = f"output_figs/svd_image_compression/Rank_rgb_{k}.png"
    img.save(img_save_str)
    print(f"Image saved as Rank_rgb_{k}.png")