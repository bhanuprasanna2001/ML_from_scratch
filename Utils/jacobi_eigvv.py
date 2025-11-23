# Let's implement jacobi method to find eigen values and eigen vectors.

# Let's talk about eigen values and eigen vectors:
# A non-zero column vector v is called the eigen vector of a Matrix A with the
# eigen value lambda, if A @ v = lambda * v.

# A important difference that is to be learnt:
# Dense matrix: is a type of matrix where most of elements in the matrix are 
# non-zero elements.
# Sparse matrix: is a type of matrix where most of the elements in the matrix
# are zero elements.


import numpy as np
np.random.seed(42)

# First we intialize epsilon
eps = 0.000001

# A = np.random.randint(low=-2, high=3, size=(3,4))

# sym_A = A @ A.T


# Step 1: Find the largest Non-Diagonal Element of the symmetric matrix
def largest_non_diagonal_element(A):
    # For as to find the largest off-diagnoal element
    # we need to check for ij where i != j
    # This function returns the max value and it's co-ordinates
    
    max = -1 * np.inf
    cord = (0, 0)
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i != j:
                if np.abs(A[i][j]) > max:
                    max = np.abs(A[i][j])
                    cord = (i, j)
                    
    return max, cord
                


# print(f"Original Matrix: \n{sym_A}")
# max_A, cord_A = largest_non_diagonal_element(sym_A)
# print(f"For sym_A matrix max: {max_A}, the co-ordinates: {cord_A}")
# print(f"Alternate same value: {sym_A[cord_A[1]][cord_A[0]]}, the co-ordinates: {cord_A[::-1]}")

# It doesn't matter if we take i, j or j, i because both are same only because
# it is a symmetric matrix.

# Step 2: Find theta
def find_theta(A, i, j):
    if A[i, i] == A[j, j]:
        return (np.pi / 4) * np.sign(A[i, j])
        
    return 0.5 * np.arctan((2 * A[i][j]) / (A[i][i] - A[j][j]))

# theta = find_theta(sym_A, cord_A[0], cord_A[1])
# print("Theta value: ", theta)

# Step 3: Find Orthogonal Matrix
# Place the rotation matrix at the cord_A that has been obtained
# Like for (0, 2), we have rotation matrix [[cos theta, -sin theta], [sin theta, cos theta]]
# So we substitute in sym_A where:
# 00 will be cos theta
# 02 will be -sin theta
# 20 will be sin theta
# 22 will be cos theta
def apply_rotation_to_sym(A, i, j, theta):
    S = np.identity(A.shape[0])
    
    S[i][i] = np.cos(theta)
    S[i][j] = -1 * np.sin(theta)
    S[j][i] = np.sin(theta)
    S[j][j] = np.cos(theta)
    
    B = S.T @ A @ S
    
    return B, S

# print(apply_rotation_to_sym(sym_A, cord_A[0], cord_A[1], theta))

def jacobi_eigenvalue(A, max_iter=1000, tol=1e-10):
    # Finds jacobi eigen values and eigen vectors
    
    n = A.shape[0]
    A = A.copy()
    V = np.eye(n)
    
    for iteration in range(max_iter):
        # 1. Find largest off-diagonal element (p, q)
        max_val, (p, q) = largest_non_diagonal_element(A)
        
        # 2. Check Convergence
        # So other than diagonal elements rest all should be close to Tolerance.
        if np.abs(max_val) < tol:
            print(f"Converged in {iteration} iterations")
            break # EXIT LOOP
        
        # 3. Compute Rotation Angle theta
        theta = find_theta(A, p, q)
        
        # 4. Apply rotation to A
        A, S = apply_rotation_to_sym(A, p, q, theta)
        
        # 5. Update V the eigen vectors, A has already been updated.
        V = V @ S
        
    # Extract eigen values from diagonal
    eigenvalues = np.diag(A)
    
    return eigenvalues, V
        
# eig_values, eig_vectors = jacobi_eigenvalue(sym_A)

# print(eig_values)
# print(eig_vectors)

# print(np.linalg.eig(sym_A))

# I have successfully implemented the eigenvalues and eigen vectors.
# But haven't learnt the intiution of Jacobi iterate iterate iterate methodology.