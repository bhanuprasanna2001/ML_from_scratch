# So, I want to build my own matrix multiplication.
# Obviously not as robust as numpy but I will try.

import numpy as np

def matrix_multiplication(x, y):
    final_mat = []
    
    if x.ndim != 2 or y.ndim != 2:
        return -1
    
    if x.shape[-1] != y.shape[0]:
        return -1
    
    for i in range(x.shape[0]):
        sum_mat = []
        for j in range(y.shape[1]):
            sum_num = 0
            for k in range(x.shape[1]):
                sum_num += x[i][k] * y[k][j]
            sum_mat.append(sum_num)
        final_mat.append(sum_mat)
        
    return np.array(final_mat)

x = np.random.randint(low=1, high=10, size=(4,3))
y = np.random.randint(low=1, high=10, size=(3,2))

matmul_out = matrix_multiplication(x,y)

if type(matmul_out) == np.ndarray:
    print(matmul_out.shape)
    assert np.allclose(np.matmul(x, y), matmul_out)
    print("Matches NumPy! Hurray ;)")
else:
    print("The dimensions did not match.")
    
# I thought of looking at n dimension, but I think it is way not necessary.
# Don't try guys. Stop at 2d. Please...