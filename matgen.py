import random
import numpy as np
import itertools

def sparse_matrix(n, p):
    """Generates an n dimensional gaussian distributed symmetric matrix with 
    sparsity p
    """
    gaussian = np.random.randn(n, n)
    mat = (gaussian + gaussian.T) / 2

    nums = [i for i in range(n)]
    indices = list(itertools.product(nums, nums))
    random.shuffle(indices)
    
    while (n * n - np.count_nonzero(mat))/(n*n) < p:
        index = indices.pop()
        mat[index] = 0
        index_ref = tuple(reversed(index))

        # (m, m) cannot be removed twice
        if index != index_ref:
            mat[index_ref] = 0
            indices.remove(index_ref)
    
    return mat
    
def lowrank_matrix(n, k):
    """Generates an n dimensional gaussian distributed symmetric matrix with 
    rank k
    """
    L = np.random.randn(n, k)
    R = np.random.randn(k, n)
    M = np.dot(L, R)
    return (M + M.T)/2

def sparse_matrix_PSD(n, p):
    A = sparse_matrix(n, p)
    return A * A.T

def lowrank_matrix_PSD(n, p):
    """Generates a gaussian distributed low-rank matrix B that can be
    written as B = L^TL, with L rank 1 and sparsity p
    """
    L = np.random.randn(n)
    entries = [i for i in range(n)]
    random.shuffle(entries)

    while (n - np.count_nonzero(L)) / n < p:
        index = entries.pop()
        L[index] = 0
    
    return L.T * L
