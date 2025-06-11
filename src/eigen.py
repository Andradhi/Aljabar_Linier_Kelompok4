import numpy as np

def power_iteration(A, num_simulations: int = 100):
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_simulations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        if b_k1_norm == 0:
            return 0, b_k
        b_k = b_k1 / b_k1_norm
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k))
    return eigenvalue, b_k

def get_top_k_eigenvectors(cov_matrix, k=30):
    eigenvectors = []
    eigenvalues = []

    A = np.copy(cov_matrix)
    for _ in range(k):
        val, vec = power_iteration(A)
        eigenvectors.append(vec)
        eigenvalues.append(val)
        A = A - val * np.outer(vec, vec)

    return np.array(eigenvalues), np.array(eigenvectors)

def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))