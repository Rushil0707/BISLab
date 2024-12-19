import numpy as np
import multiprocessing

def fitness_function(matrix):
    return np.var(matrix)

def update_cell(i, j, matrix, size):
    neighbors = []
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = (i + di) % size, (j + dj) % size
            neighbors.append(matrix[ni, nj])
    return np.mean(neighbors)

def parallel_cell_update(matrix, size):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    result = []
    for i in range(size):
        for j in range(size):
            result.append(pool.apply_async(update_cell, (i, j, matrix, size)))
    pool.close()
    pool.join()
    
    new_matrix = np.array([r.get() for r in result]).reshape(matrix.shape)
    return new_matrix

def cellular_optimization(matrix, max_iter=10):
    size = matrix.shape[0]
    for t in range(max_iter):
        matrix = parallel_cell_update(matrix, size)
        print(f"Iteration {t + 1}: Matrix:\n{matrix}")
    return matrix

matrix = np.random.rand(5, 5)
optimized_matrix = cellular_optimization(matrix)
print("Optimized Matrix:\n", optimized_matrix)
