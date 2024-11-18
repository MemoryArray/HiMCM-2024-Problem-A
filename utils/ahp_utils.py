import numpy as np

def ahp(pairwise_matrix):
    """
    Perform AHP to calculate weights and check consistency.
    
    Parameters:
    pairwise_matrix (numpy.ndarray): The pairwise comparison matrix (n x n).
    
    Returns:
    tuple: (weights, consistency_ratio)
    """
    # Normalize the pairwise comparison matrix
    column_sums = pairwise_matrix.sum(axis=0)
    normalized_matrix = pairwise_matrix / column_sums
    
    # Calculate weights (priority vector)
    weights = normalized_matrix.mean(axis=1)
    
    # Consistency check
    n = pairwise_matrix.shape[0]
    lambda_max = (pairwise_matrix @ weights).sum() / weights.sum()
    consistency_index = (lambda_max - n) / (n - 1)
    
    # Random Index (RI) values for matrices of size 1 to 10
    random_index = [0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
    ri = random_index[n - 1] if n <= 10 else 1.49  # Use 1.49 for larger matrices
    
    consistency_ratio = consistency_index / ri if ri != 0 else 0
    
    return weights, consistency_ratio