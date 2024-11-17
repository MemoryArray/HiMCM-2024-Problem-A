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


def calculate_score(factors, weights):
    """
    Calculate the weighted sum of a series of factors.
    
    Parameters:
    factors (list or numpy.ndarray): The values of the factors.
    weights (list or numpy.ndarray): The weights corresponding to the factors.
    
    Returns:
    float: The weighted score.
    """
    if len(factors) != len(weights):
        raise ValueError("The number of factors must match the number of weights.")
    
    # Calculate weighted sum
    weighted_score = np.dot(factors, weights)
    return weighted_score

# Example: Pairwise comparison matrix of three factors of popularity
pairwise_matrix = np.array([
    [1, 1/3, 3],
    [3, 1, 5],
    [1/3, 1/5, 1]
])

# Calculate weights and consistency ratio
weights, cr = ahp(pairwise_matrix)

print("Weights:", weights)
print("Consistency Ratio (CR):", cr)

if cr < 0.1:
    print("The consistency ratio is acceptable.")
else:
    print("The consistency ratio is too high. Reevaluate the pairwise comparisons.")

factors = [1, 1, 1]

# Calculate the score
score = calculate_score(factors, weights)
print("Weighted Score:", score)
