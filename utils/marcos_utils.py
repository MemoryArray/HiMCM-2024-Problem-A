import pandas as pd
import numpy as np

def calculate_combined_weighted_factors(data, weights, factors):
    """
    Computes the combined weighted score for each plan.

    Args:
        data (pd.DataFrame): The dataframe containing plan data.
        weights (list or np.ndarray): The weights for the factors.
        factors (list): List of factor column names.

    Returns:
        pd.Series: A series with combined weighted scores for each plan.
    """
    weighted_factors = data[factors] * weights
    combined_scores = weighted_factors.sum(axis=1)
    return combined_scores

def marcos_method(data, factors, weights):
    """
    Applies the MARCOS method to rank plans based on weighted factors.

    Args:
        data (pd.DataFrame): The dataframe containing plan data.
        factors (list): List of factor column names.
        weights (list or np.ndarray): Weights for the factors.

    Returns:
        pd.DataFrame: Dataframe with ranks and final scores.
    """
    # 1. Extend decision matrix
    ideal = data[factors].max()
    anti_ideal = data[factors].min()
    extended_data = data.copy()
    extended_data.loc['Ideal'] = ideal
    extended_data.loc['Anti-Ideal'] = anti_ideal

    # 2. Normalize the decision matrix
    for col in factors:
        extended_data[col] = extended_data[col] / ideal[col]

    # 3. Apply weights
    for col, weight in zip(factors, weights):
        extended_data[col] *= weight

    # 4. Scoring
    scores = extended_data[factors].sum(axis=1)
    extended_data['Score'] = scores

    # 5. Utility degrees
    ideal_score = scores.loc['Ideal']
    anti_ideal_score = scores.loc['Anti-Ideal']
    extended_data['Utility'] = scores / ideal_score - scores / anti_ideal_score

    # 6. Ranking
    extended_data['Rank'] = extended_data['Utility'].rank(ascending=False)

    return extended_data.sort_values('Rank')
