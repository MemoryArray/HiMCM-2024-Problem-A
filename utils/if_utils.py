import numpy as np

def calculate_IFs(
    N_v, M_c, T_s,  # Popularity inputs
    N_f, N_m, avg_salary_f, avg_salary_m, M_s,  # Gender Equity inputs
    E_c, R, E_e,  # Sustainability inputs
    athletes_data, alpha, beta, gamma,  # Inclusivity inputs
    G_p, Y_v,  # Relevance and Innovation inputs
    I_r, I_max, S, C_r, J  # Safety and Fairness inputs
):
    """
    Calculate all Influence Factors (IFs) and return normalized results.

    ## Example usage
    >>> inputs = {
        "N_v": 0.8, "M_c": 0.9, "T_s": 0.7,  # Popularity inputs
        "N_f": 40, "N_m": 60, "avg_salary_f": 50000, "avg_salary_m": 55000, "M_s": 0.8,  # Gender Equity inputs
        "E_c": 0.2, "R": 0.9, "E_e": 0.7,  # Sustainability inputs
        "athletes_data": [(0.9, 0.8, 0.7), (0.7, 0.6, 0.5)], "alpha": 1, "beta": 1, "gamma": 1,  # Inclusivity inputs
        "G_p": 0.8, "Y_v": 0.9,  # Relevance and Innovation inputs
        "I_r": 5, "I_max": 100, "S": 0.2, "C_r": 0.3, "J": 0.95  # Safety and Fairness inputs
        }
    >>> results = calculate_IFs(**inputs)
    >>> print("Normalized Influence Factors:")
    >>> for key, value in results.items():
            print(f"{key}: {value:.3f}")
        Normalized Influence Factors:
        Popularity: 0.5605119430
        Gender Equity: 0.1717733570
        Sustainability: 0.0000000000
        Inclusivity: 1.0000000000
        Relevance and Innovation: 0.1792243205
        Safety and Fairness: 0.0195429931

    """
    # Popularity Index
    def popularity_index(N_v, M_c, T_s):
        return np.cbrt((1 + N_v) * (1 + M_c) * (1 + T_s))
    
    # Gender Equity
    def gender_equity(N_f, N_m, avg_salary_f, avg_salary_m, M_s):
        R_f = N_f / (N_f + N_m)
        E = 1 - abs(avg_salary_f / avg_salary_m - 1)
        return 1 - np.sqrt((R_f - (1 - R_f))**2 + (E - 1)**2 + (M_s - 1)**2) / np.sqrt(3)
    
    # Sustainability Index
    def sustainability_index(E_c, R, E_e):
        penalty = E_c + (1 - R) + (1 - E_e)
        return max(10 - penalty, 0.01)
    
    # Inclusivity Index
    def inclusivity_index(athletes_data, alpha, beta, gamma):
        inclusivity_sum = sum(
            np.exp(alpha * R_k + beta * C_k + gamma * P_k) 
            for R_k, C_k, P_k in athletes_data
        )
        return np.log(1 + inclusivity_sum)
    
    # Relevance and Innovation
    def relevance_innovation(G_p, Y_v):
        if G_p > 0 and Y_v > 0:
            return 2 / (1 / G_p + 1 / Y_v)
        else:
            return 0  # Handle cases where G_p or Y_v is zero
    
    # Safety and Fairness
    def safety_fairness(I_r, I_max, S, C_r, J):
        risk = (I_r / I_max) + S + C_r
        return J * np.exp(-risk)
    
    # Calculate individual IFs
    P = popularity_index(N_v, M_c, T_s)
    G = gender_equity(N_f, N_m, avg_salary_f, avg_salary_m, M_s)
    S = sustainability_index(E_c, R, E_e)
    I = inclusivity_index(athletes_data, alpha, beta, gamma)
    R = relevance_innovation(G_p, Y_v)
    SF = safety_fairness(I_r, I_max, S, C_r, J)
    
    # Return results in a structured format
    return {
        "Popularity": P,
        "Gender Equity": G,
        "Sustainability": S,
        "Inclusivity": I,
        "Relevance and Innovation": R,
        "Safety and Fairness": SF
    }

# Normalize function
def mmnorm(df, exclude_rows=None, whitelist=None):
    # Create a copy of the DataFrame to avoid modifying the original data
    normalized_df = df.copy()
    
    # Get the numeric columns
    numeric_cols = normalized_df.select_dtypes(include=[int, float]).columns
    
    # Get the min and max values for normalization
    min_vals = normalized_df[numeric_cols].min()
    max_vals = normalized_df[numeric_cols].max()
    
    # Normalize each column except for the specified rows
    for column in numeric_cols:
        # Normalize if the column is not in the exceptions
        if (exclude_rows is None or column not in exclude_rows) and (whitelist is None or column in whitelist):
            normalized_df[column] = (normalized_df[column] - min_vals[column]) / (max_vals[column] - min_vals[column])
    
    return normalized_df

# Preprocess function
def preprocess(
    daily_visitors, media_coverage, ticket_sales,  # Popularity inputs
    female_athletes, male_athletes, avg_female_salary, avg_male_salary, media_spend,  # Gender Equity inputs
    emissions, total_waste, recycled_waste, initial_investment, NPV,  # Sustainability inputs
    race_deviation, country_deviation, regional_deviation, race_alpha, country_beta, regional_gamma,  # Inclusivity inputs
    generation_participation, youth_viewers,  # Relevance and Innovation inputs
    safety_violations, injury_severity, sanctions, total_checks, judge_impartiality,  # Safety and Fairness inputs
):
    """
    Preprocess data to extract necessary information.
    """
    # Popularity
    N_v = daily_visitors
    M_c = media_coverage
    T_s = ticket_sales
    
    # Gender Equity
    N_f = female_athletes
    N_m = male_athletes
    avg_salary_f = avg_female_salary
    avg_salary_m = avg_male_salary
    M_s = media_spend
    
    # Sustainability
    E_c = emissions
    R = (total_waste - recycled_waste) / total_waste if total_waste > 0 else 0
    E_e = (initial_investment - NPV) / initial_investment
    
    # Inclusivity
    R_k = race_deviation
    C_k = country_deviation
    P_k = regional_deviation
    alpha = race_alpha
    beta = country_beta
    gamma = regional_gamma
    
    # Relevance and Innovation
    G_p = generation_participation
    Y_v = youth_viewers
    
    # Safety and Fairness
    I_r = safety_violations
    I_max = np.max(safety_violations)
    S = injury_severity
    C_r = sanctions / total_checks
    J = judge_impartiality
    
    # Return preprocessed data
    return {
        "N_v": N_v, "M_c": M_c, "T_s": T_s,
        "N_f": N_f, "N_m": N_m, "avg_salary_f": avg_salary_f, "avg_salary_m": avg_salary_m, "M_s": M_s,
        "E_c": E_c, "R": R, "E_e": E_e,
        "athletes_data": [(R_k, C_k, P_k)],
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "G_p": G_p, "Y_v": Y_v,
        "I_r": I_r, "I_max": I_max, "S": S, "C_r": C_r, "J": J
    }