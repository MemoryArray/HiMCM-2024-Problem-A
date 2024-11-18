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
        return max(1 - penalty, 0)
    
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
    
    # Normalization
    def normalize(values):
        min_val = min(values)
        max_val = max(values)
        return [(val - min_val) / (max_val - min_val) if max_val > min_val else 0 for val in values]
    
    # Calculate individual IFs
    P = popularity_index(N_v, M_c, T_s)
    G = gender_equity(N_f, N_m, avg_salary_f, avg_salary_m, M_s)
    S = sustainability_index(E_c, R, E_e)
    I = inclusivity_index(athletes_data, alpha, beta, gamma)
    R = relevance_innovation(G_p, Y_v)
    SF = safety_fairness(I_r, I_max, S, C_r, J)
    
    # Normalize IFs
    IFs = normalize([P, G, S, I, R, SF])
    
    # Return results in a structured format
    return {
        "Popularity": IFs[0],
        "Gender Equity": IFs[1],
        "Sustainability": IFs[2],
        "Inclusivity": IFs[3],
        "Relevance and Innovation": IFs[4],
        "Safety and Fairness": IFs[5]
    }

def minmax_normalize(values: np.ndarray):
    """
    Normalize values to the range [0, 1] using min-max normalization.
    """
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)

def preprocess(
    daily_visitors, media_coverage, ticket_sales,  # Popularity inputs
    female_athletes, male_athletes, avg_female_salary, avg_male_salary, media_spend,  # Gender Equity inputs
    emissions, total_waste, recyled_waste, initial_investment, NPV,  # Sustainability inputs
    race_deviation, country_deviation, regional_deviation, race_alpha, country_beta, regional_gamma,  # Inclusivity inputs
    generation_participation, youth_viewers,  # Relevance and Innovation inputs
    safety_violations, injury_severity, sanctions, total_checks, judge_impartiality,  # Safety and Fairness inputs
):
    """
    Preprocess data to extract necessary information.
    """
    # Popularity
    N_v = minmax_normalize(daily_visitors)
    M_c = minmax_normalize(media_coverage)
    T_s = minmax_normalize(ticket_sales)
    
    # Gender Equity
    N_f = female_athletes
    N_m = male_athletes
    avg_salary_f = avg_female_salary
    avg_salary_m = avg_male_salary
    M_s = minmax_normalize(media_spend)
    
    # Sustainability
    E_c = minmax_normalize(emissions)
    R = (minmax_normalize(total_waste) - minmax_normalize(recyled_waste)) / minmax_normalize(total_waste)
    E_e = (minmax_normalize(initial_investment) - minmax_normalize(NPV)) / minmax_normalize(initial_investment)
    
    # Inclusivity
    R_k = minmax_normalize(race_deviation)
    C_k = minmax_normalize(country_deviation)
    P_k = minmax_normalize(regional_deviation)
    alpha = race_alpha
    beta = country_beta
    gamma = regional_gamma
    
    # Relevance and Innovation
    G_p = minmax_normalize(generation_participation)
    Y_v = minmax_normalize(youth_viewers)
    
    # Safety and Fairness
    I_r = minmax_normalize(safety_violations)
    I_max = np.max(safety_violations)
    S = minmax_normalize(injury_severity)
    C_r= minmax_normalize(sanctions / total_checks)
    J = minmax_normalize(judge_impartiality)
    
    # Return preprocessed data
    return {
        "N_v": N_v, "M_c": M_c, "T_s": T_s,
        "N_f": N_f, "N_m": N_m, "avg_salary_f": avg_salary_f, "avg_salary_m": avg_salary_m, "M_s": M_s,
        "E_c": E_c, "R": R, "E_e": E_e,
        "athletes_data": [(R_k, C_k, P_k) for R_k, C_k, P_k in zip(R_k, C_k, P_k)], "alpha": alpha, "beta": beta, "gamma": gamma,
        "G_p": G_p, "Y_v": Y_v,
        "I_r": I_r, "I_max": I_max, "S": S, "C_r": C_r, "J": J
    }

