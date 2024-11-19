import pandas as pd
import numpy as np

from utils import data_utils

"""
    daily_visitors, media_coverage, ticket_sales,  # Popularity inputs
    female_athletes, male_athletes, avg_female_salary, avg_male_salary, media_spend,  # Gender Equity inputs
    emissions, total_waste, recyled_waste, initial_investment, NPV,  # Sustainability inputs
    race_deviation, country_deviation, regional_deviation, race_alpha, country_beta, regional_gamma,  # Inclusivity inputs
    generation_participation, youth_viewers,  # Relevance and Innovation inputs
    safety_violations, injury_severity, sanctions, total_checks, judge_impartiality,  # Safety and Fairness inputs
"""
    
def gendata(n=120, seed=42, sigma=0.1, mu=0, **kwargs):
    np.random.seed(seed)
    data = pd.DataFrame()
    for k, v in kwargs.items():
        data[k] = np.random.normal(loc=mu, scale=sigma, size=n) * v
    return data

data_utils.load_athletes()