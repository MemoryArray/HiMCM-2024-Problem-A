import pandas as pd
import numpy as np

from data import OSIF
from utils import data_utils

"""
    daily_visitors, media_coverage, ticket_sales,  # Popularity inputs
    female_athletes, male_athletes, avg_female_salary, avg_male_salary, media_spend,  # Gender Equity inputs
    emissions, total_waste, recyled_waste, initial_investment, NPV,  # Sustainability inputs
    race_deviation, country_deviation, regional_deviation, race_alpha, country_beta, regional_gamma,  # Inclusivity inputs
    generation_participation, youth_viewers,  # Relevance and Innovation inputs
    safety_violations, injury_severity, sanctions, total_checks, judge_impartiality,  # Safety and Fairness inputs
"""
    
def gendata(n=120, seed=42, sigma=0.1, mu=1.125, int_fields=None, float_fields=None, bounded_fields=None, **kwargs):
    np.random.seed(seed)
    data = pd.DataFrame()
    for k, v in kwargs.items():
        if int_fields and k in int_fields:
            # Generate integer data for specified fields
            data[k] = np.round(np.random.normal(loc=mu, scale=sigma, size=n).clip(0.5, 2) * v).astype(int)
        elif float_fields and k in float_fields:
            # Generate float data for specified fields, ensuring values are in [0, 1]
            data[k] = np.random.normal(loc=mu, scale=sigma, size=n) * v
            data[k] = np.clip(data[k], 0, 1)  # Ensure values are in [0, 1]
        elif bounded_fields and k in bounded_fields:
            # Generate float data for bounded fields, ensuring values are in [0, 100]
            data[k] = np.random.normal(loc=mu, scale=sigma, size=n) * v
            data[k] = np.clip(data[k], 0, 100)  # Ensure values are in [0, 100]
        else:
            # Generate normal distribution data for other fields
            data[k] = np.random.normal(loc=mu, scale=sigma, size=n).clip(0.5, 2) * v
    return data

# Specify which fields should be integers
integer_fields = ["daily_visitors", "female_athletes", "male_athletes", "safety_violations", 
                  "sanctions", "total_checks", "judge_impartiality"]

# Specify which fields need to be floats in the range [0, 1]
float_fields = ["race_deviation", "country_deviation", "regional_deviation"]

# Specify which fields need to be floats in the range [0, 100]
bounded_fields = ["generation_participation", "youth_viewers"]

# Expand the data
expanded_data = []

db = OSIF.olympic_sports_factors

race_alpha=0.64794686
country_beta=0.22987118
regional_gamma=0.12218196

# Iterate through each sport
for sport_name, factors in db.items():
    # Generate data for the sport, use static values for the new fields
    generated_data = gendata(n=120, 
                              int_fields=integer_fields, 
                              float_fields=float_fields, 
                              bounded_fields=bounded_fields, 
                              **factors)  # Extract fixed values
    # Add fixed values for race_alpha, country_beta, regional_gamma
    generated_data['race_alpha'] = race_alpha
    generated_data['country_beta'] = country_beta
    generated_data['regional_gamma'] = regional_gamma
    
    # Add the sport's name and generate IDs
    generated_data.insert(0, 'ID', range(1, 121))
    generated_data.insert(1, 'Sport Name', sport_name)
    expanded_data.append(generated_data)

# Combine all expanded data into a single DataFrame
final_data = pd.concat(expanded_data, ignore_index=True)

# Output to CSV
final_data.to_csv('expanded_olympic_sports_data.csv', index=False)

print("Data has been successfully expanded and written to 'expanded_olympic_sports_data.csv'")