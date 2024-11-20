from utils import if_utils as ifu
from utils import data_utils as du
import pandas as pd
import numpy as np
import data.OSIF as osif

# Load data from csv
db = pd.read_csv(r'data\fulldb.csv')

year_column = db['Sport Name'].map(osif.introtime).apply(du.year_to_olympic_session)
sessionidx = db['ID']

# Define the rows to exclude from normalization
exclude_rows = ['race_alpha', 'country_beta', 'regional_gamma', 'recycled_waste', 'NPV', 'sanctions', 'total_waste', 'initial_investment', 'total_checks']

normdb = ifu.mmnorm(db, exclude_rows=None, whitelist='emissions')

# Applying the preprocessing function to the DataFrame
ppdb = normdb.apply(
    lambda row: ifu.preprocess(
        row['daily_visitors'],
        row['media_coverage'],
        row['ticket_sales'],
        row['female_athletes'],
        row['male_athletes'],
        row['avg_female_salary'],
        row['avg_male_salary'],
        row['media_spend'],
        row['emissions'],
        row['total_waste'],
        row['recycled_waste'],
        row['initial_investment'],
        row['NPV'],
        row['race_deviation'],
        row['country_deviation'],
        row['regional_deviation'],
        row['race_alpha'],
        row['country_beta'],
        row['regional_gamma'],
        row['generation_participation'],
        row['youth_viewers'],
        row['safety_violations'],
        row['injury_severity'],
        row['sanctions'],
        row['total_checks'],
        row['judge_impartiality']
    ),
    axis=1
)

ifdb = []
for row in ppdb:
    ifdb.append(ifu.calculate_IFs(**row))

# Construct to DataFrame
ifdb = pd.DataFrame(ifdb, columns=[
    "Popularity", 
    "Gender Equity", 
    "Sustainability", 
    "Inclusivity", 
    "Relevance and Innovation", 
    "Safety and Fairness"
])

print(ifdb.head())

# Normalize the data
norm_ifdb = ifu.mmnorm(ifdb, exclude_rows)

norm_ifdb['SessionIdx'] = sessionidx
norm_ifdb['SessionIntro'] = np.where(norm_ifdb['SessionIdx'] >= year_column, 1, 0)

print(norm_ifdb.head())

# save the normalized data to csv
norm_ifdb.to_csv(r'data\norm-ifdb.csv', index=False)