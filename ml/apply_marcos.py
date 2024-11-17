import pandas as pd
from utils.data_utils import load_data
from utils.marcos_utils import marcos_method

# Constants
DATA_PATH = "../data/plans_data.csv"
WEIGHTS_PATH = "../data/weights.csv"
OUTPUT_PATH = "../data/ranked_plans.csv"

# Main
if __name__ == "__main__":
    # Load data and weights
    data = load_data(DATA_PATH)
    weights = pd.read_csv(WEIGHTS_PATH).mean().values  # Average weights
    factors = data.columns[2:]

    # Apply MARCOS
    ranked_data = marcos_method(data, factors, weights)

    # Save ranked results
    ranked_data.to_csv(OUTPUT_PATH, index=False)
    print(f"Ranked plans saved to {OUTPUT_PATH}")
