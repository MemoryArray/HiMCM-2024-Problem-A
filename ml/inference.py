import os
import torch
import torch.nn as nn
import pandas as pd
from utils.data_utils import load_data
from utils.marcos_utils import calculate_combined_weighted_factors

# Constants
DATA_PATH = "../res/plans_data.csv"
CHECKPOINT_PATH = "../models/final_model.pth"
OUTPUT_PATH = "../res/weights.csv"

# Main
if __name__ == "__main__":
    # Load data
    data = load_data(DATA_PATH)

    # Load model
    input_dim = data.shape[1] - 2
    model = nn.RNN(input_size=input_dim, hidden_size=input_dim, num_layers=2, batch_first=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    # Predict weights
    with torch.no_grad():
        features = torch.tensor(data.iloc[:, 2:].values.astype("float32"))
        predicted_weights = model(features).numpy()

    # Save weights
    weights_df = pd.DataFrame(predicted_weights, columns=data.columns[2:])
    weights_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Weights saved to {OUTPUT_PATH}")
