import os
import torch
import pandas as pd
from utils.data_utils import load_data

# Constants
DATA_PATH = "../res/plans_data.csv"
CHECKPOINT_PATH = "../models/final_model.pth"
OUTPUT_PATH = "../res/weights.csv"

# Define the same model as in training
class ImpactFactorModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(ImpactFactorModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, input_dim)  # Predict weights for the 6 factors

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Output weights in [0, 1]

# Main
if __name__ == "__main__":
    # Load data
    data = load_data(DATA_PATH)
    features = data.iloc[:, :6].values.astype("float32")  # First six columns are factors

    # Load model
    input_dim = features.shape[1]  # Should be 6 based on the dataset
    model = ImpactFactorModel(input_dim)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    # Predict weights
    with torch.no_grad():
        features_tensor = torch.tensor(features)
        predicted_weights = model(features_tensor).numpy()  # Predict weights

    # Save predicted weights
    weights_df = pd.DataFrame(predicted_weights, columns=data.columns[:6])  # Match feature column names
    weights_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Weights saved to {OUTPUT_PATH}")
