import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils.data_utils import load_data, split_data

# Constants
DATA_PATH = "../res/plans_data.csv"
CHECKPOINT_PATH = "../models/final_model.pth"
BATCH_SIZE = 64

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.features = data.iloc[:, :6].values.astype("float32")  # First six columns are factors
        self.targets = data.iloc[:, -1].values.astype("float32")  # SessionIntro column

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.targets[idx]),
        )

# Model definition (same as training)
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

# Evaluation function
def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    criterion = torch.nn.BCELoss()  # Use BCELoss for MARCOS-based logic
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)  # Predict weights
            scores = (predictions * features).sum(dim=1)  # MARCOS weighted sum
            probabilities = torch.sigmoid(scores)  # Convert scores to probabilities
            loss = criterion(probabilities, targets)  # Binary cross-entropy loss
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Main
if __name__ == "__main__":
    # Load and split data
    data = load_data(DATA_PATH)
    _, test_data, _ = split_data(data, 0, 4920, 6000, 6120)

    # Create test dataset and loader
    test_dataset = PlanDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    input_dim = 6  # Six factors
    model = ImpactFactorModel(input_dim)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model = model.to(device)

    # Evaluate model
    test_loss = evaluate_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
