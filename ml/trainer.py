import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import utils as du  # Ensure this file has appropriate data loading/splitting functions
from sklearn.preprocessing import MinMaxScaler
import pandas as pd  # Add pandas to save weights

import scienceplots

plt.style.use('science')

# Constants
DATA_PATH = r"E:\Mathmodel\2024HiMCM\repo\HiMCM-2024-Problem-A\data\norm-ifdb.csv"
CHECKPOINT_DIR = r"E:\Mathmodel\2024HiMCM\repo\HiMCM-2024-Problem-A\models"
PLOT_DIR = "../res/"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.01

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = MinMaxScaler()

# Updated Dataset Class
class PlanDataset(Dataset):
    def __init__(self, data):
        # Extract features and target
        self.features = data.iloc[:, :6].values.astype("float32")  # First six columns are factors
        self.targets = data.iloc[:, -1].values.astype("float32")  # Last column is `SessionIntro`

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),  # Factors
            torch.tensor(self.targets[idx]),  # SessionIntro
        )

# Define the Model
class ImpactFactorModel(nn.Module):
    def __init__(self, input_dim):
        super(ImpactFactorModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, input_dim)  # Predict weights for the 6 factors

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Output weights in [0, 1]

# Custom Loss for MARCOS Logic
def marcos_loss(outputs, features, targets):
    """
    MARCOS logic: scores are computed using the predicted weights and features.
    Loss is the binary cross-entropy between the MARCOS score and the actual target.
    """
    scores = (outputs * features).sum(dim=1)  # MARCOS weighted sum
    probabilities = torch.sigmoid(scores)  # Convert scores to probabilities
    return nn.BCELoss()(probabilities, targets)  # Binary cross-entropy loss

# Training Function
def train_model(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = marcos_loss(outputs, features, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Main
if __name__ == "__main__":
    # Load data
    data = du.load_data(DATA_PATH)
    train_data, test_data, _ = du.split_data(data, 0, 4920, 6000, 6120)
    data.iloc[:, :6] = scaler.fit_transform(data.iloc[:, :6])

    # Dataset and DataLoader
    train_dataset = PlanDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model setup
    input_dim = 6  # Six factors
    model = ImpactFactorModel(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    loss_history = []
    for epoch in range(EPOCHS):
        epoch_loss = train_model(train_loader, model, optimizer, device)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    # Save final model
    final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # Output the final weights
    final_weights = {
        "fc1_weights": model.fc1.weight.detach().cpu().numpy(),
        "fc2_weights": model.fc2.weight.detach().cpu().numpy(),
        "fc3_weights": model.fc3.weight.detach().cpu().numpy()
    }

    # Save final weights to CSV
    fc1_weights_df = pd.DataFrame(final_weights["fc1_weights"])
    fc2_weights_df = pd.DataFrame(final_weights["fc2_weights"])
    fc3_weights_df = pd.DataFrame(final_weights["fc3_weights"])

    # Save weights into separate CSV files
    fc1_weights_df.to_csv(os.path.join(PLOT_DIR, "fc1_weights.csv"), index=False)
    fc2_weights_df.to_csv(os.path.join(PLOT_DIR, "fc2_weights.csv"), index=False)
    fc3_weights_df.to_csv(os.path.join(PLOT_DIR, "fc3_weights.csv"), index=False)

    print("Final weights saved to CSV.")

    # Plot training loss
    plt.plot(range(1, EPOCHS + 1), loss_history, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "training_loss.png"))
    plt.show()
