import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.transformer import TransformerModel
from utils.data_utils import load_data, split_data
import matplotlib.pyplot as plt

# Constants
DATA_PATH = "../res/plans_data.csv"
CHECKPOINT_DIR = "../models/"
PLOT_DIR = "../res/"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Dataset class
class PlanDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = data.iloc[:, 2:].values.astype("float32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx])

# Training function
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)  # Assuming an autoencoder approach
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Main
if __name__ == "__main__":
    # Load and split data
    data = load_data(DATA_PATH)
    train_data, test_data, _ = split_data(data, 1000, 21000, 0, 1000)

    # Create datasets and dataloaders
    train_dataset = PlanDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, loss, optimizer
    input_dim = train_data.shape[1] - 2
    model = TransformerModel(input_dim, input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    loss_history = []
    for epoch in range(EPOCHS):
        epoch_loss = train_model(train_loader, model, criterion, optimizer, device)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    # Save final model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pth"))

    # Plot loss history
    plt.plot(range(1, EPOCHS + 1), loss_history, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "training_loss.png"))
    plt.show()
