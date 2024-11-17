import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
data = pd.read_csv("data.csv")
factors = data.iloc[:, 2:8].values
targets = np.random.randint(0, 2, len(data))  # Replace with actual target labels if available

# Split data
train_data = factors[1000:21000]
train_labels = targets[1000:21000]
val_data = factors[0:1000]
val_labels = targets[0:1000]

# Convert to PyTorch tensors
scaler = MinMaxScaler()
train_data = torch.tensor(scaler.fit_transform(train_data), dtype=torch.float32)
val_data = torch.tensor(scaler.transform(val_data), dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
val_labels = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=64, shuffle=False)

# Transformer Model Definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=128),
            num_layers=2
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.fc(x)

# Initialize model, loss, and optimizer
model = TransformerModel(input_dim=train_data.shape[1], output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with checkpoint saving
epochs = 20
checkpoint_path = "model_checkpoint.pth"
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_data_batch, val_labels_batch in val_loader:
            val_outputs = model(val_data_batch)
            loss = criterion(val_outputs, val_labels_batch)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("loss_curve.png")
plt.show()
