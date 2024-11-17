import torch
from torch.utils.data import DataLoader
from models.transformer import TransformerModel
from utils.data_utils import load_data, split_data

# Constants
DATA_PATH = "../data/plans_data.csv"
CHECKPOINT_PATH = "../checkpoints/final_model.pth"
BATCH_SIZE = 64

# Dataset class
class PlanDataset:
    def __init__(self, data):
        self.features = data.iloc[:, 2:].values.astype("float32")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx])

# Evaluation function
def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            predictions = model(batch)
            loss = criterion(predictions, batch)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Main
if __name__ == "__main__":
    # Load and split data
    data = load_data(DATA_PATH)
    _, test_data, _ = split_data(data, 1000, 21000, 0, 1000)

    # Create test dataset and loader
    test_dataset = PlanDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    input_dim = test_data.shape[1] - 2
    model = TransformerModel(input_dim, input_dim)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate model
    test_loss = evaluate_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
