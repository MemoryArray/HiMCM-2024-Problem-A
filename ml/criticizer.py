import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from models import TransformerModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data and split
data = pd.read_csv("data.csv")
factors = data.iloc[:, 2:8].values
targets = np.random.randint(0, 2, len(data))  # Replace with actual targets

# Untouched testing data
test_data = factors[-600:]
test_labels = targets[-600:]

# Scale data
scaler = MinMaxScaler()
test_data = torch.tensor(scaler.fit_transform(test_data), dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

# Load model checkpoint
checkpoint_path = "checkpoint.pth"
model = TransformerModel(input_dim=test_data.shape[1], output_dim=1)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate model
with torch.no_grad():
    predictions = torch.sigmoid(model(test_data))
    predicted_classes = (predictions > 0.5).float()
    accuracy = accuracy_score(test_labels.numpy(), predicted_classes.numpy())
    auc = roc_auc_score(test_labels.numpy(), predictions.numpy())

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
