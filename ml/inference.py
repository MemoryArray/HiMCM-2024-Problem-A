import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models import TransformerModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data and preprocess
data = pd.read_csv("data.csv")
factors = data.iloc[:, 2:8].values

# Load scaler
scaler = MinMaxScaler()
scaled_factors = torch.tensor(scaler.fit_transform(factors), dtype=torch.float32)

# Load model checkpoint
checkpoint_path = "model_checkpoint.pth"
model = TransformerModel(input_dim=scaled_factors.shape[1], output_dim=1)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Perform inference
with torch.no_grad():
    predictions = torch.sigmoid(model(scaled_factors)).numpy()

# Save predictions
data["Predictions"] = predictions
data.to_csv("predictions.csv", index=False)
print("Predictions saved to 'predictions.csv'.")
