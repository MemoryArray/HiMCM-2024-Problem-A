import torch.nn as nn

nn.RNN()

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