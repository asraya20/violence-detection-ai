import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/violence_lstm.pth")


class ViolenceLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=256, num_layers=2):
        super(ViolenceLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.4
        )

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


def load_model():
    model = ViolenceLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def predict_sequence(model, sequence):
    sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(sequence)
    return output.item()
