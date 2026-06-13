import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# -----------------------------
# Settings
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path("datasets/sequences_clean")
MODEL_SAVE_PATH = Path("models/violence_lstm.pth")


# -----------------------------
# Dataset Class
# -----------------------------
class ViolenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# LSTM Model
# -----------------------------
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

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    violence = np.load(DATA_PATH / "violence.npy")
    non_violence = np.load(DATA_PATH / "non_violence.npy")

    # Balance classes
    min_samples = min(len(violence), len(non_violence))

    violence = violence[:min_samples]
    non_violence = non_violence[:min_samples]

    X = np.concatenate((violence, non_violence), axis=0)

    y_violence = np.ones(len(violence))
    y_non = np.zeros(len(non_violence))
    y = np.concatenate((y_violence, y_non), axis=0)

    return X, y

# -----------------------------
# Training Loop
# -----------------------------
def train():

    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = ViolenceDataset(X_train, y_train)
    val_dataset = ViolenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = ViolenceLSTM().to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining on {DEVICE}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).unsqueeze(1)

                outputs = model(X_batch)
                predicted = (outputs > 0.5).float()

                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = 100 * correct / total

        print(f"\nEpoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {train_loss:.4f} "
              f"Val Accuracy: {accuracy:.2f}%\n")

    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Model saved to:", MODEL_SAVE_PATH)
   

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1)

            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)

    print(f"\nConfusion Matrix:")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Val Accuracy: {accuracy:.2f}%\n")


if __name__ == "__main__":
    train()
