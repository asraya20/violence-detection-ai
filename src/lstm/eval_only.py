import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from train_lstm import ViolenceLSTM, load_data   # 👈 change filename if needed

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/violence_lstm.pth")


# =========================
# 🔹 LOAD MODEL
# =========================
model = ViolenceLSTM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")


# =========================
# 🔹 LOAD DATA
# =========================
X, y = load_data()

# Use part as test (same split logic)


X, y = load_data()

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 🔹 RUN INFERENCE
# =========================
all_preds = []
all_labels = []

with torch.no_grad():
    for i in range(len(X_test)):
        sample = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        label = y_test[i]

        output = model(sample)
        pred = (output > 0.5).float().item()

        all_preds.append(int(pred))
        all_labels.append(int(label))


# =========================
# 🔹 METRICS
# =========================
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n===== FINAL RESULTS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:\n", cm)

print("\nDetailed Report:\n")
print(classification_report(all_labels, all_preds))

plt.figure()

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Normal', 'Violence'],
            yticklabels=['Normal', 'Violence'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")

plt.show()