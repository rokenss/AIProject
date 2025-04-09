import numpy as np
import pandas as pd
from scipy.ndimage import sobel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import torch
import torch.nn as nn
import torch.optim as optim

# === Load Data ===
def load_famous48_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    L = int(lines[0])
    N = int(lines[1])
    data, labels = [], []
    for line in lines[2:]:
        values = list(map(float, line.strip().split()))
        if len(values) >= N + 8:
            pixels = values[:N]
            label = int(values[-6])  # a3 - class
            data.append(pixels)
            labels.append(label)
    return np.array(data), np.array(labels)

# Load all 3 parts
X1, y1 = load_famous48_file("famous48/x24x24.txt")
X2, y2 = load_famous48_file("famous48/y24x24.txt")
X3, y3 = load_famous48_file("famous48/z24x24.txt")
X = np.vstack([X1, X2, X3])
y = np.concatenate([y1, y2, y3])

# === Compute Gradient Angles ===
def compute_gradient_angles(X_flat):
    angle_features = []
    for row in X_flat:
        img = row.reshape(24, 24)
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        angle = np.arctan2(gy, gx).flatten()
        angle_features.append(angle)
    return np.array(angle_features)

X_angles = compute_gradient_angles(X)
X_augmented = np.hstack([X, X_angles])

# === Standardize + Split ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_augmented)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=42)

# === Evaluate Decision Tree and Random Forest ===
def evaluate_sklearn_models(X_train, X_test, y_train, y_test):
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=30, min_samples_leaf=2, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=50, max_features='sqrt', random_state=42)
    }
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name}: Accuracy = {acc:.4f}, Time = {time.time() - start:.2f}s")

# === PyTorch NN ===
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def evaluate_pytorch_model(X_train, X_test, y_train, y_test, epochs=30, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNN(X_train.shape[1], len(np.unique(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor.to(device)).argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        print(f"Neural Network (PyTorch): Accuracy = {acc:.4f}, Time = {time.time() - start:.2f}s")

# === Run All ===
evaluate_sklearn_models(X_train, X_test, y_train, y_test)
evaluate_pytorch_model(X_train, X_test, y_train, y_test)