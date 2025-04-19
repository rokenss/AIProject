import numpy as np
import time
from scipy.ndimage import sobel
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
            label = int(values[-6])
            data.append(pixels)
            labels.append(label)
    return np.array(data), np.array(labels)

# === Feature Extraction ===
def compute_gradient_angles(X_flat):
    features = []
    for row in X_flat:
        img = row.reshape(24, 24)
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        angles = np.arctan2(gy, gx).flatten()
        features.append(angles)
    return np.array(features)

def extract_lbp_features(X_flat, radius=1, n_points=8):
    features = []
    for row in X_flat:
        img = row.reshape(24, 24)
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), density=True)
        features.append(hist)
    return np.array(features)

def extract_hog_features(X_flat):
    features = []
    for row in X_flat:
        img = row.reshape(24, 24)
        hog_desc = hog(img, orientations=9, pixels_per_cell=(6, 6),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        features.append(hog_desc)
    return np.array(features)

def extract_haar_features(X_flat):
    features = []
    for row in X_flat:
        img = row.reshape(24, 24)
        ii = img.cumsum(axis=0).cumsum(axis=1)
        vals = []
        for y in range(0, 20, 6):
            for x in range(0, 20, 6):
                A = ii[y, x]
                B = ii[y, x+5] if x+5 < 24 else ii[y, 23]
                C = ii[y+5, x] if y+5 < 24 else ii[23, x]
                D = ii[y+5, x+5] if x+5 < 24 and y+5 < 24 else ii[23, 23]
                vals.append(D + A - B - C)
        features.append(vals)
    return np.array(features)

def extract_lab_features(X_flat):
    features = []
    for row in X_flat:
        img = row.reshape(24, 24)
        patch_vals = []
        for i in range(0, 24, 4):
            for j in range(0, 24, 4):
                patch = img[i:i+4, j:j+4]
                mean = np.mean(patch)
                lab = (patch > mean).astype(int).flatten()
                patch_vals.extend(lab)
        features.append(patch_vals)
    return np.array(features)

# === PyTorch Neural Network ===
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def evaluate_pytorch_model(X_train, X_test, y_train, y_test, epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNN(X_train.shape[1], len(np.unique(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

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
        print(f"Epoch {epoch+1}, Test Accuracy: {acc:.4f}")

# === Main Pipeline ===
def main():
    X1, y1 = load_famous48_file("famous48/x24x24.txt")
    X2, y2 = load_famous48_file("famous48/y24x24.txt")
    X3, y3 = load_famous48_file("famous48/z24x24.txt")
    X = np.vstack([X1, X2, X3])
    y = np.concatenate([y1, y2, y3])

    print("Extracting features...")
    X_angle = compute_gradient_angles(X)
    X_lbp = extract_lbp_features(X)
    X_hog = extract_hog_features(X)
    X_haar = extract_haar_features(X)
    X_lab = extract_lab_features(X)

    X_all = np.hstack([X, X_angle, X_lbp, X_hog, X_haar, X_lab])
    print("Feature shape:", X_all.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, stratify=y, random_state=42)

    print("\nTraining Random Forest...")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=400, max_depth=60, max_features='sqrt', random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest: Accuracy = {acc_rf:.4f}, Time = {time.time() - start:.2f}s")

    print("\nTraining Neural Network...")
    evaluate_pytorch_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
