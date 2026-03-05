import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 15, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='Dataset/breathing_dataset.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df['label'] = df['label'].replace({'Body event': 'Other', 'Mixed Apnea': 'Other'})

    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label'])

    flow_cols   = [c for c in df.columns if c.startswith('flow_')]
    thorac_cols = [c for c in df.columns if c.startswith('thorac_')]
    spo2_cols   = [c for c in df.columns if c.startswith('spo2_')]

    flow_data   = df[flow_cols].values.astype(np.float32)
    thorac_data = df[thorac_cols].values.astype(np.float32)
    spo2_up     = np.repeat(df[spo2_cols].values.astype(np.float32), 8, axis=1)
    X           = np.stack([flow_data, thorac_data, spo2_up], axis=1)
    y           = df['label_enc'].values.astype(np.int64)
    participants = df['participant'].values

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_pids = sorted(df['participant'].unique())
    results  = []

    for test_pid in all_pids:
        print(f"\nFold: test on {test_pid}")
        train_mask = participants != test_pid
        test_mask  = participants == test_pid

        X_train = torch.tensor(X[train_mask])
        y_train = torch.tensor(y[train_mask])
        X_test  = torch.tensor(X[test_mask])
        y_test  = torch.tensor(y[test_mask])

        classes = np.unique(y[train_mask])
        cw      = compute_class_weight('balanced', classes=classes, y=y[train_mask])
        weights = np.ones(len(le.classes_))
        weights[classes] = cw
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        loader    = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        model     = CNN1D(num_classes=len(le.classes_)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for epoch in range(15):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec  = recall_score(y_test, preds, average="macro", zero_division=0)
        cm   = confusion_matrix(y_test, preds)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        results.append({'fold': test_pid, 'accuracy': acc, 'precision': prec, 'recall': rec})

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['fold']}  acc={r['accuracy']:.4f}  prec={r['precision']:.4f}  rec={r['recall']:.4f}")
    print(f"Mean Accuracy : {np.mean([r['accuracy']  for r in results]):.4f}")
    print(f"Mean Precision: {np.mean([r['precision'] for r in results]):.4f}")
    print(f"Mean Recall   : {np.mean([r['recall']    for r in results]):.4f}")
    print(f"Classes: {list(le.classes_)}")