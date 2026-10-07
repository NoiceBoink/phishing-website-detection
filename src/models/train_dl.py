"""
Deep Learning Character-Level CNN-BiLSTM for Raw URL Phishing Classification.
Takes raw URL strings as input sequences (without manual feature engineering),
tokenizes character-by-character, and predicts phishing probability.
"""

import os
import json
import time
import string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Character vocabulary: pad (0), unk (1), plus printable ASCII
CHARS = string.ascii_letters + string.digits + string.punctuation + " "
CHAR2IDX = {c: i + 2 for i, c in enumerate(CHARS)}
CHAR2IDX["<pad>"] = 0
CHAR2IDX["<unk>"] = 1
VOCAB_SIZE = len(CHAR2IDX)
MAX_URL_LEN = 200


def encode_url(url: str, max_len: int = MAX_URL_LEN) -> np.ndarray:
    """Encodes a URL string into a fixed-length sequence of character indices."""
    indices = [CHAR2IDX.get(c, 1) for c in str(url)[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return np.array(indices, dtype=np.int64)


class URLDataset(Dataset):
    def __init__(self, urls, labels):
        self.urls = [encode_url(u) for u in urls]
        self.labels = np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.urls[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


class CharCNNLSTM(nn.Module):
    """
    1D-CNN + Bidirectional LSTM architecture for character-level representation learning.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, conv_filters=128, lstm_hidden=64):
        super(CharCNNLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 1D Conv captures n-gram character patterns (subdomains, brand names, keywords)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=conv_filters, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        # BiLSTM captures long-range sequential structure of URL components
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embeds = self.embedding(x) # [batch_size, seq_len, embed_dim]
        conv_in = embeds.permute(0, 2, 1) # [batch_size, embed_dim, seq_len]
        conv_out = self.relu(self.conv1(conv_in)) # [batch_size, conv_filters, seq_len]
        conv_out = self.dropout1(conv_out)

        lstm_in = conv_out.permute(0, 2, 1) # [batch_size, seq_len, conv_filters]
        lstm_out, _ = self.lstm(lstm_in) # [batch_size, seq_len, 2 * lstm_hidden]

        # Max pooling across sequence length
        pooled, _ = torch.max(lstm_out, dim=1) # [batch_size, 2 * lstm_hidden]
        dense = self.dropout2(self.relu(self.fc1(pooled)))
        logits = self.fc2(dense).squeeze(1)
        return logits


def train_deep_learning_model(
    raw_data_path: str = "data/raw/urls.csv",
    output_dir: str = "saved_models",
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001
):
    """
    Trains the Char-CNN-BiLSTM model and logs metrics alongside classical ML models.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Deep Learning] Training Char-CNN-LSTM on device: {device}")

    df = pd.read_csv(raw_data_path)
    urls = df["url"].values
    labels = df["label"].values

    urls_train, urls_test, y_train, y_test = train_test_split(
        urls, labels, test_size=0.20, random_state=42, stratify=labels
    )

    train_ds = URLDataset(urls_train, y_train)
    test_ds = URLDataset(urls_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = CharCNNLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)

        epoch_loss = total_loss / len(train_ds)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    train_time = time.time() - t0

    # Evaluation
    model.eval()
    all_preds = []
    all_probs = []
    t_infer_start = time.time()

    with torch.no_grad():
        for x_b, _ in test_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)

    latency_ms = ((time.time() - t_infer_start) / len(test_ds)) * 1000
    y_test_arr = np.array(y_test)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    acc = float(accuracy_score(y_test_arr, preds_arr))
    prec = float(precision_score(y_test_arr, preds_arr, zero_division=0))
    rec = float(recall_score(y_test_arr, preds_arr, zero_division=0))
    f1 = float(f1_score(y_test_arr, preds_arr, zero_division=0))
    auc = float(roc_auc_score(y_test_arr, probs_arr))

    print(f"\n[Char-CNN-LSTM Results]")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"Inference Latency: {latency_ms:.3f} ms/sample | Train Time: {train_time:.2f}s")

    # Save model weights and vocab mapping
    model_save_path = os.path.join(output_dir, "char_cnn_lstm.pt")
    torch.save(model.state_dict(), model_save_path)
    with open(os.path.join(output_dir, "char_vocab.json"), "w") as f:
        json.dump({"vocab": CHAR2IDX, "max_len": MAX_URL_LEN}, f)

    # Append to model_metrics.json
    metrics_file = os.path.join(output_dir, "model_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            summary = json.load(f)
        summary["models"]["Deep Learning (Char-CNN-LSTM)"] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
            "Inference_Latency_ms": round(latency_ms, 4),
            "Train_Time_s": round(train_time, 4),
            "Paradigm": "End-to-End Sequence (No Manual Features)"
        }
        with open(metrics_file, "w") as f:
            json.dump(summary, f, indent=2)

    return model, {"acc": acc, "f1": f1, "auc": auc}


if __name__ == "__main__":
    train_deep_learning_model()

