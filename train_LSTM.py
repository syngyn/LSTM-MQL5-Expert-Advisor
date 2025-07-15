import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import sys
import traceback

# --- IMPROVEMENT: Import shared functions from the new data_processing.py file ---
from data_processing import load_and_align_data, create_features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = SCRIPT_DIR
REQUIRED_FILES = {"EURUSD": "EURUSD60.csv", "EURJPY": "EURJPY60.csv", "USDJPY": "USDJPY60.csv", "GBPUSD": "GBPUSD60.csv", "EURGBP": "EURGBP60.csv", "USDCAD": "USDCAD60.csv", "USDCHF": "USDCHF60.csv"}
INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 12, 128, 2, 20
NUM_CLASSES, LOOKAHEAD_BARS, PROFIT_THRESHOLD_ATR = 3, 5, 0.75
EPOCHS, BATCH_SIZE, LEARNING_RATE = 20, 64, 0.001

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x); out = out[:, -1, :]; return self.fc(out)

if __name__ == "__main__":
    # --- IMPROVEMENT: Call the imported functions ---
    main_df, feature_names = create_features(load_and_align_data(REQUIRED_FILES, DATA_DIR))
    
    if len(main_df) < SEQ_LEN:
        print(f"FATAL ERROR: Not enough data ({len(main_df)} rows) to create sequences."); sys.exit(1)

    print("Creating classification targets...")
    future_price = main_df['EURUSD_close'].shift(-LOOKAHEAD_BARS)
    atr_threshold = main_df['eurusd_atr'] * PROFIT_THRESHOLD_ATR
    conditions = [future_price > main_df['EURUSD_close'] + atr_threshold, future_price < main_df['EURUSD_close'] - atr_threshold]
    choices = [2, 0] # 2=Buy, 0=Sell
    main_df['target'] = np.select(conditions, choices, default=1) # 1=Hold
    main_df.dropna(inplace=True)
    X = main_df[feature_names].values
    y = main_df['target'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Building sequences...")
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        y_seq.append(y[i + SEQ_LEN - 1])
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq), dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Created {len(X_tensor)} sequences.")
    model = LSTMClassifier(input_size=INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"\n--- Starting Model Training ---")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (xb, yb) in enumerate(loader):
            optimizer.zero_grad(); pred = model(xb); loss = loss_fn(pred, yb); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {epoch_loss/len(loader):.6f}")
    print("\n--- Training Complete ---")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    MODEL_FILE = os.path.join(MODEL_SAVE_PATH, "lstm_model.pth")
    SCALER_FILE = os.path.join(MODEL_SAVE_PATH, "scaler.pkl")
    torch.save({"model_state": model.state_dict()}, MODEL_FILE)
    print(f"(+) Model saved to {MODEL_FILE}")
    joblib.dump(scaler, SCALER_FILE)
    print(f"(+) Scaler saved to {SCALER_FILE}")