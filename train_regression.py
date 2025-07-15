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

from data_processing import load_and_align_data, create_features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = SCRIPT_DIR
REQUIRED_FILES = {"EURUSD": "EURUSD60.csv", "EURJPY": "EURJPY60.csv", "USDJPY": "USDJPY60.csv", "GBPUSD": "GBPUSD60.csv", "EURGBP": "EURGBP60.csv", "USDCAD": "USDCAD60.csv", "USDCHF": "USDCHF60.csv"}
INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 12, 128, 2, 20

# --- CHANGE: Increased prediction horizon from 5 to 24 bars ---
OUTPUT_STEPS, EPOCHS, BATCH_SIZE, LEARNING_RATE = 24, 25, 64, 0.001

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_steps):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_steps)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

if __name__ == "__main__":
    main_df, feature_names = create_features(load_and_align_data(REQUIRED_FILES, DATA_DIR))

    if len(main_df) < SEQ_LEN + OUTPUT_STEPS:
        print(f"FATAL ERROR: Not enough data ({len(main_df)}) for sequence ({SEQ_LEN}) and lookahead ({OUTPUT_STEPS})."); sys.exit(1)
    
    print(f"Creating regression targets for {OUTPUT_STEPS} steps...")
    targets = []
    price_to_predict = main_df['EURUSD_close']
    for i in range(1, OUTPUT_STEPS + 1):
        targets.append(price_to_predict.shift(-i))
    target_df = pd.concat(targets, axis=1)
    target_df.columns = [f'target_{i}' for i in range(OUTPUT_STEPS)]
    main_df = pd.concat([main_df, target_df], axis=1)
    main_df.dropna(inplace=True)
    X = main_df[feature_names].values
    y = main_df[[f'target_{i}' for i in range(OUTPUT_STEPS)]].values
    
    # This scaler is for input features
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    # This scaler is ONLY for the target prices
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y)
    
    print("Building sequences...")
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        y_seq.append(y_scaled[i + SEQ_LEN - 1])
        
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_seq), dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Created {len(X_tensor)} sequences.")
    model = LSTMRegressor(input_size=INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_steps=OUTPUT_STEPS)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"\n--- Starting REGRESSION Model Training (Predicting {OUTPUT_STEPS} Steps) ---")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (xb, yb) in enumerate(loader):
            optimizer.zero_grad(); pred = model(xb); loss = loss_fn(pred, yb); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Average MSE Loss: {epoch_loss/len(loader):.8f}")
    print("\n--- Training Complete ---")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    MODEL_FILE = os.path.join(MODEL_SAVE_PATH, "lstm_model_regression.pth")
    SCALER_FILE_TARGET = os.path.join(MODEL_SAVE_PATH, "scaler_regression.pkl")
    SCALER_FILE_FEATURE = os.path.join(MODEL_SAVE_PATH, "scaler.pkl") # Used by both models
    
    torch.save({"model_state": model.state_dict()}, MODEL_FILE)
    print(f"(+) Regression model saved to {MODEL_FILE}")
    joblib.dump(target_scaler, SCALER_FILE_TARGET)
    print(f"(+) Regression TARGET scaler saved to {SCALER_FILE_TARGET}")
    joblib.dump(feature_scaler, SCALER_FILE_FEATURE)
    print(f"(+) FEATURE scaler saved to {SCALER_FILE_FEATURE}")