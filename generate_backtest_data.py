import os
import sys
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pykalman import KalmanFilter

from data_processing import load_and_align_data, create_features

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_steps):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_steps)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
        
# --- Configuration ---
if len(sys.argv) < 2:
    print("FATAL: Please provide a symbol as a command-line argument.")
    print("Example: python generate_backtest_data.py EURUSD")
    sys.exit(1)

SYMBOL_TO_PROCESS = sys.argv[1].upper()
print(f"--- Starting Advanced Backtest Data Generation for {SYMBOL_TO_PROCESS} ---")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = SCRIPT_DIR
OUTPUT_FILE_LOCAL = os.path.join(SCRIPT_DIR, 'backtest_predictions.csv')

# --- Model Hyperparameters ---
INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 15, 128, 2, 20
OUTPUT_STEPS = 24

# --- Data File Configuration ---
REQUIRED_FILES = {
    "EURUSD": "EURUSD60.csv", "EURJPY": "EURJPY60.csv", "USDJPY": "USDJPY60.csv",
    "GBPUSD": "GBPUSD60.csv", "EURGBP": "EURGBP60.csv", "USDCAD": "USDCAD60.csv",
    "USDCHF": "USDCHF60.csv"
}

def apply_kalman_filter(prices):
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                      initial_state_mean=prices[0], initial_state_covariance=1,
                      observation_covariance=1, transition_covariance=0.01)
    state_means, _ = kf.filter(prices)
    return state_means.flatten()

def calculate_confidence_score(prices, current_price, atr):
    if atr is None or atr <= 1e-6: return 0.0
    price_changes = np.diff(np.insert(prices, 0, current_price))
    predicted_std_dev = np.std(price_changes)
    confidence = predicted_std_dev / atr
    return np.clip(confidence, 0.0, 2.0) / 2.0

def generate_predictions():
    print("Loading models and scalers...")
    try:
        device = torch.device("cpu")
        model_regressor = LSTMRegressor(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_STEPS).to(device)
        checkpoint_r = torch.load(os.path.join(MODEL_DIR, "lstm_model_regression.pth"), map_location=device)
        model_regressor.load_state_dict(checkpoint_r['model_state'])
        model_regressor.eval()
        scaler_feature = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        scaler_target = joblib.load(os.path.join(MODEL_DIR, "scaler_regression.pkl"))
        print("✓ Models and scalers loaded.")
    except Exception as e:
        print(f"FATAL: Could not load models/scalers. Error: {e}"); sys.exit(1)

    print("Loading and processing historical data...")
    main_df, feature_names = create_features(load_and_align_data(REQUIRED_FILES, DATA_DIR))
    X = main_df[feature_names].values
    X_scaled = scaler_feature.transform(X)

    print(f"Building sequences...")
    X_seq, y_indices = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        y_indices.append(i + SEQ_LEN)
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)
    print(f"✓ Created {len(X_tensor)} sequences.")

    print("Generating predictions...")
    with torch.no_grad():
        scaled_regr_preds = model_regressor(X_tensor).cpu().numpy()
    unscaled_regr_preds = scaler_target.inverse_transform(scaled_regr_preds)
    print("✓ Predictions generated.")

    print("Formatting results for CSV export...")
    results = []
    header = "timestamp;buy_prob;sell_prob;hold_prob;confidence_score;" + ";".join([f"pred_price_{i+1}" for i in range(OUTPUT_STEPS)])
    
    for i in range(len(X_tensor)):
        idx = y_indices[i]
        timestamp = main_df.index[idx].strftime('%Y.%m.%d %H:%M:%S')
        
        buy_prob, sell_prob, hold_prob = 0.5, 0.5, 0.0 # Placeholder
        
        raw_prices = unscaled_regr_preds[i]
        smoothed_prices = apply_kalman_filter(raw_prices)

        current_price = main_df['EURUSD_close'].iloc[idx]
        current_atr = main_df['eurusd_atr'].iloc[idx]
        
        confidence = calculate_confidence_score(smoothed_prices, current_price, current_atr)

        row = [timestamp, f"{buy_prob:.6f}", f"{sell_prob:.6f}", f"{hold_prob:.6f}", f"{confidence:.4f}"]
        row.extend([f"{p:.5f}" for p in smoothed_prices])
        results.append(";".join(row))
        
    print(f"\n>>> Writing {len(results)} predictions to the LOCAL project folder <<<")
    print(f"    {OUTPUT_FILE_LOCAL}")
    try:
        with open(OUTPUT_FILE_LOCAL, 'w') as f:
            f.write(header + '\n')
            f.write('\n'.join(results))
        print("✓ Backtest prediction file generated successfully!")
        print("\n>>> NEXT STEP: Manually move this file to your MQL5 Common\\Files folder and rename it to 'backtest_predictions.csv'. <<<")
    except Exception as e:
        print(f"FATAL: Could not write to local file. Error: {e}"); sys.exit(1)

if __name__ == "__main__":
    generate_predictions()