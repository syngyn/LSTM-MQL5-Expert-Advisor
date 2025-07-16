import os
import sys
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# --- IMPORTANT: Ensure these match your training scripts ---
from data_processing import load_and_align_data, create_features
from train_LSTM import LSTMClassifier
from train_regression import LSTMRegressor

# --- Configuration ---
print("--- Starting Backtest Data Generation ---")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = SCRIPT_DIR

# --- MODIFIED: Save the file locally in the same folder as the script ---
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'backtest_predictions.csv')
# --- END OF MODIFICATION ---

# --- Model Hyperparameters (Must match training scripts) ---
INPUT_FEATURES = 12
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQ_LEN = 20
NUM_CLASSES = 3
OUTPUT_STEPS = 24

# --- Data File Configuration ---
REQUIRED_FILES = {
    "EURUSD": "EURUSD60.csv", "EURJPY": "EURJPY60.csv", "USDJPY": "USDJPY60.csv",
    "GBPUSD": "GBPUSD60.csv", "EURGBP": "EURGBP60.csv", "USDCAD": "USDCAD60.csv",
    "USDCHF": "USDCHF60.csv"
}

def generate_predictions():
    # 1. Load Models and Scalers
    print("Loading models and scalers...")
    try:
        device = torch.device("cpu")
        model_classifier = LSTMClassifier(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
        checkpoint_c = torch.load(os.path.join(MODEL_DIR, "lstm_model.pth"), map_location=device)
        model_classifier.load_state_dict(checkpoint_c['model_state'])
        model_classifier.eval()
        model_regressor = LSTMRegressor(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_STEPS).to(device)
        checkpoint_r = torch.load(os.path.join(MODEL_DIR, "lstm_model_regression.pth"), map_location=device)
        model_regressor.load_state_dict(checkpoint_r['model_state'])
        model_regressor.eval()
        scaler_feature = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        scaler_target = joblib.load(os.path.join(MODEL_DIR, "scaler_regression.pkl"))
        print("✓ Models and scalers loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load models/scalers. Have you trained them first? Error: {e}")
        sys.exit(1)

    # 2. Load and Process Data
    print("Loading and processing historical data...")
    main_df, feature_names = create_features(load_and_align_data(REQUIRED_FILES, DATA_DIR))
    X = main_df[feature_names].values
    X_scaled = scaler_feature.transform(X)

    # 3. Build Sequences
    print(f"Building sequences for {len(main_df)} data points...")
    X_seq, y_indices = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        y_indices.append(i + SEQ_LEN)
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)
    print(f"✓ Created {len(X_tensor)} sequences.")

    # 4. Get Predictions
    print("Generating predictions for all sequences...")
    with torch.no_grad():
        logits = model_classifier(X_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        scaled_regr_preds = model_regressor(X_tensor).cpu().numpy()
    unscaled_regr_preds = scaler_target.inverse_transform(scaled_regr_preds)
    print("✓ Predictions generated.")

    # 5. Prepare results for CSV
    print("Formatting results for CSV export...")
    results = []
    header = "timestamp;buy_prob;sell_prob;hold_prob;" + ";".join([f"pred_price_{i+1}" for i in range(OUTPUT_STEPS)])
    for i in range(len(X_tensor)):
        idx = y_indices[i]
        timestamp = main_df.index[idx].strftime('%Y.%m.%d %H:%M:%S')
        sell_prob, hold_prob, buy_prob = probabilities[i][0], probabilities[i][1], probabilities[i][2]
        price_preds = unscaled_regr_preds[i]
        row = [timestamp, f"{buy_prob:.6f}", f"{sell_prob:.6f}", f"{hold_prob:.6f}"]
        row.extend([f"{p:.5f}" for p in price_preds])
        results.append(";".join(row))
        
    # 6. Write to CSV file
    print(f"\n>>> Writing {len(results)} predictions to the LOCAL project folder <<<")
    print(f"    {OUTPUT_FILE}")
    print("----------------------------------------------------------------------")
    try:
        with open(OUTPUT_FILE, 'w') as f:
            f.write(header + '\n')
            f.write('\n'.join(results))
        print("✓ Backtest prediction file generated successfully!")
        print("\n>>> NEXT STEP: Manually move this file to your MQL5 Common\\Files folder. <<<")
    except Exception as e:
        print(f"FATAL: Could not write to local file. Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_predictions()