import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib
from torch import nn
import traceback

# --- CONFIGURATION ---
# Ensure these paths and settings match your other scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = SCRIPT_DIR

# --- OUTPUT FILE ---
# This is the file the MQL5 EA will read during backtesting
PREDICTIONS_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "backtest_predictions.csv")

# --- MODEL AND FEATURE DEFINITIONS (Must be identical to training/daemon) ---
from train_LSTM import LSTMClassifier, load_and_align_data, create_features
from train_regression import LSTMRegressor

# Hyperparameters (must match)
INPUT_FEATURES = 12
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQ_LEN = 20
NUM_CLASSES = 3
OUTPUT_STEPS = 5

def generate_predictions():
    """
    Loads all data, runs models over it sequentially, and saves all predictions to a file.
    """
    print("--- Starting Offline Prediction Generation for Backtesting ---")

    # --- 1. Load Models and Scalers ---
    print("\n[1/5] Loading trained models and scalers...")
    device = torch.device("cpu")
    try:
        # Classification Model
        model_classifier = LSTMClassifier(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
        checkpoint_c = torch.load(os.path.join(MODEL_DIR, "lstm_model.pth"), map_location=device)
        model_classifier.load_state_dict(checkpoint_c['model_state'])
        model_classifier.to(device).eval()
        scaler_classifier = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        print("✓ Classification model loaded.")
    except Exception as e:
        print(f"✗ ERROR: Could not load classification model: {e}"); return

    try:
        # Regression Model
        model_regressor = LSTMRegressor(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_STEPS)
        checkpoint_r = torch.load(os.path.join(MODEL_DIR, "lstm_model_regression.pth"), map_location=device)
        model_regressor.load_state_dict(checkpoint_r['model_state'])
        model_regressor.to(device).eval()
        scaler_regressor = joblib.load(os.path.join(MODEL_DIR, "scaler_regression.pkl"))
        print("✓ Regression model loaded.")
    except Exception as e:
        print(f"✗ ERROR: Could not load regression model: {e}"); return

    # --- 2. Load and Prepare All Historical Data ---
    print("\n[2/5] Loading and preparing historical data...")
    from train_LSTM import REQUIRED_FILES
    try:
        main_df, feature_names = create_features(load_and_align_data(REQUIRED_FILES))
        if len(main_df) < SEQ_LEN:
            print(f"✗ ERROR: Not enough data ({len(main_df)} rows) for a single sequence."); return
    except Exception as e:
        print(f"✗ ERROR: Failed during data loading/feature creation: {e}"); return
    
    features_data = main_df[feature_names].values

    # --- 3. Generate Predictions for Every Possible Bar ---
    print(f"\n[3/5] Generating predictions for {len(features_data) - SEQ_LEN} bars. This may take a while...")
    all_predictions = []
    
    with torch.no_grad():
        for i in range(len(features_data) - SEQ_LEN):
            # Get the sequence of features
            feature_sequence = features_data[i : i + SEQ_LEN]
            
            # The prediction is for the bar at the END of the sequence
            timestamp = main_df.index[i + SEQ_LEN - 1]

            # --- Classification Prediction ---
            scaled_seq_c = scaler_classifier.transform(feature_sequence)
            tensor_c = torch.tensor(scaled_seq_c, dtype=torch.float32).reshape(1, SEQ_LEN, INPUT_FEATURES).to(device)
            logits = model_classifier(tensor_c)
            probabilities = torch.softmax(logits, dim=1)[0]
            sell_prob, hold_prob, buy_prob = probabilities[0].item(), probabilities[1].item(), probabilities[2].item()

            # --- Regression Prediction ---
            scaled_seq_r = scaler_regressor.transform(feature_sequence)
            tensor_r = torch.tensor(scaled_seq_r, dtype=torch.float32).reshape(1, SEQ_LEN, INPUT_FEATURES).to(device)
            scaled_preds = model_regressor(tensor_r)[0].numpy()
            
            dummy_array = np.zeros((OUTPUT_STEPS, INPUT_FEATURES))
            dummy_array[:, 0] = scaled_preds
            unscaled_predictions = scaler_regressor.inverse_transform(dummy_array)
            predicted_prices = unscaled_predictions[:, 0].tolist()

            # Store the results
            result_row = {
                "timestamp": timestamp.strftime('%Y.%m.%d %H:%M:%S'),
                "buy_prob": buy_prob,
                "sell_prob": sell_prob,
                "hold_prob": hold_prob
            }
            for j in range(OUTPUT_STEPS):
                result_row[f'pred_price_{j}'] = predicted_prices[j]
            
            all_predictions.append(result_row)

            if (i + 1) % 1000 == 0:
                print(f"  ... processed {i + 1} bars ...")

    print("\n[4/5] Assembling predictions into a final DataFrame...")
    predictions_df = pd.DataFrame(all_predictions)

    # --- 5. Save to CSV ---
    print(f"\n[5/5] Saving predictions to '{PREDICTIONS_OUTPUT_FILE}'...")
    predictions_df.to_csv(PREDICTIONS_OUTPUT_FILE, index=False, float_format='%.8f')

    print("\n--- ✓ Generation Complete! ---")
    print(f"You can now run the Strategy Tester. It will read from '{os.path.basename(PREDICTIONS_OUTPUT_FILE)}'.")


if __name__ == "__main__":
    generate_predictions()