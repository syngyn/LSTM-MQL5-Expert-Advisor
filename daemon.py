import os
import json
import time
import torch
import joblib
import numpy as np
import traceback
from torch import nn
from datetime import datetime
import sys

# --- 1. CONFIGURATION (with Auto-Detection) ---
def find_mql5_files_path():
    appdata = os.getenv('APPDATA')
    if not appdata or 'win' not in sys.platform: return None
    metaquotes_path = os.path.join(appdata, 'MetaQuotes', 'Terminal')
    if not os.path.isdir(metaquotes_path): return None
    for entry in os.listdir(metaquotes_path):
        terminal_path = os.path.join(metaquotes_path, entry)
        if os.path.isdir(terminal_path) and len(entry) > 30 and all(c in '0123456789ABCDEF' for c in entry.upper()):
            mql5_files_path = os.path.join(terminal_path, 'MQL5', 'Files')
            if os.path.isdir(mql5_files_path): return mql5_files_path
    return None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
COMM_DIR_BASE = find_mql5_files_path() or SCRIPT_DIR
DATA_DIR = os.path.join(COMM_DIR_BASE, "LSTM_Trading", "data")
print(f"--> Using Model Path: {MODEL_DIR}")
print(f"--> Using Communication Path: {DATA_DIR}")

MODEL_FILE_CLASSIFICATION = os.path.join(MODEL_DIR, "lstm_model.pth")
SCALER_FILE_FEATURE = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_FILE_REGRESSION = os.path.join(MODEL_DIR, "lstm_model_regression.pth")
SCALER_FILE_REGRESSION_TARGET = os.path.join(MODEL_DIR, "scaler_regression.pkl")

INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 12, 128, 2, 20
POLL_INTERVAL = 0.1
NUM_CLASSES = 3
# --- CHANGE: Increased prediction horizon from 5 to 24 bars ---
OUTPUT_STEPS = 24

# --- 2. Model Definitions ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x); out = out[:, -1, :]; return self.fc(out)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_steps):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_steps)
    def forward(self, x):
        out, _ = self.lstm(x); out = out[:, -1, :]; return self.fc(out)

# --- 3. Daemon Class ---
class LSTMDaemon:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model_classifier, self.model_regressor = None, None
        self.scaler_feature, self.scaler_regressor_target = None, None
        self._load_models()

    def _load_models(self):
        print("Loading models and scalers...")
        try:
            # Load common feature scaler
            self.scaler_feature = joblib.load(SCALER_FILE_FEATURE)
            print(f"✓ Feature scaler loaded successfully.")
            
            # Load Classifier
            self.model_classifier = LSTMClassifier(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
            checkpoint_c = torch.load(MODEL_FILE_CLASSIFICATION, map_location=self.device)
            self.model_classifier.load_state_dict(checkpoint_c['model_state'])
            self.model_classifier.to(self.device).eval()
            print(f"✓ LSTM classification model loaded successfully.")

            # Load Regressor
            self.model_regressor = LSTMRegressor(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_STEPS)
            checkpoint_r = torch.load(MODEL_FILE_REGRESSION, map_location=self.device)
            self.model_regressor.load_state_dict(checkpoint_r['model_state'])
            self.model_regressor.to(self.device).eval()
            self.scaler_regressor_target = joblib.load(SCALER_FILE_REGRESSION_TARGET)
            print(f"✓ LSTM regression model and target scaler loaded successfully.")
        except Exception as e:
            print(f"FATAL: Could not load models/scalers: {e}. The daemon cannot run.")
            traceback.print_exc()
            sys.exit(1)

    def _get_classification_prediction(self, features: list) -> tuple:
        if not self.model_classifier or not self.scaler_feature: raise RuntimeError("Classification model/scaler not loaded.")
        arr = np.array(features, dtype=np.float32).reshape(1, SEQ_LEN, INPUT_FEATURES)
        scaled_features = self.scaler_feature.transform(arr.reshape(-1, INPUT_FEATURES))
        scaled_sequence = scaled_features.reshape(1, SEQ_LEN, INPUT_FEATURES)
        tensor = torch.tensor(scaled_sequence, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model_classifier(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            return probabilities[0].item(), probabilities[1].item(), probabilities[2].item()

    def _get_regression_prediction(self, features: list) -> list:
        if not self.model_regressor or not self.scaler_feature or not self.scaler_regressor_target: raise RuntimeError("Regression model/scalers not loaded.")
        arr = np.array(features, dtype=np.float32).reshape(1, SEQ_LEN, INPUT_FEATURES)
        scaled_features = self.scaler_feature.transform(arr.reshape(-1, INPUT_FEATURES))
        scaled_sequence = scaled_features.reshape(1, SEQ_LEN, INPUT_FEATURES)
        tensor = torch.tensor(scaled_sequence, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            scaled_predictions = self.model_regressor(tensor)[0].numpy()
        unscaled_predictions = self.scaler_regressor_target.inverse_transform(scaled_predictions.reshape(1, -1))
        return unscaled_predictions[0].tolist()

    def _handle_request(self, filepath: str):
        request_id, response = "unknown", {}
        try:
            with open(filepath, 'r') as f: data = json.load(f)
            request_id = data.get("request_id", os.path.basename(filepath))
            action, features = data.get("action"), data.get("features")
            if not all([action, features]): raise ValueError("Request JSON missing 'action' or 'features'.")

            if action == "predict_classification":
                sell_prob, hold_prob, buy_prob = self._get_classification_prediction(features)
                response = {"request_id": request_id, "status": "success", "sell_probability": sell_prob, "hold_probability": hold_prob, "buy_probability": buy_prob}
            elif action == "predict_regression":
                predicted_prices = self._get_regression_prediction(features)
                response = {"request_id": request_id, "status": "success", "predicted_prices": predicted_prices}
            else:
                raise ValueError(f"Unknown action: '{action}'")

        except Exception as e:
            print(f"⚠ ERROR processing {request_id}: {e}"); traceback.print_exc()
            response = {"request_id": request_id, "status": "error", "message": str(e)}

        resp_path_tmp = os.path.join(DATA_DIR, f"response_{request_id}.tmp")
        resp_path_final = os.path.join(DATA_DIR, f"response_{request_id}.json")
        with open(resp_path_tmp, 'w') as f: json.dump(response, f, indent=2)
        os.rename(resp_path_tmp, resp_path_final)

    def run(self):
        print("\n--- Dual-Mode LSTM Daemon is running. ---")
        while True:
            try:
                for fname in os.listdir(DATA_DIR):
                    if fname.startswith("request_") and fname.endswith(".json"):
                        fpath = os.path.join(DATA_DIR, fname)
                        self._handle_request(fpath)
                        try: os.remove(fpath)
                        except OSError as e: print(f"Warning: Could not remove request file {fpath}: {e}")
                time.sleep(POLL_INTERVAL)
            except KeyboardInterrupt: print("\nDaemon shutting down."); break
            except Exception: print("An error occurred in the main loop. Restarting watch..."); traceback.print_exc(); time.sleep(5)

# --- Main Execution Block ---
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    daemon = LSTMDaemon()
    daemon.run()