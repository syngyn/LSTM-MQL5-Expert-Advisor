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
from pykalman import KalmanFilter

# --- Configuration ---
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

# --- Model File Paths ---
SCALER_FILE_FEATURE = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_FILE_REGRESSION = os.path.join(MODEL_DIR, "lstm_model_regression.pth")
SCALER_FILE_REGRESSION_TARGET = os.path.join(MODEL_DIR, "scaler_regression.pkl")

# --- Constants ---
INPUT_FEATURES = 15
HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 128, 2, 20
POLL_INTERVAL = 0.1
OUTPUT_STEPS = 24

# --- Model Definition ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_steps):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_steps)
    def forward(self, x):
        out, _ = self.lstm(x); out = out[:, -1, :]; return self.fc(out)

# --- Daemon Class with Advanced Features ---
class LSTMDaemon:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model_regressor, self.scaler_feature, self.scaler_regressor_target = None, None, None
        self._load_models()

    def _load_models(self):
        print("Loading models and scalers...")
        try:
            self.scaler_feature = joblib.load(SCALER_FILE_FEATURE)
            print(f"✓ Feature scaler loaded successfully.")
            
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

    def _apply_kalman_filter(self, prices):
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=prices[0],
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=0.01)
        state_means, _ = kf.filter(prices)
        return state_means.flatten().tolist()

    def _calculate_confidence_score(self, prices, current_price, atr):
        if atr is None or atr <= 1e-6: return 0.0
        price_changes = np.diff(np.insert(prices, 0, current_price))
        predicted_std_dev = np.std(price_changes)
        confidence = predicted_std_dev / atr
        return np.clip(confidence, 0.0, 2.0) / 2.0

    def _get_regression_prediction(self, features: list, current_price: float, atr: float) -> dict:
        if not all([self.model_regressor, self.scaler_feature, self.scaler_regressor_target]):
            raise RuntimeError("Regression model or its scalers are not loaded.")
        
        arr = np.array(features, dtype=np.float32).reshape(1, SEQ_LEN, INPUT_FEATURES)
        scaled_features = self.scaler_feature.transform(arr.reshape(-1, INPUT_FEATURES))
        scaled_sequence = scaled_features.reshape(1, SEQ_LEN, INPUT_FEATURES)
        tensor = torch.tensor(scaled_sequence, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            scaled_predictions = self.model_regressor(tensor)[0].numpy()
        
        unscaled_predictions = self.scaler_regressor_target.inverse_transform(scaled_predictions.reshape(1, -1))[0]
        smoothed_prices = self._apply_kalman_filter(unscaled_predictions)
        confidence = self._calculate_confidence_score(smoothed_prices, current_price, atr)
        
        return {
            "predicted_prices": smoothed_prices,
            "confidence_score": confidence
        }

    def _handle_request(self, filepath: str):
        request_id, response = "unknown", {}
        try:
            with open(filepath, 'r') as f: data = json.load(f)
            request_id = data.get("request_id", os.path.basename(filepath))
            action = data.get("action")
            features = data.get("features")
            current_price = data.get("current_price")
            atr = data.get("atr")

            if not all([action, features, isinstance(current_price, float), isinstance(atr, float)]):
                raise ValueError("Request JSON missing/invalid keys (action, features, current_price, atr).")

            if action == "predict_regression":
                prediction_result = self._get_regression_prediction(features, current_price, atr)
                response = {
                    "request_id": request_id,
                    "status": "success",
                    "predicted_prices": prediction_result["predicted_prices"],
                    "confidence_score": prediction_result["confidence_score"]
                }
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Responded to {request_id} (Conf: {prediction_result['confidence_score']:.2f})")
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
        print("\n--- Advanced LSTM Daemon is running. ---")
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

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    daemon = LSTMDaemon()
    daemon.run()