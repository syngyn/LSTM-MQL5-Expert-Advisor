import os
import json
import time
import torch
import joblib
import numpy as np
import traceback
from torch import nn
from datetime import datetime
import sys  # --- FIX: Import the 'sys' module at the top of the script. ---

# --- 1. CONFIGURATION (with Auto-Detection) ---

def find_mql5_files_path():
    """
    Attempts to automatically find the MQL5/Files directory for the first detected
    MetaTrader 5 terminal instance. This is much more reliable than relative paths.
    """
    # The path is typically C:\Users\<user>\AppData\Roaming\MetaQuotes\Terminal\<hash>\MQL5\Files
    appdata = os.getenv('APPDATA')
    if not appdata or 'win' not in sys.platform:
        # Return None if not on Windows or APPDATA is not set
        return None

    metaquotes_path = os.path.join(appdata, 'MetaQuotes', 'Terminal')
    if not os.path.isdir(metaquotes_path):
        return None  # MetaTrader not installed in the default location

    # Find the first terminal instance (folder with a long hexadecimal name)
    for entry in os.listdir(metaquotes_path):
        terminal_path = os.path.join(metaquotes_path, entry)
        # Check if it's a directory and the name looks like a terminal hash
        if os.path.isdir(terminal_path) and len(entry) > 30 and all(c in '0123456789ABCDEF' for c in entry.upper()):
            mql5_files_path = os.path.join(terminal_path, 'MQL5', 'Files')
            if os.path.isdir(mql5_files_path):
                return mql5_files_path  # Return the first one found
    return None

# Use the script's own location for robust model pathing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

# --- CRITICAL: Determine the communication directory ---
# The MQL5 EA writes files to its sandboxed data folder. We try to find it.
COMM_DIR_BASE = find_mql5_files_path()

if COMM_DIR_BASE:
    print(f"✓ Automatically detected MQL5 Files folder: {COMM_DIR_BASE}")
else:
    print("⚠ Could not automatically find MQL5 Files folder.")
    print("  Defaulting to a path relative to the script. This may not work.")
    print("  If communication fails, ensure your Python project is inside the terminal's 'MQL5/Files' directory.")
    # Fallback to the old method if auto-detection fails
    COMM_DIR_BASE = SCRIPT_DIR

# The MQL5 EA is configured to use a subfolder "LSTM_Trading\data".
# We construct the final path based on our detected or fallback base path.
DATA_DIR = os.path.join(COMM_DIR_BASE, "LSTM_Trading", "data")

print(f"--> Using Model Path: {MODEL_DIR}")
print(f"--> Using Communication Path: {DATA_DIR}")


# Classification Model (for trading signals)
MODEL_FILE_CLASSIFICATION = os.path.join(MODEL_DIR, "lstm_model.pth")
SCALER_FILE_CLASSIFICATION = os.path.join(MODEL_DIR, "scaler.pkl")

# Regression Model (for price prediction)
MODEL_FILE_REGRESSION = os.path.join(MODEL_DIR, "lstm_model_regression.pth")
SCALER_FILE_REGRESSION = os.path.join(MODEL_DIR, "scaler_regression.pkl")

# Shared Hyperparameters
INPUT_FEATURES = 12
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQ_LEN = 20
POLL_INTERVAL = 0.1

# Specific Hyperparameters
NUM_CLASSES = 3      # For classification
OUTPUT_STEPS = 5     # For regression (predicting 5 bars ahead)

# --- 2. Model Definitions (Must be identical to training scripts) ---
class LSTMClassifier(nn.Module):
    """The LSTM model architecture for classification."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Use last time step
        return self.fc(out)

class LSTMRegressor(nn.Module):
    """The LSTM model architecture for multi-step price regression."""
    def __init__(self, input_size, hidden_size, num_layers, output_steps):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Use last time step
        return self.fc(out)


# --- 3. Daemon Class ---
class LSTMDaemon:
    """The main daemon class that handles communication and prediction for both modes."""
    def __init__(self):
        self.device = torch.device("cpu")
        
        self.model_classifier = None
        self.scaler_classifier = None
        self.model_regressor = None
        self.scaler_regressor = None
        
        self._load_classifier()
        self._load_regressor()

    def _load_classifier(self):
        print("Loading CLASSIFICATION model and scaler...")
        try:
            self.model_classifier = LSTMClassifier(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
            checkpoint = torch.load(MODEL_FILE_CLASSIFICATION, map_location=self.device)
            self.model_classifier.load_state_dict(checkpoint['model_state'])
            self.model_classifier.to(self.device).eval()
            self.scaler_classifier = joblib.load(SCALER_FILE_CLASSIFICATION)
            print(f"✓ LSTM classification model and scaler loaded successfully.")
        except Exception as e:
            print(f"WARNING: Could not load CLASSIFICATION model/scaler: {e}. Trading will be unavailable.")

    def _load_regressor(self):
        print("Loading REGRESSION model and scaler...")
        try:
            self.model_regressor = LSTMRegressor(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_STEPS)
            checkpoint = torch.load(MODEL_FILE_REGRESSION, map_location=self.device)
            self.model_regressor.load_state_dict(checkpoint['model_state'])
            self.model_regressor.to(self.device).eval()
            self.scaler_regressor = joblib.load(SCALER_FILE_REGRESSION)
            print(f"✓ LSTM regression model and scaler loaded successfully.")
        except Exception as e:
            print(f"WARNING: Could not load REGRESSION model/scaler: {e}. Price prediction will be unavailable.")

    def _get_classification_prediction(self, features: list) -> tuple:
        """Processes features and returns probabilities for Sell, Hold, and Buy."""
        if not self.model_classifier or not self.scaler_classifier:
            raise RuntimeError("Classification model is not loaded.")
            
        arr = np.array(features, dtype=np.float32).reshape(1, SEQ_LEN, INPUT_FEATURES)
        scaled_features = self.scaler_classifier.transform(arr.reshape(-1, INPUT_FEATURES))
        scaled_sequence = scaled_features.reshape(1, SEQ_LEN, INPUT_FEATURES)
        
        tensor = torch.tensor(scaled_sequence, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model_classifier(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            return probabilities[0].item(), probabilities[1].item(), probabilities[2].item()

    def _get_regression_prediction(self, features: list) -> list:
        """Processes features and returns a list of 5 predicted prices."""
        if not self.model_regressor or not self.scaler_regressor:
            raise RuntimeError("Regression model is not loaded.")
        
        arr = np.array(features, dtype=np.float32).reshape(1, SEQ_LEN, INPUT_FEATURES)
        scaled_features = self.scaler_regressor.transform(arr.reshape(-1, INPUT_FEATURES))
        scaled_sequence = scaled_features.reshape(1, SEQ_LEN, INPUT_FEATURES)
        
        tensor = torch.tensor(scaled_sequence, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            scaled_predictions = self.model_regressor(tensor)[0].numpy()

        dummy_array = np.zeros((OUTPUT_STEPS, INPUT_FEATURES))
        dummy_array[:, 0] = scaled_predictions
        unscaled_predictions = self.scaler_regressor.inverse_transform(dummy_array)
        final_prices = unscaled_predictions[:, 0].tolist()
        
        return final_prices

    def _handle_request(self, filepath: str):
        """Processes a single request file and generates a response based on the 'action' key."""
        request_id = "unknown"
        response = {}
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            request_id = data.get("request_id", os.path.basename(filepath))
            features = data.get("features")
            action = data.get("action")
            
            if not features: raise ValueError("Request JSON is missing 'features' key.")
            if not action: raise ValueError("Request JSON is missing 'action' key.")

            if action == "predict_classification":
                sell_prob, hold_prob, buy_prob = self._get_classification_prediction(features)
                response = {
                    "request_id": request_id, "status": "success",
                    "sell_probability": sell_prob, "hold_probability": hold_prob, "buy_probability": buy_prob
                }
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Responded to {request_id} (Class): B={buy_prob:.2f} H={hold_prob:.2f} S={sell_prob:.2f}")

            elif action == "predict_regression":
                predicted_prices = self._get_regression_prediction(features)
                response = {
                    "request_id": request_id, "status": "success",
                    "predicted_prices": predicted_prices
                }
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Responded to {request_id} (Regr): {len(predicted_prices)} prices predicted.")

            else:
                raise ValueError(f"Unknown action specified: '{action}'")

        except Exception as e:
            print(f"⚠ ERROR processing request {request_id}: {e}")
            traceback.print_exc()
            response = {"request_id": request_id, "status": "error", "message": str(e)}

        resp_path_tmp = os.path.join(DATA_DIR, f"response_{request_id}.tmp")
        resp_path_final = os.path.join(DATA_DIR, f"response_{request_id}.json")
        with open(resp_path_tmp, 'w') as f:
            json.dump(response, f, indent=2)
        os.rename(resp_path_tmp, resp_path_final)

    def run(self):
        """Main daemon loop to watch for and process request files."""
        print("\n--- Dual-Mode LSTM Daemon is running. ---")
        while True:
            try:
                for fname in os.listdir(DATA_DIR):
                    if fname.startswith("request_") and fname.endswith(".json"):
                        fpath = os.path.join(DATA_DIR, fname)
                        self._handle_request(fpath)
                        try:
                            os.remove(fpath)
                        except OSError as e:
                            print(f"Warning: Could not remove request file {fpath}: {e}")
                time.sleep(POLL_INTERVAL)
            except KeyboardInterrupt:
                print("\nDaemon shutting down."); break
            except Exception:
                print("An error occurred in the main loop. Restarting watch...")
                traceback.print_exc()
                time.sleep(5)

# --- Main Execution Block ---
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("="*50 + "\n      Starting Dual-Mode LSTM Daemon\n" + "="*50)
    try:
        daemon = LSTMDaemon()
        daemon.run()
    except Exception as e:
        print(f"\nDaemon failed to start. Please check logs and ensure model files exist.")