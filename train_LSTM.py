
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = SCRIPT_DIR
REQUIRED_FILES = {"EURUSD": "EURUSD60.csv", "EURJPY": "EURJPY60.csv", "USDJPY": "USDJPY60.csv", "GBPUSD": "GBPUSD60.csv", "EURGBP": "EURGBP60.csv", "USDCAD": "USDCAD60.csv", "USDCHF": "USDCHF60.csv"}
INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 12, 128, 2, 20
NUM_CLASSES, LOOKAHEAD_BARS, PROFIT_THRESHOLD_ATR = 3, 5, 0.75
EPOCHS, BATCH_SIZE, LEARNING_RATE = 20, 64, 0.001

def load_and_align_data(file_map):
    print("Loading and aligning all currency data sources...")
    all_dfs = {}
    file_column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    for symbol, filename in file_map.items():
        full_path = os.path.join(DATA_DIR, filename)
        print(f"Attempting to load: {full_path}")
        try:
            df = pd.read_csv(full_path, sep='\t', header=0, names=file_column_names, skiprows=1, quotechar='"', encoding='utf-8-sig')
            for col in ['open', 'high', 'low', 'close', 'tickvol']: df[col] = pd.to_numeric(df[col], errors='coerce')
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            df.dropna(subset=['datetime', 'open', 'high', 'low', 'close', 'tickvol'], inplace=True)
            df.drop_duplicates(subset='datetime', keep='first', inplace=True)
            df.set_index('datetime', inplace=True)
            all_dfs[symbol] = df[['high', 'low', 'close', 'tickvol']].rename(columns={'high': f'{symbol}_high', 'low': f'{symbol}_low', 'close': f'{symbol}_close', 'tickvol': f'{symbol}_vol'})
        except FileNotFoundError:
            print(f"FATAL ERROR: Data file '{filename}' not found at path '{full_path}'."); sys.exit(2)
        except Exception as e:
            print(f"FATAL ERROR processing {filename}: {e}"); traceback.print_exc(); sys.exit(1)
    master_df = pd.concat(all_dfs.values(), axis=1, join='inner')
    master_df.ffill(inplace=True); master_df.dropna(inplace=True)
    print(f"Loaded and aligned {len(master_df)} data points.")
    return master_df

def create_features(df):
    print("Creating comprehensive feature set...")
    for col in df.columns:
        if 'close' in col: df[f'{col}_return'] = df[col].pct_change()
    df['eurusd_return'] = df['EURUSD_close_return']
    df['eurusd_volume'] = df['EURUSD_vol']
    df['eurusd_atr'] = (df['EURUSD_high'] - df['EURUSD_low']).rolling(14).mean()
    ema_fast = df['EURUSD_close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['EURUSD_close'].ewm(span=26, adjust=False).mean()
    df['eurusd_macd'] = ema_fast - ema_slow
    delta = df['EURUSD_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['eurusd_rsi'] = 100 - (100 / (1 + rs))
    low_14 = df['EURUSD_low'].rolling(window=14).min()
    high_14 = df['EURUSD_high'].rolling(window=14).max()
    df['eurusd_stoch'] = 100 * ((df['EURUSD_close'] - low_14) / (high_14 + 1e-10))
    tp = (df['EURUSD_high'] + df['EURUSD_low'] + df['EURUSD_close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['eurusd_cci'] = (tp - sma_tp) / (0.015 * (mad + 1e-10))
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['USD_Index_Proxy'] = (df['USDJPY_close_return'] + df['USDCAD_close_return'] + df['USDCHF_close_return']) - (df['EURUSD_close_return'] + df['GBPUSD_close_return'])
    df['EUR_Index_Proxy'] = df['EURUSD_close_return'] + df['EURJPY_close_return'] + df['EURGBP_close_return']
    df['JPY_Index_Proxy'] = -(df['EURJPY_close_return'] + df['USDJPY_close_return'])
    feature_list = ['eurusd_return', 'eurusd_volume', 'eurusd_atr', 'eurusd_macd', 'eurusd_rsi', 'eurusd_stoch', 'eurusd_cci', 'hour_of_day', 'day_of_week', 'USD_Index_Proxy', 'EUR_Index_Proxy', 'JPY_Index_Proxy']
    df.replace([np.inf, -np.inf], np.nan, inplace=True); df.ffill(inplace=True); df.dropna(inplace=True)
    print(f"Data size after feature creation and dropna: {len(df)}")
    return df, feature_list

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x); out = out[:, -1, :]; return self.fc(out)

if __name__ == "__main__":
    main_df, feature_names = create_features(load_and_align_data(REQUIRED_FILES))
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