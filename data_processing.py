import os
import sys
import traceback
import pandas as pd
import numpy as np

def load_and_align_data(file_map, data_dir):
    """
    Loads and aligns all currency data sources from a specified directory.
    This function is now centralized to be used by multiple training scripts.
    """
    print("Loading and aligning all currency data sources...")
    all_dfs = {}
    file_column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    for symbol, filename in file_map.items():
        full_path = os.path.join(data_dir, filename)
        print(f"Attempting to load: {full_path}")
        try:
            df = pd.read_csv(full_path, sep='\t', header=0, names=file_column_names, skiprows=1, quotechar='"', encoding='utf-8-sig')
            for col in ['open', 'high', 'low', 'close', 'tickvol']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            df.dropna(subset=['datetime', 'open', 'high', 'low', 'close', 'tickvol'], inplace=True)
            df.drop_duplicates(subset='datetime', keep='first', inplace=True)
            df.set_index('datetime', inplace=True)
            all_dfs[symbol] = df[['high', 'low', 'close', 'tickvol']].rename(
                columns={
                    'high': f'{symbol}_high', 'low': f'{symbol}_low',
                    'close': f'{symbol}_close', 'tickvol': f'{symbol}_vol'
                }
            )
        except FileNotFoundError:
            print(f"FATAL ERROR: Data file '{filename}' not found at path '{full_path}'.")
            sys.exit(2)
        except Exception as e:
            print(f"FATAL ERROR processing {filename}: {e}")
            traceback.print_exc()
            sys.exit(1)

    master_df = pd.concat(all_dfs.values(), axis=1, join='inner')
    master_df.ffill(inplace=True)
    master_df.dropna(inplace=True)
    print(f"Loaded and aligned {len(master_df)} data points.")
    return master_df

def create_features(df):
    """
    Creates a comprehensive feature set for the model.
    This function is now centralized to be used by multiple training scripts.
    """
    print("Creating comprehensive feature set...")
    for col in df.columns:
        if 'close' in col:
            df[f'{col}_return'] = df[col].pct_change()

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
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    print(f"Data size after feature creation and dropna: {len(df)}")
    return df, feature_list