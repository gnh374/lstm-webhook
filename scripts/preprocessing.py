import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import sys

def preprocess(df: pd.DataFrame, lookback_window: int, rolling_window: int):
    df[['transformed_mean_CPU_usage_rate']] = np.log1p(df[['mean_CPU_usage_rate']])
    # df[['transformed_canonical_memory_usage']] = np.log1p(df[['canonical_memory_usage']])
    
    # df['transformed_canonical_memory_usage'] = savgol_filter(df['transformed_canonical_memory_usage'], window_length=11, polyorder=6)
    df['transformed_mean_CPU_usage_rate'] = savgol_filter(df['transformed_mean_CPU_usage_rate'], window_length=11, polyorder=6)
    
    for i in range(1, lookback_window + 1):
        df[f'mean_CPU_usage_rate_-{i*2}min'] = df['mean_CPU_usage_rate'].shift(i)
        # df[f'canonical_memory_usage_-{i*2}min'] = df['canonical_memory_usage'].shift(i)

    df['rolling_mean_CPU_usage'] = df['mean_CPU_usage_rate'].rolling(center=False, window=rolling_window).mean()
    df['rolling_std_CPU_usage'] = df['mean_CPU_usage_rate'].rolling(center=False, window=rolling_window).std()

    # df['rolling_mean_memory_usage'] = df['canonical_memory_usage'].rolling(center=False, window=rolling_window).mean()
    # df['rolling_std_memory_usage'] = df['canonical_memory_usage'].rolling(center=False, window=rolling_window).std()

    # df.fillna(0, inplace=True)
    df = df.iloc[lookback_window:]
    return df

def split_dataset(df: pd.DataFrame):
    # Calculate the sizes for each split
    train_size = int(0.8 * len(df))
    test_size = len(df) - train_size

    train_set = df.iloc[:train_size]
    test_set = df.iloc[train_size:]
    
    return train_set, test_set
    

def scale(train_set: pd.DataFrame, features: list[str], feature_scaler) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_set[features] = feature_scaler.fit_transform(train_set[features])
    
    return train_set

if __name__ == "main":
    fileName = sys.argv[1] 
    df = pd.read_csv(fileName)
    df = preprocess(df, 5, 2)
    print(df.head())