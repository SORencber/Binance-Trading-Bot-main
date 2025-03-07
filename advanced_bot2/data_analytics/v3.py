import numpy as np
import pandas as pd
import torch
import traceback
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import BorderlineSMOTE

from pyro.optim import ClippedAdam
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Gelişmiş Teknik Göstergeler ve Veri Hazırlama
def calculate_enhanced_features(df, timeframe, higher_df=None, higher_tf=None, threshold_quantile=0.03):
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    
    # Üst zaman dilimi özellikleri
    if higher_df is not None and higher_tf is not None:
        df = calculate_multi_tf_features(df, higher_df, timeframe, higher_tf)
    
    # Temel Özellikler
    df['log_returns'] = np.log(df['close']/df['close'].shift(1))
    df['volatility'] = df[f'high'] - df['low']
    
    # Momentum Göstergeleri
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['signal'] = calculate_macd(df['close'])
    
    # Volatilite Göstergeleri
    df['atr'] = calculate_atr(df)
    df['adx'] = calculate_adx(df)
    
    # Hacim Göstergeleri
    df['obv'] = calculate_obv(df)
    
    # Hedef Değişken Oluşturma
    future_returns = df['close'].pct_change(5).shift(-5).dropna()
    if len(future_returns) == 0:
        raise ValueError("Yetersiz veri için hedef oluşturulamadı")
    
    threshold = future_returns.quantile(threshold_quantile)
    df['target'] = df['close'].pct_change(5).shift(-5) > threshold
    df['target'] = df['target'].astype(int).reindex(df.index)
    
    # Sınıf Dengesizliği Kontrolü
    class_counts = df['target'].value_counts()
    if len(class_counts) < 2:
        for q in [0.5, 0.55, 0.6, 0.65, 0.7]:
            threshold = future_returns.quantile(q)
            df['target'] = df[f'close'].pct_change(5).shift(-5) > threshold
            df['target'] = df['target'].astype(int)
            if df['target'].nunique() > 1: break
    
    return df.dropna()

def calculate_multi_tf_features(df, higher_df, tf, higher_tf):
    merged = pd.merge_asof(
        df.sort_index(),
        higher_df[['close', 'volume']].sort_index(),
        left_index=True,
        right_index=True,
        direction='nearest', 
        suffixes=('', f'_{higher_tf}')
    )
    # Convert merged columns to numeric
    merged[f'close_{higher_tf}'] = pd.to_numeric(merged[f'close_{higher_tf}'], errors='coerce')
    merged[f'volume_{higher_tf}'] = pd.to_numeric(merged[f'volume_{higher_tf}'], errors='coerce')
    merged[f'prev_close_{higher_tf}'] = merged[f'close_{higher_tf}'].shift(1)
    merged[f'higher_tf_trend'] = np.where(
        merged[f'close_{higher_tf}'] > merged[f'prev_close_{higher_tf}'], 1, 0)
    merged[f'rsi_{higher_tf}'] = calculate_rsi(merged[f'close_{higher_tf}'])
    merged[f'macd_{higher_tf}'], _ = calculate_macd(merged[f'close_{higher_tf}'])
    return merged
def calculate_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_atr(df, period=14):
    high = df[f'high']
    low = df[f'low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    plus_dm = high.diff().where((high.diff() > low.diff()) & (high.diff() > 0), 0)
    minus_dm = low.diff().where((low.diff() > high.diff()) & (low.diff() > 0), 0)
    tr = calculate_atr(df, period)
    plus_di = plus_dm.ewm(alpha=1/period, adjust=False).mean() / (tr + 1e-10) * 100
    minus_di = minus_dm.ewm(alpha=1/period, adjust=False).mean() / (tr + 1e-10) * 100
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

def calculate_obv(df):
    return (np.sign(df['close'].diff()) * df['volume']).cumsum()

# 3. Veri Seti ve Model Mimarisi
class FinancialDataset(Dataset):
    def __init__(self, features, targets, window_size=60, horizon=5):
        self.features = np.nan_to_num(features)
        self.targets = targets.reset_index(drop=True)  # İndex sıfırlandı
        self.window = window_size
        self.horizon = horizon

    def __len__(self):
        return max(len(self.features) - self.window - self.horizon + 1, 0)

    def __getitem__(self, idx):
        end_idx = idx + self.window
        target_idx = end_idx + self.horizon - 1
        if target_idx >= len(self.targets):
            raise IndexError("Hedef indeks veri boyutunu aşıyor.")
        x = self.features[idx:end_idx]
        y = self.targets.iloc[target_idx]
        return torch.FloatTensor(x).to(device), torch.FloatTensor([y]).to(device)

class MultiTemporalModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, dilation=4, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.attention = nn.MultiheadAttention(256, 4)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(2, 0, 1)
        x, _ = self.attention(x, x, x)
        return self.fc(x[-1])

# 4. Eğitim ve Pipeline
class MarketDetector:
    def __init__(self, input_size):
        self.model = MultiTemporalModel(input_size).to(device)
        self.scaler = RobustScaler()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self, train_loader, val_loader, epochs=50):
        best_loss = float('inf')
        for epoch in tqdm(range(epochs)):
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs.squeeze(), y.squeeze())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    outputs = self.model(x)
                    loss = self.criterion(outputs.squeeze(), y.squeeze())
                    val_loss += loss.item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

def optimized_pipeline(symbol, df, tf):
    try:
        df = calculate_enhanced_features(df, tf, threshold_quantile=0.05)
        if 'target' not in df.columns:
            raise ValueError("Hedef değişken oluşturulamadı")
        
        features = df.drop(columns=['target'], errors='ignore').fillna(0)
        targets = df['target'].fillna(0).astype(int)
        
        if len(features) < 100:
            raise ValueError("Yetersiz veri boyutu")
        if targets.nunique() < 2:
            raise ValueError("Yetersiz sınıf çeşitliliği")
        
        scaler = RobustScaler()
        features = scaler.fit_transform(features)
        
        tscv = TimeSeriesSplit(n_splits=3)
        results = {}
        for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
            if len(train_idx) < 100 or len(test_idx) < 50:
                print(f"{tf} - Fold {fold}: Yetersiz veri, atlanıyor.")
                continue
                
            train_set = FinancialDataset(features[train_idx], targets.iloc[train_idx])
            test_set = FinancialDataset(features[test_idx], targets.iloc[test_idx])
            
            if len(train_set) == 0 or len(test_set) == 0:
                print(f"{tf} - Fold {fold}: Boş dataset, atlanıyor.")
                continue
                
            train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=128)
            
            detector = MarketDetector(features.shape[1])
            detector.train(train_loader, test_loader, epochs=30)
            
            current = torch.FloatTensor(features[-60:]).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = detector.model(current).item()
            
            results[fold] = {
                'probability': round(prob*100, 2),
                'position': np.clip(prob*0.15, 0.02, 0.1)
            }
        
        return results if results else {'error': 'Tüm foldlar başarısız'}
    except Exception as e:
        print(f"[HATA] {tf}: {str(e)}")
        return {'error': str(e)}

# 5. Ana Fonksiyon
def main(symbol, data_paths):
    tf_dfs = {}
    for tf, path in data_paths.items():
        try:

            df = pd.read_csv(path, parse_dates=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype('int64') // 10**9
            df.columns = [col.lower() for col in df.columns]

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            tf_dfs[tf] = df
        except Exception as e:
            print(f"{tf} verisi yüklenemedi: {str(e)}")
    
    results = {}
    for tf in ['5m','15m','30m','1h','4h','1d']:
        if tf not in tf_dfs:
            continue
        try:
            res = optimized_pipeline(symbol, tf_dfs[tf], tf)
            results[tf] = res
        except Exception as e:
            print(f"{tf} işlenirken hata: {str(e)}")
    return results

if __name__ == "__main__":
    s="BTCUSDT"
    csv_5m_path = f"../data_storage/{s}_5m.csv"
    csv_15m_path = f"../data_storage/{s}_15m.csv"
    csv_30m_path = f"../data_storage/{s}_30m.csv"
    csv_1h_path = f"../data_storage/{s}_1h.csv"
    csv_4h_path = f"../data_storage/{s}_4h.csv"
    csv_1d_path = f"../data_storage/{s}_1d.csv"
    csv_1w_path = f"../data_storage/{s}_1w.csv"
    data_paths = {
        '5m': csv_5m_path,
        '15m': csv_15m_path,
        '30m':  csv_30m_path,
        '1h': csv_1h_path,
        '4h': csv_4h_path,
        '1d': csv_1d_path ,
        '1w':csv_1w_path
    }
    results = main(s, data_paths)
    print("\nSonuçlar:")
    for tf, res in results.items():
        print(f"{tf}:")
        for fold, data in res.items():
            print(f"  Fold {fold}: {data}")