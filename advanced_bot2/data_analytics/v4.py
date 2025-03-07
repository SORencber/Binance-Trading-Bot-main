import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import BorderlineSMOTE
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. VERİ İŞLEME VE ÖZELLİK MÜHENDİSLİĞİ
def calculate_enhanced_features(df, timeframe, threshold_quantile=0.05):
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    
    # Teknik Göstergeler
    df['log_returns'] = np.log(df['close']/df['close'].shift(1))
    df['volatility'] = df['high'] - df['low']
    
    # RSI Hesaplama
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Hedef Değişken
    future_returns = df['close'].pct_change(5).shift(-5)
    if future_returns.dropna().empty:
        raise ValueError("Yetersiz veri için hedef oluşturulamadı")
    threshold = future_returns.quantile(threshold_quantile)
    df['target'] = (future_returns > threshold).astype(int)
    
    # Sınıf Dengesizliği Kontrolü
    if df['target'].nunique() < 2:
        for q in [0.5, 0.55, 0.6]:
            threshold = future_returns.quantile(q)
            df['target'] = (future_returns > threshold).astype(int)
            if df['target'].nunique() > 1: 
                break
    
    return df.dropna()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from joblib import dump, load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================
# 1. VERİ İŞLEME
# =====================
def calculate_advanced_features(df, horizon=5):
    df = df.copy()
    
    # Fiyat Temelli Özellikler
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['high'] - df['low']
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3) / df['volume'].rolling(24).sum()

    # RSI Hesaplaması
    df['rsi'] = 100 - (100 / (1 + (
        df['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / 
        (-df['close'].diff().where(lambda x: x < 0, 0).rolling(14).mean())
    )))

    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    
    # Hedef Değişken
    future_returns = df['close'].pct_change(horizon).shift(-horizon)
    df['target'] = (future_returns > future_returns.rolling(horizon*3).std() * 1.5).astype(int)
    
    # Sınıf Dengesi için Ağırlıklar
    class_weights = 1 / df['target'].value_counts(normalize=True)
    df['sample_weight'] = df['target'].map(class_weights)
    
    return df.dropna()

# =====================
# 2. VERİ SETİ & LOADER
# =====================
class MarketDataset(Dataset):
    def __init__(self, features, targets, weights, window=60):
        self.features = np.nan_to_num(features, nan=0)
        self.targets = targets.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.features) - self.window

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.window]
        y = self.targets[idx + self.window]
        w = self.weights[idx + self.window]
        return (
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.float32, device=device),
            torch.as_tensor(w, dtype=torch.float32, device=device)
        )

# =====================
# 3. MODEL MİMARİSİ
# =====================
class AlphaNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                         batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(2 * hidden_size, 4, dropout=dropout)
        self.norm = nn.LayerNorm(2 * hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x, _ = self.gru(x)  
        x = x.permute(1, 0, 2)  
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = x[-1]  
        return self.fc(x)

# =====================
# 4. MODEL EĞİTİMİ
# =====================
class AlphaTrainer:
    def __init__(self, input_size):
        self.input_size = input_size
        
    def train_model(self, train_loader, val_loader, params):
        model = AlphaNet(self.input_size, params['hidden_size'], params['num_layers'], params['dropout']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.5, device=device))

        best_val_loss = float('inf')
        patience = 0

        for epoch in range(100):
            model.train()
            for x, y, w in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(x).squeeze()
                loss = criterion(pred, y) * w.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            val_loss = self.evaluate(model, val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience += 1
                if patience >= 15:
                    break
        
        return best_val_loss

    def evaluate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, w in val_loader:
                pred = model(x).squeeze()
                loss = criterion(pred, y) * w.mean()
                val_loss += loss.item()
        return val_loss / len(val_loader)

# =====================
# 5. BACKTEST
# =====================
def backtest_strategy(signals, prices):
    equity = [10000]
    for i in range(1, len(signals)):
        if signals[i-1] == "AL":
            cost = prices[i]
            equity.append(equity[-1] - cost)
        elif signals[i-1] == "SAT" and equity[-1] > 0:
            equity.append(equity[-1] + prices[i])
        else:
            equity.append(equity[-1])
    return equity

# =====================
# 6. TÜM ZAMAN DİLİMLERİ İÇİN ÇALIŞTIRMA
# =====================
if __name__ == "__main__":
    s="BTCUSDT"
    csv_5m_path = f"../data_storage/{s}_5m.csv"
    csv_15m_path = f"../data_storage/{s}_15m.csv"
    csv_30m_path = f"../data_storage/{s}_30m.csv"
    csv_1h_path = f"../data_storage/{s}_1h.csv"
    csv_4h_path = f"../data_storage/{s}_4h.csv"
    csv_1d_path = f"../data_storage/{s}_1d.csv"
    csv_1w_path = f"../data_storage/{s}_1w.csv"
    timeframes = {
        '5m': csv_5m_path,
        '15m': csv_15m_path,
        '30m':  csv_30m_path,
        '1h': csv_1h_path,
        '4h': csv_4h_path,
        '1d': csv_1d_path ,
        '1w':csv_1w_path
    }
    
    final_results = {}
    for tf, path in timeframes.items():
        df = pd.read_csv(path, parse_dates=['timestamp'])
        df.columns = [col.lower() for col in df.columns]

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = calculate_advanced_features(df)

        scaler = RobustScaler()
        features = scaler.fit_transform(df.drop(columns=['target', 'sample_weight', 'timestamp']))
        targets = df['target'].values
        weights = df['sample_weight'].values
        dump(scaler, f'scaler_{tf}.joblib')

        dataset = MarketDataset(features, targets, weights)
        train_loader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(dataset, batch_size=512, shuffle=False)

        trainer = AlphaTrainer(features.shape[1])
        best_loss = trainer.train_model(train_loader, val_loader, {
            'hidden_size': 256, 'num_layers': 3, 'dropout': 0.3, 'lr': 1e-4
        })

        print(f"{tf} - Best Loss: {best_loss:.4f}")

