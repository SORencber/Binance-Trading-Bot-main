import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, 
    f1_score, recall_score, roc_auc_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import optuna
from optuna.samplers import TPESampler
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================
# 1. GELİŞMİŞ VERİ İŞLEME
# =====================
def calculate_features(df, horizon=5):
    df = df.copy()
    
    # Fiyat Transformasyonları
    df['log_returns'] = np.log(df['close']/df['close'].shift(1))
    df['volatility'] = df['high'] - df['low']
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close'])/3 / df['volume'].rolling(24).sum())
    
    # Momentum & Trend Göstergeleri
    for window in [14, 50, 100]:
        df[f'ma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
    
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(14).mean() / 
                             df['close'].diff().clip(upper=0).abs().rolling(14).mean())))
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['adx'] = (abs(df['high'] - df['low']) / df['vwap']) * 100
    
    # Hedef Değişken Optimizasyonu
    future_returns = df['close'].pct_change(horizon).shift(-horizon)
    volatility = df['volatility'].rolling(horizon*3).std()
    df['target'] = ((future_returns > (volatility * 1.5)) & 
                   (df['volume'] > df['volume'].rolling(50).mean())).astype(int)
    
    # Sınıf Dengeleme
    pos_weight = (len(df) - df['target'].sum()) / df['target'].sum()
    df['weight'] = np.where(df['target'] == 1, pos_weight, 1)
    
    return df.dropna()

# =====================
# 2. VERİ SETİ OPTİMİZASYONU
# =====================
class MarketDataset(Dataset):
    def __init__(self, features, targets, weights, window=120, horizon=5):
        self.features = np.nan_to_num(features, nan=0)
        self.targets = targets.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.window = window
        self.horizon = horizon

    def __len__(self):
        return len(self.features) - self.window - self.horizon
    
    def __getitem__(self, idx):
        end_idx = idx + self.window
        x = self.features[idx:end_idx]
        y = self.targets[end_idx + self.horizon - 1]
        w = self.weights[end_idx + self.horizon - 1]
        return (
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.float32, device=device),
            torch.as_tensor(w, dtype=torch.float32, device=device)
        )

# =====================
# 3. SOTA MODEL MİMARİSİ
# =====================
class AlphaNetV2(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=4, dropout=0.4):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size)  # Giriş normalizasyonu
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                         batch_first=True, bidirectional=True, dropout=dropout)
        
        self.attention = nn.MultiheadAttention(
            2*hidden_size, 8, dropout=dropout, batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_size, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.input_bn(x.permute(0,2,1)).permute(0,2,1)
        x, _ = self.gru(x)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out  # Residual connection
        return self.fc(x[:,-1,:])

# =====================
# 4. PROFESYONEL EĞİTİM SİSTEMİ
# =====================
class InstitutionalTrainer:
    def __init__(self, input_size):
        self.input_size = input_size
        self.best_params = None
        
    def objective(self, trial, train_set, val_set):
        params = {
            'lr': trial.suggest_float('lr', 1e-6, 5e-4, log=True),
            'hidden_size': trial.suggest_categorical('hidden_size', [256, 512, 1024]),
            'num_layers': trial.suggest_int('num_layers', 3, 5),
            'dropout': trial.suggest_float('dropout', 0.2, 0.6),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        }
        
        model = AlphaNetV2(self.input_size, **params).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], 
                               weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=params['lr'], 
            steps_per_epoch=100, epochs=200
        )
        criterion = nn.BCELoss(reduction='none')
        
        train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False)
        
        # Early Stopping
        best_val_auc = 0
        patience = 0
        
        for epoch in range(200):
            # Training
            model.train()
            epoch_loss = 0
            for x, y, w in train_loader:
                optimizer.zero_grad()
                pred = model(x)
                loss = (criterion(pred.squeeze(), y) * w).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            
            # Validation
            val_auc = self._evaluate(model, val_loader)
            
            # Early Stopping Logic
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience += 1
                if patience >= 20:
                    break
        
        return best_val_auc

    def _evaluate(self, model, loader):
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                pred = model(x)
                probs.extend(pred.cpu().numpy())
                labels.extend(y.cpu().numpy())
        return roc_auc_score(labels, probs)

# =====================
# 5. GERÇEKÇİ BACKTEST SİSTEMİ
# =====================
class AdvancedBacktester:
    def __init__(self, initial_capital=100000):
        self.capital = initial_capital
        self.positions = []
        self.equity = [initial_capital]
        self.trade_log = []
        
    def execute_trade(self, signal, price, timestamp, commission=0.0002, slippage=0.0001):
        if signal == "HOLD":
            return
        
        executed_price = price * (1 + slippage) if signal == "BUY" else price * (1 - slippage)
        cost = commission * executed_price
        
        if signal == "BUY" and self.capital > 0:
            size = self.capital / executed_price
            self.positions.append({
                'entry_price': executed_price,
                'size': size,
                'entry_time': timestamp
            })
            self.capital = 0
            self.trade_log.append({
                'type': 'BUY',
                'price': executed_price,
                'cost': cost,
                'timestamp': timestamp
            })
            
        elif signal == "SELL" and self.positions:
            for position in self.positions:
                profit = (executed_price - position['entry_price']) * position['size'] - cost
                self.capital += position['entry_price'] * position['size'] + profit
                self.trade_log.append({
                    'type': 'SELL',
                    'entry_price': position['entry_price'],
                    'exit_price': executed_price,
                    'profit': profit,
                    'timestamp': timestamp
                })
            self.positions = []
            
        self.equity.append(self.capital + sum(
            p['size'] * price for p in self.positions
        ))

    def get_performance(self):
        returns = pd.Series(self.equity).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        mdd = (pd.Series(self.equity).cummax() - pd.Series(self.equity)).max()
        return {
            'sharpe': sharpe,
            'max_drawdown': mdd,
            'total_return': (self.equity[-1]/self.equity[0] - 1)*100,
            'trades': len(self.trade_log)
        }

# =====================
# 6. TÜM SÜREÇ YÖNETİMİ
# =====================
def run_pipeline(df, timeframe):
    # 1. Veri Hazırlama
    df = calculate_features(df)
    features = df.drop(columns=['target', 'weight'])
    targets = df['target']
    weights = df['weight']
    #print(df.head())
    # 2. Zaman Serisi Bölme
    tscv = TimeSeriesSplit(n_splits=3)
    results = {}
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
        if len(train_idx) < 1000 or len(test_idx) < 500:
            continue
            
        # 3. Ölçeklendirme
        scaler = RobustScaler()
        X_train = scaler.fit_transform(features.iloc[train_idx])
        X_test = scaler.transform(features.iloc[test_idx])
        os.makedirs('scalers', exist_ok=True)
        dump(scaler, f'scalers/scaler_{timeframe}_{fold}.joblib')
        
        # 4. Dataset
        train_set = MarketDataset(X_train, targets.iloc[train_idx], weights.iloc[train_idx])
        val_set = MarketDataset(X_test, targets.iloc[test_idx], weights.iloc[test_idx])
        
        # 5. Hiperparametre Optimizasyonu
        study = optuna.create_study(
            direction='maximize', 
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner()
        )
        trainer = InstitutionalTrainer(X_train.shape[1])
        study.optimize(
            lambda trial: trainer.objective(trial, train_set, val_set), 
            n_trials=100,
            timeout=3600
        )
        
        # 6. En İyi Model ile Tahmin
        best_model = AlphaNetV2(X_train.shape[1], **study.best_params).to(device)
        best_model.load_state_dict(torch.load('best_model.pth'))
        
        # 7. Backtest
        backtester = AdvancedBacktester()
        test_dates = df.iloc[test_idx].index[-len(val_set):]
        preds = []
        
        with torch.no_grad():
            for i in tqdm(range(len(val_set))):
                x, _, _ = val_set[i]
                pred = best_model(x.unsqueeze(0))
                preds.append(pred.item())
                price = df.iloc[test_idx].iloc[i]['close']
                signal = "BUY" if pred > 0.7 else "SELL" if pred < 0.3 else "HOLD"
                backtester.execute_trade(signal, price, test_dates[i])
        
        # 8. Sonuçlar
        performance = backtester.get_performance()
        results[f'fold_{fold}'] = {
            **performance,
            'accuracy': accuracy_score(targets.iloc[test_idx], (np.array(preds) > 0.5).astype(int)),
            'auc': roc_auc_score(targets.iloc[test_idx], preds)
        }
    
    return results


# 7. ÇALIŞTIRMA VE SONUÇ
# =====================
if __name__ == "__main__":
    # Örnek Veri Yükleme
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
    
    df = pd.read_csv(csv_1h_path, parse_dates=['timestamp']).set_index('timestamp')
    df.columns = [col.lower() for col in df.columns]
    ##print(df.head())  # Veri setinin ilk birkaç satırını kontrol et
    # Veriyi kontrol et
    print(df.shape)  # Satır ve sütun sayısı
    print(df.head())  # İlk 5 satır
    print(df.isnull().sum())  # Eksik veri kontrolü
    # Pipeline Çalıştırma
    results = run_pipeline(df, '1h')
    
    # Sonuçları Görselleştirme
    print("\n=== FINAL PERFORMANCE ===")
    for fold, metrics in results.items():
        print(f"\n{fold}:")
        print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Total Return: {metrics['total_return']:.2f}%")
        print(f"  Trades: {metrics['trades']}")
        print(f"  AUC-ROC: {metrics['auc']:.2f}")