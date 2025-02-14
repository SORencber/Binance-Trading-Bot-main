import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import traceback
from functools import lru_cache

# =============================================
# 1. Teknik Gösterge Hesaplamaları
# =============================================
class TechnicalIndicators:
    @staticmethod
    def calculate_ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.DataFrame({
            'h-l': high - low,
            'h-pc': abs(high - close.shift(1)),
            'l-pc': abs(low - close.shift(1))
        }).max(axis=1)
        
        plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
        minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)
        
        tr_ema = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_ema)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_ema)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

# =============================================
# 2. Piyasa Rejimi Analizi
# =============================================
class MarketRegimeAnalyzer:
    DEFAULT_CONFIG = {
        'adx_trend': 25,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'bollinger_period': 20,
        'ma_fast': 50,
        'ma_slow': 200,
        'atr_multiplier': 1.5
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.indicators = TechnicalIndicators()

    def detect_regime(self, df: pd.DataFrame, timeframe: str) -> Dict:
        try:
            df = self._preprocess_data(df, timeframe)
            indicators = self._calculate_indicators(df, timeframe)
            signals = self._generate_signals(df, indicators)
            return signals
        except Exception as e:
            traceback.print_exc()
            return {'error': str(e), 'timeframe': timeframe}

    def _preprocess_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        required_cols = [f'{col}_{timeframe}' for col in ['open', 'high', 'low', 'close', 'volume']]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Eksik kolonlar: {required_cols}")
        return df[required_cols].copy()

    def _calculate_indicators(self, df: pd.DataFrame, timeframe: str) -> Dict:
        close = df[f'close_{timeframe}']
        return {
            'ma_fast': close.rolling(self.config['ma_fast']).mean(),
            'ma_slow': close.rolling(self.config['ma_slow']).mean(),
            'adx': self.indicators.calculate_adx(
                df[f'high_{timeframe}'], 
                df[f'low_{timeframe}'], 
                close
            ),
            'rsi': self.indicators.calculate_rsi(close),
            'atr': self._calculate_atr(df, timeframe),
            'bollinger': self._calculate_bollinger_bands(close)
        }

    def _calculate_atr(self, df: pd.DataFrame, timeframe: str) -> pd.Series:
        high, low, close = df[f'high_{timeframe}'], df[f'low_{timeframe}'], df[f'close_{timeframe}']
        tr = pd.DataFrame({
            'h-l': high - low,
            'h-pc': abs(high - close.shift(1)),
            'l-pc': abs(low - close.shift(1))
        }).max(axis=1)
        return tr.rolling(14).mean()

    def _calculate_bollinger_bands(self, close: pd.Series) -> Dict:
        sma = close.rolling(self.config['bollinger_period']).mean()
        std = close.rolling(self.config['bollinger_period']).std()
        return {
            'upper': sma + 2*std,
            'lower': sma - 2*std,
            'width': (sma + 2*std - (sma - 2*std)) / sma
        }

    def _generate_signals(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        signals = {}
        
        # Trend Analizi
        signals['trend'] = self._determine_trend(
            df[f'close_{self.timeframe}'],
            indicators['ma_fast'],
            indicators['ma_slow'],
            indicators['adx']
        )
        
        # Momentum Analizi
        signals['momentum'] = self._determine_momentum(
            indicators['rsi'],
            self.indicators.calculate_macd(df[f'close_{self.timeframe}'])
        )
        
        # Breakout Sinyalleri
        signals['breakout'] = self._detect_breakouts(
            df[f'high_{self.timeframe}'],
            df[f'low_{self.timeframe}'],
            indicators['atr']
        )
        
        return signals

# =============================================
# 3. Çoklu Zaman Dilimi Yönetimi
# =============================================
class MultiTimeframeAnalyzer:
    TIMEFRAME_HIERARCHY = {
        '5m': '15m',
        '15m': '30m',
        '30m': '1h',
        '1h': '4h',
        '4h': '1d',
        '1d': '1w'
    }

    def __init__(self, historical_data: Dict[str, pd.DataFrame]):
        self.data = historical_data
        self.analyzers = {tf: MarketRegimeAnalyzer() for tf in self.data.keys()}

    def analyze_all_timeframes(self) -> Dict:
        results = {}
        for tf in self.TIMEFRAME_HIERARCHY.keys():
            if tf not in self.data:
                continue
            higher_tf = self.TIMEFRAME_HIERARCHY.get(tf)
            context = {
                'higher_tf_data': self.data.get(higher_tf),
                'higher_tf': higher_tf
            }
            results[tf] = self.analyzers[tf].detect_regime(self.data[tf], context)
        return results

# =============================================
# 4. Sinyal Entegrasyonu ve Filtreleme
# =============================================
class SignalIntegrator:
    PATTERN_MAPPING = {
        'bullish': ['double_bottom', 'inverse_head_shoulders'],
        'bearish': ['head_shoulders', 'double_top'],
        'neutral': ['triangle', 'rectangle']
    }

    def integrate_signals(self, regime_data: Dict, patterns: Dict) -> Dict:
        integrated = {}
        for tf, analysis in regime_data.items():
            if 'error' in analysis:
                continue
            valid_patterns = self._filter_patterns(
                patterns.get(tf, []),
                analysis['regime']
            )
            integrated[tf] = {
                'regime': analysis,
                'patterns': valid_patterns,
                'composite_score': self._calculate_score(analysis, valid_patterns)
            }
        return integrated

    def _filter_patterns(self, patterns: List, regime: str) -> List:
        regime_type = 'bullish' if 'bull' in regime.lower() else \
                      'bearish' if 'bear' in regime.lower() else 'neutral'
        return [p for p in patterns if p['type'] in self.PATTERN_MAPPING[regime_type]]

# =============================================
# 5. Yardımcı Fonksiyonlar ve Decorator'lar
# =============================================
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            print(f"Eksik veri hatası: {str(e)}")
            return {'error': f'missing_data:{str(e)}'}
        except Exception as e:
            traceback.print_exc()
            return {'error': str(e)}
    return wrapper

# =============================================
# Kullanım Örneği
# =============================================
if __name__ == "__main__":
    # Örnek Veri Yükleme
    timeframes = ['15m', '1h', '4h']
    historical_data = {
        tf: pd.read_csv(f'../data_storage/BTCUSDT_{tf}.csv', index_col='timestamp', parse_dates=True)
        for tf in timeframes
    }
    
    # Çoklu Zaman Dilimi Analizi
    mt_analyzer = MultiTimeframeAnalyzer(historical_data)
    regime_results = mt_analyzer.analyze_all_timeframes()
    
    # Pattern Entegrasyonu
    pattern_data = {
        '15m': [{'type': 'double_bottom', 'confidence': 0.85}],
        '1h': [{'type': 'head_shoulders', 'confidence': 0.72}]
    }
    
    integrator = SignalIntegrator()
    final_signals = integrator.integrate_signals(regime_results, pattern_data)
    
    print("Sonuçlar:")
    for tf, data in final_signals.items():
        print(f"{tf} Zaman Dilimi:")
        print(f"- Rejim: {data['regime'].get('regime','Bilinmiyor')}")
        print(f"- Tespit Edilen Patternler: {len(data['patterns'])}")
        print(f"- Kompozit Skor: {data['composite_score']:.2f}")
        print("----------------------")
