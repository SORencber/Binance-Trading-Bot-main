import ccxt
import pandas as pd
import numpy as np
import schedule
import time
from datetime import datetime
import traceback

########################################
# (B) Geliştirilmiş optimize_parameters Fonksiyonu (Grid Search Örneği)
########################################
def detect_regime_with_override(df, timeframe, param_override):
    """
    detect_regime fonksiyonunu, param_override sözlüğü ile çağırır.
    Bu fonksiyon, optimize_parameters içindeki grid aramasında aday parametreleri denemek için kullanılır.
    """
    # optimize_params=False yapıyoruz çünkü param_override ile doğrudan parametreleri veriyoruz.
    return detect_regime(df, timeframe, optimize_params=False, param_override=param_override)


def optimize_parameters(df, timeframe):
    """
    Basit grid search ile tarihsel veri üzerinde farklı parametre kombinasyonlarını deneyip,
    en yüksek dinamik skora sahip olan parametreleri seçen örnek optimize_parameters fonksiyonu.
    Burada dinamik skoru (dynamic_score) performans metriği olarak kullanıyoruz.
    """
    best_score = -np.inf
    best_params = None

    # Aday parametre aralıkları (örnek değerler; ihtiyaca göre ayarlanabilir)
    adx_trends = [20, 22, 24]
    ma_fast_periods = [50, 60, 70]
    ma_slow_periods = [180, 200, 220]
    rsi_overbought_vals = [70, 68, 72]
    rsi_oversold_vals = [30, 32, 28]
    atr_mult_breakouts = [1.5, 1.6, 1.7]

    # Grid search döngüsü
    for adx_trend in adx_trends:
        for ma_fast in ma_fast_periods:
            for ma_slow in ma_slow_periods:
                for rsi_overbought in rsi_overbought_vals:
                    for rsi_oversold in rsi_oversold_vals:
                        for atr_mult in atr_mult_breakouts:
                            candidate = {
                                'adx_trend': adx_trend,
                                'rsi_overbought': rsi_overbought,
                                'rsi_oversold': rsi_oversold,
                                'boll_band_squeeze': 0.045,  # Sabit değer örneği
                                'volume_min': None,
                                'atr_mult_breakout': atr_mult,
                                'ma_fast_period': ma_fast,
                                'ma_slow_period': ma_slow
                            }
                            try:
                                candidate_signals = detect_regime_with_override(df, timeframe, candidate)
                                # Burada backtest yerine örnek olarak dynamic_score kullanıyoruz.
                                candidate_score = candidate_signals.get("dynamic_score", 0)
                                if candidate_score > best_score:
                                    best_score = candidate_score
                                    best_params = candidate
                            except Exception as e:
                                print(f"[Candidate ERROR] {e}")
                                traceback.print_exc()
    print(f"En iyi parametreler: {best_params} (score: {best_score})")
    return best_params

########################################
# 1) detect_regime Fonksiyonu (Gelişmiş ve Hata Yönetimli)
########################################
def detect_regime(df, timeframe, higher_df=None, higher_timeframe=None, optimize_params=False, param_override=None):
    """
    Verilen dataframe'de (df) belirli bir timeframe'e göre trend, momentum, hacim ve
    volatilite analizini yapıp, 'regime', 'signal' vb. bilgileri döndürür.
    
    Ek Gelişmeler:
      - Eğer optimize_params True ise, tarihsel veriden optimum parametreler belirlenir.
      - Dinamik ağırlıklandırma: ADX ve hacim verilerine göre bir skorlama yapılır.
      - Hesaplama bloklarının çoğu try/except ile sarılarak hata yönetimi artırılmıştır.
    """
    try:
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        # Zaman dilimlerine göre temel parametre ayarları (varsayılanlar)
        config = {
            '5m':  {'adx_trend': 20, 'rsi_overbought': 70, 'rsi_oversold': 30,
                    'boll_band_squeeze': 0.045, 'volume_min': None, 'atr_mult_breakout': 1.5,
                    'ma_fast_period': 50, 'ma_slow_period': 180},
            '15m': {'adx_trend': 20, 'rsi_overbought': 70, 'rsi_oversold': 30,
                    'boll_band_squeeze': 0.045, 'volume_min': None, 'atr_mult_breakout': 1.5,
                    'ma_fast_period': 50, 'ma_slow_period': 180},
            '30m': {'adx_trend': 20, 'rsi_overbought': 70, 'rsi_oversold': 30,
                    'boll_band_squeeze': 0.045, 'volume_min': None, 'atr_mult_breakout': 1.5,
                    'ma_fast_period': 50, 'ma_slow_period': 180},
            '1h':  {'adx_trend': 20, 'rsi_overbought': 70, 'rsi_oversold': 30,
                    'boll_band_squeeze': 0.045, 'volume_min': None, 'atr_mult_breakout': 1.5,
                    'ma_fast_period': 50, 'ma_slow_period': 180},
            '4h':  {'adx_trend': 20, 'rsi_overbought': 70, 'rsi_oversold': 30,
                    'boll_band_squeeze': 0.045, 'volume_min': None, 'atr_mult_breakout': 1.5,
                    'ma_fast_period': 50, 'ma_slow_period': 180},
            '1d':  {'adx_trend': 20, 'rsi_overbought': 70, 'rsi_oversold': 30,
                    'boll_band_squeeze': 0.045, 'volume_min': None, 'atr_mult_breakout': 1.5,
                    'ma_fast_period': 50, 'ma_slow_period': 180},
            '1w':  {'adx_trend': 20, 'rsi_overbought': 70, 'rsi_oversold': 30,
                    'boll_band_squeeze': 0.045, 'volume_min': None, 'atr_mult_breakout': 1.5,
                    'ma_fast_period': 50, 'ma_slow_period': 180},
        }
        cfg = config.get(timeframe, config[f"{timeframe}"])

        # Parametre optimizasyonu (isteğe bağlı)
        if optimize_params:
            try:
                optimized = optimize_parameters(df, timeframe)
                cfg.update(optimized)
            except Exception as e:
                print(f"[OPTIMIZATION ERROR] {e}")
                traceback.print_exc()

        def EMA(series, span):
            try:
                return series.ewm(span=span, adjust=False).mean()
            except Exception as e:
                print(f"[EMA ERROR] {e}")
                traceback.print_exc()
                return series

        # Zorunlu kolonlar: high_{timeframe}, low_{timeframe}, close_{timeframe}, volume_{timeframe} (opsiyonel)
        try:
            df.columns = [col.lower() for col in df.columns]
            high = df[f"high_{timeframe}"]
            low = df[f"low_{timeframe}"]
            close = df[f"close_{timeframe}"]
        except Exception as e:
            print(f"[COLUMN ERROR] Eksik kolon: {e}")
            traceback.print_exc()
            return {"timeframe": timeframe, "error": "Gerekli kolonlar eksik."}

        # Hareketli Ortalamalar: optimize edilmiş periyotları kullanın
        try:
            ma_fast = close.rolling(cfg.get("ma_fast_period", 50)).mean()
            ma_slow = close.rolling(cfg.get("ma_slow_period", 200)).mean()
        except Exception as e:
            print(f"[MA ERROR] {e}")
            traceback.print_exc()
            ma_fast = close.rolling(50).mean()
            ma_slow = close.rolling(200).mean()

        # ADX Hesaplaması
        try:
            period_adx = 14
            high_shift = high.shift(1)
            low_shift = low.shift(1)
            close_shift = close.shift(1)
            tr_df = pd.DataFrame({
                'h-l': high - low,
                'h-c': (high - close_shift).abs(),
                'l-c': (low - close_shift).abs()
            })
            tr_df['true_range'] = tr_df.max(axis=1)
            plus_dm = high - high_shift
            minus_dm = low_shift - low
            plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
            minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0

            tr_smooth = tr_df['true_range'].ewm(alpha=1/period_adx, adjust=False).mean()
            plus_dm_smooth = plus_dm.ewm(alpha=1/period_adx, adjust=False).mean()
            minus_dm_smooth = minus_dm.ewm(alpha=1/period_adx, adjust=False).mean()
            plus_di = 100 * plus_dm_smooth / tr_smooth
            minus_di = 100 * minus_dm_smooth / tr_smooth
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx_series = dx.ewm(alpha=1/period_adx, adjust=False).mean()
        except Exception as e:
            print(f"[ADX ERROR] {e}")
            traceback.print_exc()
            adx_series = pd.Series([np.nan]*len(close))

        # MACD Hesaplaması
        try:
            ema_fast_12 = EMA(close, 12)
            ema_slow_26 = EMA(close, 26)
            macd_line = ema_fast_12 - ema_slow_26
            signal_line = EMA(macd_line, 9)
            macd_hist = macd_line - signal_line
        except Exception as e:
            print(f"[MACD ERROR] {e}")
            traceback.print_exc()
            macd_hist = pd.Series([np.nan]*len(close))

        # RSI Hesaplaması
        try:
            period_rsi = 14
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/period_rsi, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period_rsi, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
        except Exception as e:
            print(f"[RSI ERROR] {e}")
            traceback.print_exc()
            rsi = pd.Series([np.nan]*len(close))

        # Stoch RSI
        try:
            period_stoch = 14
            min_rsi = rsi.rolling(period_stoch).min()
            max_rsi = rsi.rolling(period_stoch).max()
            stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
        except Exception as e:
            print(f"[Stoch RSI ERROR] {e}")
            traceback.print_exc()
            stoch_rsi = pd.Series([np.nan]*len(rsi))

        # Momentum (basit örnek)
        try:
            period_mom = 10
            momentum = close - close.shift(period_mom)
        except Exception as e:
            print(f"[Momentum ERROR] {e}")
            traceback.print_exc()
            momentum = pd.Series([np.nan]*len(close))

        # Hacim, OBV, MFI
        volume = df.get(f'volume_{timeframe}')
        obv, mfi = None, None
        try:
            if volume is not None:
                price_dir = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                obv = (price_dir * volume).cumsum()

                tp = (df[f'high_{timeframe}'] + df[f'low_{timeframe}'] + df[f'close_{timeframe}']) / 3
                money_flow = tp * volume
                pos_flow = np.where(tp.diff() > 0, money_flow, 0)
                neg_flow = np.where(tp.diff() < 0, money_flow, 0)
                pos_sum = pd.Series(pos_flow, index=df.index).rolling(14).sum()
                neg_sum = pd.Series(neg_flow, index=df.index).rolling(14).sum()
                money_flow_ratio = pos_sum / neg_sum.replace(0, np.nan)
                mfi_series = 100 - (100 / (1 + money_flow_ratio))
                mfi = mfi_series
        except Exception as e:
            print(f"[Volume/OBV/MFI ERROR] {e}")
            traceback.print_exc()

        # Bollinger Bantları (volatilite)
        try:
            bb_period = 20
            mid = close.rolling(bb_period).mean()
            std = close.rolling(bb_period).std()
            bb_upper = mid + 2 * std
            bb_lower = mid - 2 * std
            bb_width = (bb_upper - bb_lower) / mid
        except Exception as e:
            print(f"[Bollinger ERROR] {e}")
            traceback.print_exc()
            bb_width = pd.Series([np.nan]*len(close))

        # ATR (ortalama gerçek aralık)
        try:
            period_atr = 14
            atr = tr_df['true_range'].rolling(period_atr).mean()
        except Exception as e:
            print(f"[ATR ERROR] {e}")
            traceback.print_exc()
            atr = pd.Series([np.nan]*len(close))

        # Yakın geçmişteki en yüksek / en düşük
        try:
            lookback_sr = min(30, len(df))
            recent_high = high.rolling(lookback_sr).max().iloc[-1] if len(df) > 0 else None
            recent_low = low.rolling(lookback_sr).min().iloc[-1] if len(df) > 0 else None
        except Exception as e:
            print(f"[Recent High/Low ERROR] {e}")
            traceback.print_exc()
            recent_high, recent_low = None, None

        # Breakout tespiti (basit)
        breakout_up = False
        breakout_down = False
        volume_min = df[f'volume_{timeframe}'].mean()  # Normal hacim

        try:
            if recent_high is not None and close.iloc[-1] > recent_high * 1.001:
                if (volume is None or volume_min is None or volume.iloc[-1] > volume_min or 0):
                    if (bb_width.iloc[-1] > bb_width.iloc[-2]) and (
                        atr.iloc[-1] is not None and (close.iloc[-1] - recent_high) > cfg['atr_mult_breakout'] * atr.iloc[-1]
                    ):
                        breakout_up = True

            if recent_low is not None and close.iloc[-1] < recent_low * 0.999:
                if (volume is None or cfg['volume_min'] is None or volume.iloc[-1] > (cfg['volume_min'] or 0)):
                    if (bb_width.iloc[-1] > bb_width.iloc[-2]) and (
                        atr.iloc[-1] is not None and (recent_low - close.iloc[-1]) > cfg['atr_mult_breakout'] * atr.iloc[-1]
                    ):
                        breakout_down = True
        except Exception as e:
            print(f"[Breakout ERROR] {e}")
            traceback.print_exc()

        # Düşük volatilite: Bollinger width küçük ve ADX düşük
        low_volatility = False
        try:
            if (not np.isnan(bb_width.iloc[-1])
                and bb_width.iloc[-1] < cfg['boll_band_squeeze']
                and adx_series.iloc[-1] < cfg['adx_trend']):
                low_volatility = True
        except Exception as e:
            print(f"[Low Volatility ERROR] {e}")
            traceback.print_exc()

        # Üst zaman dilimi analizi (opsiyonel)
        higher_trend = None
        try:
            if (higher_df is not None) and (higher_timeframe is not None) and (higher_timeframe in config):
                hd = higher_df.copy()
                hd.columns = [col.lower() for col in hd.columns]

                close_h = hd[f'close_{higher_timeframe}']
                high_h = hd[f'high_{higher_timeframe}']
                low_h = hd[f'low_{higher_timeframe}']

                ma_fast_h = close_h.rolling(50).mean()
                ma_slow_h = close_h.rolling(200).mean()

                period_adx_h = 14
                tr_h = pd.DataFrame({
                    'h-l': high_h - low_h,
                    'h-c': (high_h - close_h.shift(1)).abs(),
                    'l-c': (low_h - close_h.shift(1)).abs()
                })
                tr_h['true_range'] = tr_h.max(axis=1)
                plus_dm_h = high_h - high_h.shift(1)
                minus_dm_h = low_h.shift(1) - low_h
                plus_dm_h[(plus_dm_h < 0) | (plus_dm_h < minus_dm_h)] = 0
                minus_dm_h[(minus_dm_h < 0) | (minus_dm_h < plus_dm_h)] = 0

                tr_h_smooth = tr_h['true_range'].ewm(alpha=1/period_adx_h, adjust=False).mean()
                plus_dm_h_smooth = plus_dm_h.ewm(alpha=1/period_adx_h, adjust=False).mean()
                minus_dm_h_smooth = minus_dm_h.ewm(alpha=1/period_adx_h, adjust=False).mean()
                plus_di_h = 100 * plus_dm_h_smooth / tr_h_smooth
                minus_di_h = 100 * minus_dm_h_smooth / tr_h_smooth
                dx_h = 100 * (plus_di_h - minus_di_h).abs() / (plus_di_h + minus_di_h)
                adx_h = dx_h.ewm(alpha=1/period_adx_h, adjust=False).mean()

                higher_trend = 'sideways'
                if not np.isnan(ma_slow_h.iloc[-1]):
                    if (close_h.iloc[-1] > ma_fast_h.iloc[-1] > ma_slow_h.iloc[-1]
                        and adx_h.iloc[-1] > config[higher_timeframe]['adx_trend']):
                        higher_trend = 'bullish'
                    elif (close_h.iloc[-1] < ma_fast_h.iloc[-1] < ma_slow_h.iloc[-1]
                          and adx_h.iloc[-1] > config[higher_timeframe]['adx_trend']):
                        higher_trend = 'bearish'
                else:
                    if (close_h.iloc[-1] > ma_fast_h.iloc[-1] and adx_h.iloc[-1] > config[higher_timeframe]['adx_trend']):
                        higher_trend = 'bullish'
                    elif (close_h.iloc[-1] < ma_fast_h.iloc[-1] and adx_h.iloc[-1] > config[higher_timeframe]['adx_trend']):
                        higher_trend = 'bearish'
        except Exception as e:
            print(f"[Higher TF ERROR] {e}")
            traceback.print_exc()

        # Trend belirleme
        trend = 'sideways'
        try:
            if not np.isnan(ma_slow.iloc[-1]):
                if (close.iloc[-1] > ma_fast.iloc[-1] > ma_slow.iloc[-1]) and (adx_series.iloc[-1] > cfg['adx_trend']):
                    trend = 'bullish'
                elif (close.iloc[-1] < ma_fast.iloc[-1] < ma_slow.iloc[-1]) and (adx_series.iloc[-1] > cfg['adx_trend']):
                    trend = 'bearish'
            else:
                if (close.iloc[-1] > ma_fast.iloc[-1]) and (adx_series.iloc[-1] > cfg['adx_trend']):
                    trend = 'bullish'
                elif (close.iloc[-1] < ma_fast.iloc[-1]) and (adx_series.iloc[-1] > cfg['adx_trend']):
                    trend = 'bearish'
        except Exception as e:
            print(f"[Trend ERROR] {e}")
            traceback.print_exc()

        # Momentum analizi
        momentum_status = 'neutral'
        try:
            if (macd_hist.iloc[-1] > 0) and (rsi.iloc[-1] > 50) and (momentum.iloc[-1] > 0):
                momentum_status = 'bullish'
                if (rsi.iloc[-1] > cfg['rsi_overbought']) or (not np.isnan(stoch_rsi.iloc[-1]) and stoch_rsi.iloc[-1] > 0.8):
                    momentum_status = 'bullish (overbought)'
            elif (macd_hist.iloc[-1] < 0) and (rsi.iloc[-1] < 50) and (momentum.iloc[-1] < 0):
                momentum_status = 'bearish'
                if (rsi.iloc[-1] < cfg['rsi_oversold']) or (not np.isnan(stoch_rsi.iloc[-1]) and stoch_rsi.iloc[-1] < 0.2):
                    momentum_status = 'bearish (oversold)'
        except Exception as e:
            print(f"[Momentum Analysis ERROR] {e}")
            traceback.print_exc()

        # Hacim durumu
        volume_status = None
        try:
            if volume is not None and len(volume) >= 20:
                vol_ma = volume.rolling(20).mean()
                if vol_ma.iloc[-1] > 0:
                    if volume.iloc[-1] > 1.5 * vol_ma.iloc[-1]:
                        volume_status = f'high_{higher_trend}'
                    elif volume.iloc[-1] < 0.5 * vol_ma.iloc[-1]:
                        volume_status = f'low_{higher_trend}'
                    else:
                        volume_status = 'average'
        except Exception as e:
            print(f"[Volume Status ERROR] {e}")
            traceback.print_exc()

        # Volatilite durumu (ATR'ye göre)
        volatility_status = 'normal'
        try:
            if (not atr.empty) and (len(atr) >= 20):
                atr_ma = atr.rolling(20).mean()
                if (not np.isnan(atr_ma.iloc[-1])) and (atr_ma.iloc[-1] > 0):
                    if atr.iloc[-1] > 1.2 * atr_ma.iloc[-1]:
                        volatility_status = f'high_{higher_trend}'
                    elif atr.iloc[-1] < 0.8 * atr_ma.iloc[-1]:
                        volatility_status = f'low_{higher_trend}'
        except Exception as e:
            print(f"[Volatility ERROR] {e}")
            traceback.print_exc()

        # Rejim ve sinyal belirleme
        regime = 'Neutral'
        try:
            if trend == 'bullish':
                regime = 'Bullish Trend'
                if higher_trend == 'bearish':
                    regime = 'Bullish (counter higher trend)'
                if breakout_up:
                    regime = 'Bullish Breakout'
            elif trend == 'bearish':
                regime = 'Bearish Trend'
                if higher_trend == 'bullish':
                    regime = 'Bearish (counter higher trend)'
                if breakout_down:
                    regime = 'Bearish Breakout'
            else:
                if low_volatility:
                    regime = 'Low Volatility Consolidation'
                else:
                    regime = 'Range/Choppy'
                if breakout_up:
                    regime = 'Bullish Breakout from Range'
                elif breakout_down:
                    regime = 'Bearish Breakout from Range'
        except Exception as e:
            print(f"[Regime Determination ERROR] {e}")
            traceback.print_exc()

        # Temel sinyal: trend takibi
        signal = None
        try:
            if regime.startswith('Bullish'):
                if ('Trend' in regime) or ('Breakout' in regime):
                    signal = 'Buy signal (long)'
            elif regime.startswith('Bearish'):
                if ('Trend' in regime) or ('Breakout' in regime):
                    signal = 'Sell signal (short)'
             
             
                    # detect_regime fonksiyonuna eklenen yeni mantık
            if (
                rsi.iloc[-1] > 75 and  # Daha agresif RSI eşiği
                (trend in ['sideways', 'bearish']) and 
                (macd_hist.iloc[-1] < macd_hist.iloc[-3]) and  # Son 3 barda MACD düşüş
                (volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.2) and  # Hacim artışı
                (higher_trend in ['sideways', 'bearish'])  # Üst TF onayı
            ):
                signal = 'Sell signal (RSI Overbought - Strong)'
        except Exception as e:
            print(f"[Signal ERROR] {e}")
            traceback.print_exc()

        # Ek sinyaller / notlar
        range_signal = None
        try:
            if timeframe == '4h':
                if regime in ['Range/Choppy', 'Low Volatility Consolidation']:
                    if rsi.iloc[-1] < 35:
                        range_signal = "Range-Buy near support (4H)"
                    elif rsi.iloc[-1] > 65:
                        range_signal = "Range-Sell near resistance (4H)"
                    else:
                        range_signal = "Mid-range, no immediate trade (4H)"
        except Exception as e:
            print(f"[Range Signal ERROR] {e}")
            traceback.print_exc()

        volatility_signal = None
        breakout_direction_note = None
        try:
            if low_volatility:
                volatility_signal = "Bollinger Squeeze => Olası Breakout"
                if momentum_status.startswith("bullish") or trend == "bullish":
                    breakout_direction_note = "Muhtemel Yukarı Breakout (Trend/Momentum bullish)"
                elif momentum_status.startswith("bearish") or trend == "bearish":
                    breakout_direction_note = "Muhtemel Aşağı Breakout (Trend/Momentum bearish)"
                else:
                    breakout_direction_note = "Olası Breakout (Yön belirsiz)"
        except Exception as e:
            print(f"[Volatility Signal ERROR] {e}")
            traceback.print_exc()

        caution_note = None
        try:
            if (trend == 'bullish') and ("overbought" in momentum_status):
                caution_note = "Trend bullish + RSI overbought => Ters short açmayın, stopları takip edin."
        except Exception as e:
            print(f"[Caution Note ERROR] {e}")
            traceback.print_exc()

        strategic_buy_note = None
        try:
            if rsi.iloc[-1] < 25:
                strategic_buy_note = "RSI < 25 => Aşırı satım, dip alımı potansiyeli (uzun vade)."
        except Exception as e:
            print(f"[Strategic Buy Note ERROR] {e}")
            traceback.print_exc()
    # Stratejik notları güncelle (YENİ EKLENDİ)
        strategic_sell_note = None
        try:
            if rsi.iloc[-1] > cfg['rsi_overbought']:
                strategic_sell_note = f"RSI {rsi.iloc[-1]:.1f} > {cfg['rsi_overbought']} - Potansiyel Short Fırsatı"
        except Exception as e:
            print(f"[Strategic Sell Note ERROR] {e}")
            traceback.print_exc()
        # Dinamik Ağırlıklandırma / Skorlama: ADX ve Hacim etkisi
        dynamic_score = 0
        try:
            adx_val = adx_series.iloc[-1]
            if not np.isnan(adx_val):
                dynamic_score += adx_val  # Trend gücü olarak ADX katkısı
        except Exception as e:
            print(f"[Dynamic Score ADX ERROR] {e}")
            traceback.print_exc()
        try:
            if volume is not None and len(volume) >= 20:
                vol_ma = volume.rolling(20).mean()
                if vol_ma.iloc[-1] > 0:
                    vol_ratio = volume.iloc[-1] / vol_ma.iloc[-1]
                    if vol_ratio > 1.5:
                        dynamic_score += 5
                    elif vol_ratio < 0.5:
                        dynamic_score -= 5
        except Exception as e:
            print(f"[Dynamic Score Volume ERROR] {e}")
            traceback.print_exc()

        # Sonuç sinyalleri
        signals = {
            'timeframe': timeframe,
            'trend': trend,
            'momentum': momentum_status,
            'volume': volume_status,
            'volatility': volatility_status,
            'higher_timeframe_trend': higher_trend,
            'regime': regime,
            'signal': signal,
            'adx': float(adx_series.iloc[-1]) if not np.isnan(adx_series.iloc[-1]) else None,
            'rsi': float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None,
            'macd_hist': float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else None,
            'obv': float(obv.iloc[-1]) if (obv is not None and not np.isnan(obv.iloc[-1])) else None,
            'mfi': float(mfi.iloc[-1]) if (mfi is not None and not np.isnan(mfi.iloc[-1])) else None,
            'bollinger_width': float(bb_width.iloc[-1]) if not np.isnan(bb_width.iloc[-1]) else None,
            'range_signal': range_signal,
            'volatility_signal': volatility_signal,
            'caution_note': caution_note,
            'strategic_buy_note': strategic_buy_note,
            'potential_breakout_note': breakout_direction_note,
            'strategic_sell_note': strategic_sell_note,
        'rsi_value': float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None,

            'dynamic_score': dynamic_score 
        }

        return signals

    except Exception as e:
        print(f"[ERROR] detect_regime genel hata: {e}")
        traceback.print_exc()
        return {"timeframe": timeframe, "error": str(e)}


########################################
# 2) Çoklu Zaman Dilimi Fonksiyonları
########################################
def get_higher_tf(tf):
    mapping = {
        '5m': '30m',
        '15m': '1h',
        '30m': '4h',
        '1h': '4h',
        '4h': '1d',
        '1d': "1w"
    }
    return mapping.get(tf)

def get_all_regimes(df_5m, df_15m, df_30m, df_1h, df_4h, df_1d, df_1w, optimize_params=False):
    tf_dfs = {
        '5m': df_5m,
        '15m': df_15m,
        '30m': df_30m,
        '1h':  df_1h,
        '4h':  df_4h,
        '1d':  df_1d,
        '1w':  df_1w
    }

    results = {}
    for tf in ['5m','15m','30m','1h','4h','1d','1w']:
        try:
            df = tf_dfs[tf]
            higher_tf = get_higher_tf(tf)
            optimized_params=None
            if optimize_params:
                optimized_params = optimize_parameters(df, tf)
                print("Optimize Edilmiş Parametreler:")
                print(optimized_params)
            if higher_tf:
                df_higher = tf_dfs[higher_tf]
                out = detect_regime(df, tf, higher_df=df_higher, higher_timeframe=higher_tf, optimize_params=optimize_params, param_override=optimized_params)
            else:
                out = detect_regime(df, tf, optimize_params=optimize_params)
            results[tf] = out
            print(out)
        except Exception as e:
            print(f"[ERROR] TF={tf}, {e}")
            traceback.print_exc()
            results[tf] = {"timeframe": tf, "error": str(e)}
    return results

########################################
# 3) Pattern Sınıflandırması ve Filtreleme
########################################
PATTERN_REJIME_MAP = {
    "Bullish": [
        "inverse_head_and_shoulders",
        "double_bottom",
        "triple_bottom_advanced",
        "cup_handle",
        "gann",
        "harmonic",
        "wolfe"
    ],
    "Bearish": [
        "head_and_shoulders",
        "double_top",
        "triple_top_advanced",
        "gann"
    ],
    "Range": [
        "rectangle",
        "channel",
        "flag_pennant",
        "gann",
        "wolfe"
    ],
    "LowVol": [
        "flag_pennant",
        "harmonic",
        "wolfe",
        "elliott"
    ],
    # Çift yönlü patternler için ayrı kategori ekleniyor:
    "Bidirectional": [
        "triangle",
        "wedge"
    ]
}


def combine_regime_and_pattern_signals(regime_info, pattern_signals: dict):
    """
    1) detect_regime -> rejim bilgisi
    2) extract_pattern_trade_levels_filtered -> pattern sinyalleri (pattern_signals)
    3) filter_patterns_by_regime -> rejime uygun pattern sinyallerini döndürür

    Dönüş: {
       "regime_info": {...},    # detect_regime çıktısı
       "final_signals": [...]   # rejim + pattern filtrelenmiş sinyaller listesi
    }
    """
    final_signals = filter_patterns_by_regime(
        pattern_signals, 
        regime_info.get("regime", "Neutral"), 
        regime_info.get("timeframe", "1m"),
        regime_info.get("rsi_value", 0.0)
    )
    regime_direction = None
    if regime_info.get("signal") == "Buy signal (long)":
        regime_direction = "LONG"
    elif regime_info.get("signal") == "Sell signal (short)":
        regime_direction = "SHORT"

    filtered_final = []
    for sig in final_signals:
        pdir = sig.get("direction")
        if regime_direction is None:
            filtered_final.append(sig)
        else:
            if regime_direction == "LONG" and pdir == "SHORT":
                continue
            elif regime_direction == "SHORT" and pdir == "LONG":
                continue
            filtered_final.append(sig)
    return {
        "regime_info": regime_info,
        "final_signals": filtered_final
    }




def filter_patterns_by_regime(pattern_signals: dict, regime: str,tf:str="1m",rsi_value=0.0) -> list:
    """
    pattern_signals: extract_pattern_trade_levels_filtered(...) sonucu,
                     örn: { 'head_and_shoulders': [ {...}, {...} ], 
                            'triangle': [ {...} ], ... }
    regime: detect_regime(...)["regime"] => "Bullish Trend", "Range/Choppy", "Low Vol Squeeze", vb.

    Dönüş: final_signals => [ { entry_price, stop_loss, take_profit, direction, pattern_type, ... }, ... ]
    """

    # 1) Rejimi "Bullish", "Bearish", "Range", "LowVol" gibi ana kategoriye dönüştürelim
    #    Örn. "Bullish Trend" veya "Bullish Breakout" => "Bullish"
    #         "Bearish Trend" => "Bearish"
    #         "Low Vol Squeeze", "Low Vol Consolidation" => "LowVol"
    #         "Range/Choppy" => "Range"
    #    vb.
    regime_key = "Neutral"
    if regime.startswith("Bullish") :
 
        regime_key = "Bullish"
    elif regime.startswith("Bearish") :

        regime_key = "Bearish"
    elif "Range" in regime:
        regime_key = "Range"
    elif "Low Vol" in regime:
        regime_key = "LowVol"
 
    # 2) Bu regime_key'e göre hangi pattern'lerin "geçerli" olduğunu bul
    valid_list = []
    if regime_key == "Bullish":
        valid_list = PATTERN_REJIME_MAP["Bullish"]
    elif regime_key == "Bearish":
        valid_list = PATTERN_REJIME_MAP["Bearish"]
    elif regime_key == "Range":
        valid_list = PATTERN_REJIME_MAP["Range"]
    elif regime_key == "LowVol":
        valid_list = PATTERN_REJIME_MAP["LowVol"]
        # Aşırı alımda Bearish pattern'leri aktifleştir (YENİ EKLENDİ)
   # Aşırı alımda yalnızca confirmed bearish pattern'leri etkinleştir
    if regime_key == "Neutral" and rsi_value > 75:
        valid_list += ["head_and_shoulders", "double_top", "wedge"]
        #print(tf, regime_key,valid_list)
    final_signals = []
    #print(valid_list)
    # 3) pattern_signals => dict: { 'triangle': [ { entry_price, ... }, ... ], ... }
    for ptype, arr in pattern_signals.items():
        #print(ptype)

        # Sadece "geçerli pattern" listesinde olanları al
        if ptype not in valid_list:
            #print(ptype,valid_list)
            continue
        # Şimdi arr içinde her sinyal (entry, stop, direction vs.)
        for sinyal in arr:
            #print("---------------------",ptype,sinyal)
            # direction mismatch kontrolü (opsiyonel)
            # Örneğin "Bullish" rejimde SHORT sinyalini iptal edebilirsiniz
            mismatch = False
            if regime_key == "Bullish" and sinyal.get("direction") == "SHORT":
                mismatch = True
            if regime_key == "Bearish" and sinyal.get("direction") == "LONG":
                mismatch = True

            if not mismatch:
                # pattern_type ekleyelim
                sinyal["pattern_type"] = ptype
                final_signals.append(sinyal)
    #print(tf,final_signals) 
    return final_signals


def combine_regime_and_pattern_signals(
    regime_info,
    pattern_signals: dict,
   
):
    """
    1) detect_regime -> rejim 
    2) extract_pattern_trade_levels_filtered -> pattern sinyalleri
    3) filter_patterns_by_regime -> rejime uygun pattern sinyallerini döndür

    Dönüş: {
       "regime_info": {...},    # detect_regime fonk. çıktısı
       "final_signals": [...]   # rejim + pattern filtrelenmiş sinyaller listesi
    }
    """
    #print(pattern_signals)
   
 # 1) Rejime göre pattern tipi filtrenizi yapın
    final_signals = filter_patterns_by_regime(

        pattern_signals, 
        regime_info["regime"], regime_info["timeframe"],regime_info["potential_breakout_note"]
    )
    #print(".............",final_signals,pattern_signals, regime_info["regime"])
   # 2) Rejimin signal alanına bak (Buy signal (long) / Sell signal (short) / None)
    regime_direction = None
    if regime_info["signal"] == "Buy signal (long)":
        regime_direction = "LONG"
    elif regime_info["signal"] == "Sell signal (short)":
        regime_direction = "SHORT"

    # 3) Mismatch
    filtered_final = []
    for sig in final_signals:
        pdir = sig.get("direction")
        print(pdir)
        if regime_direction is None:
            # Rejim side => accept both? your call
            filtered_final.append(sig)
        else:
            # eğer regime_direction "LONG" => short pattern yok
            if regime_direction=="LONG" and pdir=="SHORT":
                # mismatch => atla
                continue
            elif regime_direction=="SHORT" and pdir=="LONG":
                continue
            filtered_final.append(sig)
    #
    #pattern_dict = group_patterns_by_type(final_signals)
    return {
        "regime_info": regime_info,
        "final_signals": filtered_final
    }


def produce_realistic_signal(
    all_regimes: dict,
    pattern_signals_map: dict,
    main_tf: str = "1h",
    combo_name: str = "intraday"
):
    """
    - all_regimes: get_all_regimes(...) çıktısı, dict => { '1h': {...}, '4h': {...}, ... }
    - pattern_signals_map: aynı yapıda pattern sinyalleri => { '1h': {...}, '4h': {...}, ... }
      örn: pattern_signals_map["1h"] = {'triangle': [ { direction="LONG", ...}, ... ], ...}
    - main_tf: asıl işlem yapılacak TF
    - combo_name: MTF alignment seti, "intraday", "swing", vs.

    Dönüş:
      {
        "final_side": "LONG" or "SHORT",
        "score_details": {...}, 
        "main_regime": {...},   # detect_regime(main_tf) çıktısı
        "patterns_used": [...], # seçilen pattern sinyalleri
        "alignment": ...,       # MTF alignment raporu
      }
    """
    # A) Ana TF rejimi:
    #main_regime_info = all_regimes.get(main_tf, {})
    main_signal = all_regimes.get("signal")  # "Buy signal (long)", "Sell signal (short)", None
    main_trend = all_regimes.get("trend")    # "bullish", "bearish", "sideways"
    main_momentum = all_regimes.get("momentum", "neutral")
    main_regime_label = all_regimes.get("regime", "Neutral")  # "Bullish Trend", "Range/Choppy", ...
    main_regime_potential_breakout_note = all_regimes.get("potential_breakout_note", "None")  # "Bullish Trend", "Range/Choppy", ...

    # B) Pattern sinyalleri (ana TF)
    #pat_signals_tf = pattern_signals_map.get(main_tf, {})
    combined = combine_regime_and_pattern_signals(all_regimes, pattern_signals_map)
    final_patterns = combined["final_signals"]  # rejime + potential_breakout_note'a uygun pattern listesi
    #print("2-----------",combined["final_signals"])
    # C) MTF Alignment
    synergy = analyze_multi_tf_alignment(all_regimes, combo_name)
    alignment = synergy.get("alignment", "Mixed or Sideways")  # "Mostly Bullish", "Mostly Bearish", vs.

    # Bazı ek veriler
    htf_trend = all_regimes.get("higher_timeframe_trend")  # 'bullish','bearish','sideways', etc.
    rsi_val = all_regimes.get("rsi", 50.0)

    # D) Skor oluşturma
    score_details = {
        "rejim_signal": 0,
        "trend": 0,
        "momentum": 0,
        "alignment": 0,
        "pattern": 0,
        # vs...
    }

    # Örnek puanlar (tamamen keyfî, istediğiniz gibi değiştirebilirsiniz):
    # 1) Rejim sinyali
    if main_signal == "Buy signal (long)":
        score_details["rejim_signal"] = +3
    elif main_signal == "Sell signal (short)":
        score_details["rejim_signal"] = -3

    # 2) Trend
    if main_trend == "bullish":
        score_details["trend"] = +2
    elif main_trend == "bearish":
        score_details["trend"] = -2

    # 3) Momentum
    # "bullish" => +1, "bearish" => -1, 
    if "bullish" in main_momentum:
        score_details["momentum"] = +1
    elif "bearish" in main_momentum:
        score_details["momentum"] = -1

    # 4) MTF Alignment
    if "Bullish" in alignment:
        score_details["alignment"] = +2
    elif "Bearish" in alignment:
        score_details["alignment"] = -2
    # "Mixed" => 0, no change

    # 5) Pattern sinyalleri (basitçe hepsini toplayabiliriz)
    pattern_score = 0
    for p in final_patterns:
        pdir = p.get("direction")
        if pdir == "LONG":
            pattern_score += 2
        elif pdir == "SHORT":
            pattern_score -= 2
    score_details["pattern"] = pattern_score

    # Şimdi total
    total_score = sum(score_details.values())

    # E) Sona gelindi => net karar
    if total_score > 0:
        final_side = "LONG"
    elif total_score < 0:
        final_side = "SHORT"
    else:
        # Tie durum => fallback
        # Mesela higher_timeframe_trend'e bakabilir veya RSI < 50 => SHORT, RSI>50 => LONG diyebiliriz
        if htf_trend == "bullish":
            final_side = "LONG"
        elif htf_trend == "bearish":
            final_side = "SHORT"
        else:
            pass
            
    pattern_dict = group_patterns_by_type(final_patterns)

    print("Sonuclar......>>>>>", pattern_dict)
    return {
        "score_details": score_details,
        "break_out_note":main_regime_potential_breakout_note,
        "total_score": total_score,
        "main_regime": all_regimes,
        "patterns_used": pattern_dict,
        "alignment": synergy
    }



def group_patterns_by_type(pattern_list: list) -> dict:
    """
    pattern_list = [
       {"pattern_type": "head_and_shoulders", "entry_price":..., ...},
       {"pattern_type": "triangle", ...},
       ...
    ]
    Dönüş => {
       "head_and_shoulders": [ {...}, {...} ],
       "triangle": [ {...} ],
       ...
    }
    """
    grouped = {}
    for p in pattern_list:
        ptype = p.get("pattern_type", "unknown")
        if ptype not in grouped:
            grouped[ptype] = []
        grouped[ptype].append(p)
    return grouped

#######################################################
# 4) Çoklu Zaman Dilimi Uyum Analizi (Örnek)
#######################################################
TIMEFRAME_COMBINATIONS = {
    "scalping":  ["5m", "15m", "1h"],
    "intraday":  ["15m", "1h", "4h"],
    "swing":     ["1h", "4h", "1d"],   # ya da "1h","4h","1d" olarak da düzenlenebilir
    "position":  ["1d", "1w"]         # vs.
}
def analyze_multi_tf_alignment(all_regimes: dict, combo_name: str = "intraday"):
    """
    all_regimes: get_all_regimes fonksiyonundan dönen {tf: detect_regime_output} dict'i
    combo_name: TIMEFRAME_COMBINATIONS içinden bir set seçilir (örn. "intraday" => 15m,1h,4h)

    Amaç: Seçili zaman dilimleri arasında trend ve rejim uyumunu ölçmek.
    Örnek:
      - 15m => Bullish Trend
      - 1h  => Bullish (counter higher trend) -> mismatch?
      - 4h  => Bearish Trend

    Sonuç: "Mixed signals" veya "Bearish alignment" vs.
    """
    if combo_name not in TIMEFRAME_COMBINATIONS:
        return {"error": f"Kombinasyon bulunamadı: {combo_name}"}
    
    tfs = TIMEFRAME_COMBINATIONS[combo_name]
    synergy_results = {}
    directions = []
    for tf in tfs:
        reg_info = all_regimes.get(tf)
        if reg_info is None:
            synergy_results[tf] = "No data"
            continue
        synergy_results[tf] = reg_info["regime"]
        if reg_info["signal"] == "Buy signal (long)":
            directions.append("bullish")
        elif reg_info["signal"] == "Sell signal (short)":
            directions.append("bearish")
        else:
            directions.append("sideways")

    # Basit bir "uyum" ölçütü:
    # Tüm TF'ler bullish veya en az 2/3 bullish => "Bullish alignment"
    # Tümü bearish veya en az 2/3 bearish => "Bearish alignment"
    # Diğer durumlarda "Mixed"
    bull_count = directions.count("bullish")
    bear_count = directions.count("bearish")
    n = len(directions)

    if bull_count == n:
        synergy_str = "Güclü Yukari Trend"
    elif bear_count == n:
        synergy_str = "Güclü Asagi Trend"
    elif bull_count >= (n//2 + 1):
        synergy_str = "Cogu Mum Yukari Trend"
    elif bear_count >= (n//2 + 1):
        synergy_str = "Cogu  Asagi Trend"
    else:
        synergy_str = "Karisik ve Yatay"

    synergy_report = {
        "combo_timeframes": tfs,
        "regimes": synergy_results,
        "alignment": synergy_str
    }
    return synergy_report

########################################
# 4) Backtest Entegrasyonu
########################################
def backtest_strategy(df, regime_signals, initial_capital=10000):
    """
    Basit backtest fonksiyonu.
    regime_signals içindeki sinyallere göre trade simülasyonu yapılır.
    Örnek: 'Buy signal (long)' alım, 'Sell signal (short)' satım sinyali olduğunda pozisyon açılır.
    (Gerçek stratejilerde risk yönetimi, stop-loss, take-profit vb. eklenmelidir.)
    """
    try:
        capital = initial_capital
        position = 0  # 0: pozisyon yok, 1: long, -1: short
        entry_price = None

        # Örneğin 1h timeframe kapanış fiyatları üzerinden simülasyon
        close_prices = df[f'close_1h']
        signals = regime_signals.get('1h', {})
        
        for i in range(len(close_prices)):
            # Örnek olarak sinyal sabit kabul ediliyor; gerçek durumda sinyallerin zaman içindeki değişimi izlenmelidir.
            if signals.get('signal') == 'Buy signal (long)' and position == 0:
                position = 1
                entry_price = close_prices.iloc[i]
            elif signals.get('signal') == 'Sell signal (short)' and position == 0:
                position = -1
                entry_price = close_prices.iloc[i]
            # Her 10 barda pozisyon kapatılıyor (örnek)
            if position != 0 and i % 10 == 0 and entry_price is not None:
                exit_price = close_prices.iloc[i]
                if position == 1:
                    profit = exit_price - entry_price
                elif position == -1:
                    profit = entry_price - exit_price
                capital += profit
                position = 0
                entry_price = None

        return {
            "initial_capital": initial_capital,
            "final_capital": capital,
            "total_return_percent": ((capital - initial_capital) / initial_capital) * 100
        }
    except Exception as e:
        print(f"[Backtest ERROR] {e}")
        traceback.print_exc()
        return {"error": str(e)}
