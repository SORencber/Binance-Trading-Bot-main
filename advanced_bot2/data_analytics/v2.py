import ccxt
import pandas as pd
import numpy as np
import schedule
import time
from datetime import datetime
import traceback
import aiohttp
from typing import Dict
from scipy.stats import zscore

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


async def detect_regime(symbol, df, timeframe, higher_df=None, higher_timeframe=None, optimize_params=False, param_override=None):
    
   
    def EMA(series, span):
        return series.ewm(span=span, adjust=False).mean()
    
    def calculate_rsi(close, period=14):
        try:
            delta = close.diff().dropna()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            print(f"RSI Error: {str(e)}")
            return pd.Series([np.nan]*len(close))

    def detect_fakeout(high, low, close, volume, rsi, open_):
        fakeout_signal = None
        fakeout_confidence = 0
        
        try:
            recent_high = high.rolling(20).max().iloc[-1]
            recent_low = low.rolling(20).min().iloc[-1]
            
            breakout_up = close.iloc[-1] > recent_high * 1.001
            breakout_down = close.iloc[-1] < recent_low * 0.999
            
            vol_ma = volume.rolling(20).mean().iloc[-1] if volume is not None else 0
            low_volume = (volume.iloc[-1] < vol_ma * 0.8) if (volume is not None and vol_ma > 0) else False
            
            candle_range = high.iloc[-1] - low.iloc[-1]
            wick_ratio = 0
            if candle_range > 0:
                if breakout_up:
                    wick_ratio = (high.iloc[-1] - close.iloc[-1])/candle_range
                elif breakout_down:
                    wick_ratio = (close.iloc[-1] - low.iloc[-1])/candle_range
                
            close_confirmation = (breakout_up and close.iloc[-1] < open_.iloc[-1]) or (breakout_down and close.iloc[-1] > open_.iloc[-1])
            rsi_divergence = (rsi.iloc[-1] > 70 and breakout_up) or (rsi.iloc[-1] < 30 and breakout_down)
            
            fakeout_conditions = [
                low_volume,
                close_confirmation,
                rsi_divergence,
                wick_ratio > 0.7
            ]
            
            fakeout_score = sum(fakeout_conditions)
            if fakeout_score >= 3:
                fakeout_confidence = min(90, fakeout_score * 25)
                direction = 'Up' if breakout_up else 'Down'
                fakeout_signal = f"Potential Fakeout ({direction}) | Confidence: {fakeout_confidence}%"

        except Exception as e:
            print(f"Fakeout Error: {str(e)}")
        
        return fakeout_signal, fakeout_confidence
    
    def determine_regime(trend, adx_value, bb_width, close, ma_fast, ma_slow, 
                            higher_trend, breakout_up, breakout_down, low_volatility):
            regime = "Neutral"
            
            # Trend Temelli Rejimler
            if trend == "bullish":
                if adx_value > 25:
                    regime = "Strong Bull Trend"
                    if breakout_up:
                        regime = "Bull Trend with Breakout"
                else:
                    regime = "Weak Bull Trend"
                
                if higher_trend == "bearish":
                    regime += " (Counter Higher TF)"
            
            elif trend == "bearish":
                if adx_value > 25:
                    regime = "Strong Bear Trend"
                    if breakout_down:
                        regime = "Bear Trend with Breakdown"
                else:
                    regime = "Weak Bear Trend"
                
                if higher_trend == "bullish":
                    regime += " (Counter Higher TF)"
            
            # Volatilite Temelli Rejimler
            else:
                if low_volatility and bb_width < 0.05:
                    regime = "Low Volatility Consolidation"
                    if close.iloc[-1] > ma_fast.iloc[-1]:
                        regime += " (Bullish Bias)"
                    else:
                        regime += " (Bearish Bias)"
                else:
                    regime = "Choppy Market"
                    
                if breakout_up:
                    regime = "Bullish Breakout from Range"
                elif breakout_down:
                    regime = "Bearish Breakdown from Range"

            return regime
   
   # Yeni Eklenen Entegrasyonlar
    async def fetch_institutional_data(session: aiohttp.ClientSession) -> Dict:
        """Ücretsiz API'lerle kurumsal veri entegrasyonu"""
        try:
            # 1. Dune'dan Zincir Verisi (Net Akış)
            dune_url = "https://api.dune.com/api/v2/query/1286871/results"
            dune_data = await session.get(dune_url, headers={"X-Dune-API-Key": "LauqGb3BK9eoQJYGZJKGth7zGHnmQxar"})
            
            print(dune_data)
            net_flow = (await dune_data.json())['result']['rows'][-1]['net_flow']
           
            # 2. Binance Futures Büyük Emirler
            oi_url = "https://fapi.binance.com/fapi/v1/openInterest"
            oi_data = await session.get(oi_url, params={"symbol": symbol})
            oi = (await oi_data.json())['openInterest']

            return {
                'net_flow': net_flow,
                'oi_change': (float(oi) - df['sumOpenInterest'].iloc[-2])/df['sumOpenInterest'].iloc[-2]
            }
        except Exception as e:
            print(f"Inst. Data Error: {str(e)}")
            return {}

    def enhanced_volume_analysis(df: pd.DataFrame,timeframe:str="5m") -> Dict:
        """Gelişmiş hacim analizi"""
        # Hacim Profili
        bins = np.linspace(df[f'low_{timeframe}'].min(), df[f'high_{timeframe}'].max(), 20)
        df['vp_bin'] = pd.cut(df[f'close_{timeframe}'], bins=bins)
        vp = df.groupby('vp_bin')[f'volume_{timeframe}'].sum()
        
        # VWAP
        df['vwap'] = (df[f'volume_{timeframe}'] * (df[f'high_{timeframe}'] + df[f'low_{timeframe}'] + df[f'close_{timeframe}'])/3).cumsum() / df[f'volume_{timeframe}'].cumsum()
        
        return {
            'volume_profile': vp.idxmax(),
            'vwap_deviation': (df[f'close_{timeframe}'].iloc[-1] - df['vwap'].iloc[-1])/df['vwap'].iloc[-1],
            'volume_zscore': zscore(df[f'volume_{timeframe}'])[-1]
        }

   
        # 10. AL/SAT Sinyal Hesaplama
 
    def generate_signals(high, low, close, volume, rsi, open_):
          
            signals = {
                'primary_signal': 'neutral',
                'signal_confidence': 0,
                'signal_conditions': [],
                'emergency_trigger': False
            }

            # MACD Hesaplama (Sinyaller için eklendi)
            macd_line = EMA(close, 12) - EMA(close, 26)
            signal_line = EMA(macd_line, 9)
            macd_hist = macd_line - signal_line

            # Kritik Sinyal Koşulları
            current_condition = {
                'bull_regime': 'bull' in regime.lower(),
                'bear_regime': 'bear' in regime.lower(),
                'adx_strong': adx.iloc[-1] > 25,
                'rsi_oversold': rsi.iloc[-1] < 35,
                'rsi_overbought': rsi.iloc[-1] > 70,
                'macd_bullish': macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3],
                'macd_bearish': macd_hist.iloc[-1] < macd_hist.iloc[-2] < macd_hist.iloc[-3],
                'volume_spike': volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 2,
                'ma_crossover': ma_fast.iloc[-1] > ma_slow.iloc[-1] and ma_fast.iloc[-2] < ma_slow.iloc[-2],
                'higher_tf_bull': higher_trend == 'bullish' if higher_trend else False
            }

            # AL Sinyal Kriterleri (3 Seviye)
            buy_conditions = [
                current_condition['bull_regime'],
                current_condition['adx_strong'],
                current_condition['rsi_oversold'],
                current_condition['macd_bullish'],
                current_condition['volume_spike'],
                current_condition['ma_crossover'],
                current_condition['higher_tf_bull']
            ]

            # SAT Sinyal Kriterleri (3 Seviye + Acil Durum)
            sell_conditions = [
                current_condition['bear_regime'],
                current_condition['rsi_overbought'],
                current_condition['macd_bearish'],
                (current_condition['bull_regime'] and fakeout_conf >= 70),
                close.iloc[-1] < ma_slow.iloc[-1] * 0.93  # %7 Stop-Loss benzeri
            ]

            # Sinyal Güven Hesaplama
            buy_score = sum([1 for cond in buy_conditions[:5] if cond])  # İlk 5 temel koşul
            sell_score = sum([1 for cond in sell_conditions if cond])

            # AL Sinyal Mantığı
            if buy_score >= 3 and not any(sell_conditions):
                signals['primary_signal'] = 'buy'
                signals['signal_confidence'] = min(95, 25 + (buy_score * 15))
                signals['signal_conditions'] = [k for k,v in current_condition.items() if v]
                
                # Volume spike ve higher TF onayı bonus puan
                if current_condition['volume_spike']: signals['signal_confidence'] += 5
                if current_condition['higher_tf_bull']: signals['signal_confidence'] += 10

            # SAT Sinyal Mantığı
            elif sell_score >= 1:
                signals['primary_signal'] = 'sell'
                signals['signal_confidence'] = min(95, 30 + (sell_score * 20))
                signals['signal_conditions'] = [k for k,v in current_condition.items() if v]
                
                if sell_conditions[-1]:  # Acil durum
                    signals['emergency_trigger'] = True
                    signals['signal_confidence'] = 99

            return signals


    try:
        # 1. Veri Hazırlık ve Kontroller
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        #print(df.columns)
        # mandatory_cols = [f'high_{timeframe}', f'low_{timeframe}', f'close_{timeframe}', f'open_{timeframe}']
        
        # if any(col not in df.columns for col in mandatory_cols):
        #     missing = [col for col in mandatory_cols if col not in df.columns]
        #     return {"error": f"Missing columns: {missing}"}

        # 2. Temel Göstergeler
        #print(f"CLoseeeeee _{timeframe}", df.columns)
 # Parametre optimizasyonu (isteğe bağlı)
        if optimize_params:
            try:
                optimized = optimize_parameters(df, timeframe)
                #cfg.update(optimized)
            except Exception as e:
                print(f"[OPTIMIZATION ERROR] {e}")
                traceback.print_exc()

        close = df[f'close_{timeframe}']
        high = df[f'high_{timeframe}']
        low = df[f'low_{timeframe}']
        open_ = df[f'open_{timeframe}']
        volume = df.get(f'volume_{timeframe}')

        # 3. Gösterge Hesaplamaları
        rsi = calculate_rsi(close)
        fakeout_signal, fakeout_conf = detect_fakeout(high, low, close, volume, rsi, open_)

        # 4. ADX Hesaplama
        period_adx = 14
        try:
            tr = pd.DataFrame({
                'h-l': high - low,
                'h-pc': (high - close.shift(1)).abs(),
                'l-pc': (low - close.shift(1)).abs()
            }).max(axis=1)
            
            plus_dm = high - high.shift(1)
            minus_dm = low.shift(1) - low
            plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
            minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0
            
            tr_smooth = tr.ewm(alpha=1/period_adx, adjust=False).mean()
            plus_smooth = plus_dm.ewm(alpha=1/period_adx, adjust=False).mean()
            minus_smooth = minus_dm.ewm(alpha=1/period_adx, adjust=False).mean()
            
            plus_di = 100 * (plus_smooth / tr_smooth)
            minus_di = 100 * (minus_smooth / tr_smooth)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.ewm(alpha=1/period_adx, adjust=False).mean()
        except:
            adx = pd.Series([np.nan]*len(close))

        # 5. Trend Analizi
        ma_fast = EMA(close, 50)
        ma_slow = EMA(close, 200)
        trend = 'sideways'
        
        try:
            last_close = close.iloc[-1]
            if (plus_di.iloc[-1] > minus_di.iloc[-1]) and (last_close > ma_fast.iloc[-1] > ma_slow.iloc[-1]) and (adx.iloc[-1] > 25):
                trend = 'bullish'
            elif (plus_di.iloc[-1] < minus_di.iloc[-1]) and (last_close < ma_fast.iloc[-1] < ma_slow.iloc[-1]) and (adx.iloc[-1] > 25):
                trend = 'bearish'
        except:
            pass
        # 6. Üst Zaman Dilimi Analizi
        higher_trend = None
        if higher_df is not None and higher_timeframe:
            try:
                higher_df.columns = [col.lower() for col in df.columns]

                higher_close = higher_df[f'close_{higher_timeframe}']
                higher_ma_fast = EMA(higher_close, 50)
                higher_ma_slow = EMA(higher_close, 200)
                
                higher_trend = "bullish" if (higher_close.iloc[-1] > higher_ma_fast.iloc[-1] > higher_ma_slow.iloc[-1]) else \
                              "bearish" if (higher_close.iloc[-1] < higher_ma_fast.iloc[-1] < higher_ma_slow.iloc[-1]) else \
                              "sideways"
            except:
                pass

        # 7. Bollinger Bant Genişliği
        bb_period = 20
        bb_std = 2
        try:
            basis = close.rolling(bb_period).mean()
            dev = close.rolling(bb_period).std()
            bb_width = ((basis + (bb_std * dev)) - (basis - (bb_std * dev))) / basis
            low_volatility = bb_width.iloc[-1] < 0.05
        except:
            bb_width = pd.Series([np.nan]*len(close))
            low_volatility = False

        # 8. Breakout Tespiti
        lookback = 20
        try:
            recent_high = high.rolling(lookback).max().iloc[-1]
            recent_low = low.rolling(lookback).min().iloc[-1]
            breakout_up = close.iloc[-1] > recent_high * 1.005  # %0.5 filtre
            breakout_down = close.iloc[-1] < recent_low * 0.995
        except:
            breakout_up = breakout_down = False

        # 9. Nihai Regime Belirleme
        regime = determine_regime(
            trend=trend,
            adx_value=adx.iloc[-1],
            bb_width=bb_width.iloc[-1],
            close=close,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            higher_trend=higher_trend,
            breakout_up=breakout_up,
            breakout_down=breakout_down,
            low_volatility=low_volatility
        )
        # 11. Entegre Veri Toplama
        async with aiohttp.ClientSession() as session:
            inst_data = await fetch_institutional_data(session)
            vol_analysis = enhanced_volume_analysis(df,timeframe)

        # 12. Gelişmiş Sinyal Mantığı
        signal_boost = 0
        if inst_data:
            if inst_data['net_flow'] < -1000:  # 1000 BTC+ borsa çıkışı
                signal_boost += 15
            if inst_data['oi_change'] > 0.05: # %5 OI artışı
                signal_boost += 20
        
        if abs(vol_analysis['volume_zscore']) > 2:
            signal_boost += 10

               # Sinyalleri ana çıktıya ekle
        signal_info = generate_signals(high, low, close, volume, rsi, open_)
        # ... [Mevcut sinyal üretim kısmına ekle] ...
        signal_info['signal_confidence'] = min(100, signal_info['signal_confidence'] + signal_boost)

        return {
            **{  # Önceki tüm değerler
                'symbol': symbol,
                'timeframe': timeframe,
                'regime': regime,
                'trend': trend,
                'adx': round(adx.iloc[-1], 1),
                'rsi': round(rsi.iloc[-1], 1),
                'fakeout_signal': fakeout_signal,
                'fakeout_confidence': fakeout_conf,
                'volume_status': 'high' if (volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5) else 'normal',
                'last_close': round(last_close, 4),
                'condition_summary': {
                    'ma_cross': 'bullish' if ma_fast.iloc[-1] > ma_slow.iloc[-1] else 'bearish',
                    'adx_strength': 'strong' if adx.iloc[-1] > 25 else 'weak',
                    'rsi_status': 'overbought' if rsi.iloc[-1] > 70 else 'oversold' if rsi.iloc[-1] < 30 else 'neutral'
                }
            },
            **signal_info,
            'institutional_flow': inst_data,
            'volume_analysis': vol_analysis,
          # Sinyal bilgileri
        }

    except Exception as e:
        print(f"Critical Error: {str(e)}")  
        traceback.print_exc()
        return {'error': str(e)}



########################################
# 2) Çoklu Zaman Dilimi Fonksiyonları
########################################
async def get_higher_tf(tf):
    mapping = {
        '5m': '30m',
        '15m': '1h',
        '30m': '4h',
        '1h': '4h',
        '4h': '1d',
        '1d': "1w"
    }
    return mapping.get(tf)

async def get_all_regimes(symbol,df_5m, df_15m, df_30m, df_1h, df_4h, df_1d, df_1w, optimize_params=False):
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
            print(df.columns)
            higher_tf = await get_higher_tf(tf)
            optimized_params=None
            if optimize_params:
                optimized_params = optimize_parameters(df, tf)
                print("Optimize Edilmiş Parametreler:")
                print(optimized_params)
            if higher_tf:
                df_higher = tf_dfs[higher_tf]
                out = await detect_regime(symbol,df, tf, higher_df=df_higher, higher_timeframe=higher_tf, optimize_params=optimize_params, param_override=optimized_params)
            else:
                out = await detect_regime(symbol,df, tf, optimize_params=optimize_params)
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
