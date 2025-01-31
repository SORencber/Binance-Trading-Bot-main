import pandas as pd
import numpy as np
import random



##################################################
# 1) v6 Fonksiyon
##################################################

def analyze_trends_and_signals_v6(df: pd.DataFrame) -> dict:
    """
    v6 => MTF + synergy + onchain + macro + sentiment + 100+ indicator
    """
    if len(df)<10:
        return {
            "immediate_action": 2,
            "immediate_reason": "NOT_ENOUGH_DATA",
            "total_score": 0.0,
            "detail_scores": {},
            "regime": "NONE",
            "delayed_signals": []
        }

    row = df.iloc[-1]

    # MTF skor (örnek: 1m, 5m, 4h, 1d)
    sc_1m  = calc_tf_score_v6(row, "1m")
    sc_5m  = calc_tf_score_v6(row, "5m")
    sc_4h  = calc_tf_score_v6(row, "4h")
    sc_1d  = calc_tf_score_v6(row, "1d")

    # synergy => 100+ gosterge tarama
    synergy_val = evaluate_all_indicators_v6(row)

    # onchain / macro / sentiment
    onchain_s = calc_onchain_v6(row)
    macro_s   = calc_macro_v6(row)
    senti_s   = calc_sentiment_v6(row)

    # 4h rejim
    regime_4h = detect_regime_4h_v6(df, row)

    # baseMTF => 1m+5m + 2*(4h) + 2*(1d) gibi
    base_mtf  = (sc_1m + sc_5m) + 2*sc_4h + 2*sc_1d
    total_s   = base_mtf + synergy_val + onchain_s + macro_s + senti_s

    # confirm
    bull_conf, bear_conf = confirm_advanced_v6(df, row)

    # final
    final_act, reason_ = final_decision_v6(regime_4h, total_s, bull_conf, bear_conf)

    # delayed
    delayed_signals = generate_delayed_signals_v6(final_act, synergy_val, row)

    detail = {
        "sc_1m": sc_1m,
        "sc_5m": sc_5m,
        "sc_4h": sc_4h,
        "sc_1d": sc_1d,
        "synergy": synergy_val,
        "onchain": onchain_s,
        "macro": macro_s,
        "sentiment": senti_s
    }

    return {
        "immediate_action": final_act,
        "immediate_reason": reason_,
        "total_score": total_s,
        "detail_scores": detail,
        "regime": regime_4h,
        "delayed_signals": delayed_signals
    }


##################################################
# 2) Yardımcı Fonksiyonlar (Tek Tek Açıklama)
##################################################

def calc_tf_score_v6(row, tf: str) -> float:
    """1) RSI_{tf}, MACD_{tf}, MACDSig_{tf}, ADX_{tf}, Candle, Boll vb."""
    score = 0.0
    # RSI
    rsi_val = row.get(f"RSI_{tf}", 50)
    if rsi_val>60:   score+=1
    elif rsi_val<40: score-=1

    # MACD
    macd_  = row.get(f"MACD_{tf}", 0)
    macds_ = row.get(f"MACDSig_{tf}", 0)
    if macd_>macds_: score+=1
    else:            score-=1

    # ADX
    adx_ = row.get(f"ADX_{tf}", 0)
    if adx_>25:
        score+=1

    # Candle Engulf => +1/-1
    cdl_ = row.get(f"CDL_ENGULFING_{tf}", 0)
    if cdl_>0: score+=1
    elif cdl_<0: score-=1

    return score

def evaluate_all_indicators_v6(row: pd.Series) -> float:
    """
    Tüm "Power_*", "Divergence_*" vs. => synergy
    (Sınırsız ekleyebilirsiniz)
    """
    synergy = 0.0
    # Örnek weighting
    # 'Power_Renko': +1 if val>0, -1 if val<0 ...
    # 'Divergence_Bull': +2 if val>0 ...
    # vs.
    val_pr = row.get("Power_Renko", 0)
    if val_pr>0:   synergy +=1
    elif val_pr<0: synergy -=1

    val_pf = row.get("Power_Fractal", 0)
    if val_pf>0: synergy+=1
    elif val_pf<0: synergy-=1

    div_bull = row.get("Divergence_Bull", 0)
    if div_bull>0: synergy+=2
    elif div_bull<0: synergy-=2

    div_bear = row.get("Divergence_Bear", 0)
    if div_bear>0: synergy-=2  # (bear demek -2)
    # vs.

    return synergy

def calc_onchain_v6(row: pd.Series) -> float:
    score = 0.0
    mvrv = row.get("MVRV_Z_1d", 0)
    if mvrv<-0.5:  score+=2
    elif mvrv>0.7: score-=2
    fund = row.get("FundingRate", 0)
    if fund>0.01:  score+=1
    elif fund<-0.01:score-=1
    return score

def calc_macro_v6(row: pd.Series) -> float:
    """SP500, DXY, VIX => buraya ekleyebilirsiniz"""
    return 0.0

def calc_sentiment_v6(row: pd.Series) -> float:
    s = 0.0
    fgi = row.get("FearGreedIndex", 0.5)
    if fgi<0.3: s+=1
    elif fgi>0.7: s-=1
    news_ = row.get("NewsSentiment", 0)
    if news_>0.2: s+=1
    elif news_<-0.2: s-=1
    return s

def detect_regime_4h_v6(df: pd.DataFrame, row_4h) -> str:
    """ basit: ADX_4h, RSI_4h => 'TREND_UP/DOWN' / 'BREAKOUT_SOON' / 'RANGE' """
    adx_ = row_4h.get("ADX_4h", 0)
    rsi_ = row_4h.get("RSI_4h", 50)
    macd_ = row_4h.get("MACD_4h", 0)
    macds_= row_4h.get("MACDSig_4h", 0)

    # Boll, band pct...
    # Kısaltıyoruz
    if adx_>20:
        if rsi_>55 and macd_>macds_:
            return "TREND_UP"
        elif rsi_<45 and macd_<macds_:
            return "TREND_DOWN"
        else:
            return "TREND_FLAT"
    else:
        return "RANGE"

def confirm_advanced_v6(df: pd.DataFrame, row) -> (bool, bool):
    """
    5m volume spike, 4h T3 slope vb. => bull/bear confirm
    DEMO => random
    """
    return (False, False)

def final_decision_v6(regime: str, total_score: float, bull_conf: bool, bear_conf: bool) -> (int, str):
    act = 2
    reason = ""
    if regime=="TREND_UP":
        if total_score>2 and bull_conf:
            act=1
            reason="TREND_UP_BULLCONF"
        else:
            act=2
            reason="TREND_UP_BUT_WEAK"
    elif regime=="TREND_DOWN":
        if total_score<-2 and bear_conf:
            act=0
            reason="TREND_DOWN_BEARCONF"
        else:
            act=2
            reason="TREND_DOWN_BUT_WEAK"
    else:
        # RANGE vs.
        if total_score>5 and bull_conf:
            act=1
            reason="RANGE_BULLISH"
        elif total_score<-5 and bear_conf:
            act=0
            reason="RANGE_BEARISH"
        else:
            act=2
            reason="SIDEWAYS"
    return (act, reason)

def generate_delayed_signals_v6(final_act: int, synergy: float, row) -> list:
    """
    Dinamik bar gecikmesi => synergy
    """
    signals = []
    if final_act==1:  # BUY
        d1 = dynamic_delay_based_on_synergy(synergy)
        signals.append({
            "bars_delay": d1,
            "action": "SELL",
            "qty_ratio": 0.5,
            "condition": {
                "type": "SYNERGY",
                "operator": ">",
                "value": synergy+1
            },
            "reason": f"Partial SELL if synergy>{synergy+1}"
        })
    elif final_act==0: # SELL
        d2 = dynamic_delay_based_on_synergy(synergy)
        signals.append({
            "bars_delay": d2,
            "action": "BUY",
            "qty_ratio": 0.3,
            "condition": {
                "type": "SYNERGY",
                "operator": "<",
                "value": synergy-2
            },
            "reason": f"Contrarian BUY if synergy<{synergy-2}"
        })
    return signals

def dynamic_delay_based_on_synergy(val: float) -> int:
    if val>5:   return 1
    elif val>2: return 2
    elif val>0: return 3
    else:       return 5

