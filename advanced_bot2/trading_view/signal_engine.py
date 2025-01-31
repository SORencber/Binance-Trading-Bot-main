# signal_engine.py

import pandas as pd
import numpy as np

from pivots_waves import AdvancedPivotScanner, build_zigzag_wave
from pattern_detector import detect_all_patterns
from ml_model import PatternEnsembleModel
from config import PATTERN_PARAMS, SYSTEM_PARAMS

def check_breakout_volume(df, price_col="Close", vol_col="Volume"):
    if "ATR" not in df.columns:
        df["H-L"]=df["High"]-df["Low"]
        df["H-PC"]=(df["High"]-df["Close"].shift(1)).abs()
        df["L-PC"]=(df["Low"]-df["Close"].shift(1)).abs()
        df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1)
        df["ATR"]= df["TR"].rolling(14).mean()

    last_close= df[price_col].iloc[-1]
    prev_close= df[price_col].iloc[-2] if len(df)>1 else last_close
    last_atr  = df["ATR"].iloc[-1] if not pd.isna(df["ATR"].iloc[-1]) else 0
    breakout_up   = (last_close - prev_close) > (1.0 * last_atr)
    breakout_down = (prev_close - last_close)> (1.0 * last_atr)

    volume_spike=False
    if vol_col in df.columns and len(df)>20:
        v_now = df[vol_col].iloc[-1]
        v_mean= df[vol_col].rolling(20).mean().iloc[-2]
        volume_spike= (v_now> 1.5*v_mean)

    return breakout_up, breakout_down, volume_spike


def generate_signals(df, price_col="Close", vol_col="Volume", ml_model=None):
    # 1) pivot
    sc = AdvancedPivotScanner(left_bars=SYSTEM_PARAMS["pivot_left_bars"],
                             right_bars=SYSTEM_PARAMS["pivot_right_bars"],
                             volume_filter=SYSTEM_PARAMS["volume_filter"],
                             min_atr_factor=SYSTEM_PARAMS["min_atr_factor"],
                             df=df)
    pivots= sc.find_pivots(df[price_col])
    wave = build_zigzag_wave(pivots, df=df, min_wave_atr=SYSTEM_PARAMS["min_atr_factor"])

    # 2) pattern
    patterns = detect_all_patterns(pivots, wave, df=df, config=PATTERN_PARAMS)

    # 3) ML
    ml_label= None
    if ml_model:
        feats= ml_model.extract_features(wave)
        pred = ml_model.predict(feats)
        ml_label= pred[0]

    # 4) breakout/volume
    b_up,b_down,v_spike= check_breakout_volume(df, price_col, vol_col)

    # 5) final
    signal="HOLD"
    reason=[]
    if patterns["elliott"] or patterns["wolfe"]:
        signal="BUY"
        reason.append("ElliottOrWolfe")
    if patterns["harmonic"]:
        signal="SELL"
        reason.append("Harmonic")
    if len(patterns["headshoulders"])>0:
        signal="SELL"
        reason.append("HeadShoulders")
    if len(patterns["inverse_headshoulders"])>0:
        signal="BUY"
        reason.append("InverseHeadShoulders")
    if len(patterns["double_top"])>0:
        signal="SELL"
        reason.append("DoubleTop")
    if len(patterns["double_bottom"])>0:
        signal="BUY"
        reason.append("DoubleBottom")
    if patterns["triangle"] and b_up:
        signal="BUY"
        reason.append("Triangle_UpBreak")
    elif patterns["triangle"] and b_down:
        signal="SELL"
        reason.append("Triangle_DownBreak")
    if patterns["wedge"] and b_up:
        signal="BUY"
        reason.append("Wedge_Up")
    elif patterns["wedge"] and b_down:
        signal="SELL"
        reason.append("Wedge_Down")

    # ML
    if ml_label==1:
        signal="BUY"
        reason.append("MLLabel1")
    elif ml_label==2:
        signal="SELL"
        reason.append("MLLabel2")

    # volume + breakout teyit
    if signal=="BUY" and b_up and v_spike:
        reason.append("BreakoutUp_VolSpike")
    elif signal=="SELL" and b_down and v_spike:
        reason.append("BreakoutDown_VolSpike")

    reason_str = "_".join(reason) if reason else "NONE"
    return {
        "signal": signal,
        "reason": reason_str,
        "patterns": patterns,
        "ml_label": ml_label,
        "breakout_up": b_up,
        "breakout_down": b_down,
        "volume_spike": v_spike
    }
