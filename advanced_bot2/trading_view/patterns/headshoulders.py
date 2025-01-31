# patterns/headshoulders.py
import pandas as pd
from core.logging_setup import log

def get_col_name(base_col: str, time_frame: str) -> str:
    """
    Örnek: get_col_name("High", "5m") -> "High_5m"
    """
    return f"{base_col}_{time_frame}"

def detect_head_and_shoulders_advanced(
  df: pd.DataFrame,
    time_frame: str="1m",
    left_bars: int=10,
    right_bars: int=10,
    min_distance_bars: int=10,
    shoulder_tolerance: float=0.03,
    volume_decline: bool=True,
    neckline_break: bool=True,
    max_shoulder_width_bars: int=50,
    atr_filter: float=0.0
) -> list:
    

    """
    Gelişmiş Head & Shoulders dedektörü.
    (Önce gösterdiğimiz advanced versiyon.)
    Dönen => list[dict]
    """
    # Kolon isimlerini ayarla
    high_col   = f"High_{time_frame}"
    low_col    = f"Low_{time_frame}"
    close_col  = f"Close_{time_frame}"
    volume_col = f"Volume_{time_frame}"
    atr_col    = f"ATR_{time_frame}"

    for needed in [high_col, low_col, close_col]:
        if needed not in df.columns:
            raise ValueError(f"Missing '{needed}' column for {time_frame}")

    # ATR istersen => ekleyelim
    if atr_filter>0:
        if atr_col not in df.columns:
            df[f"H-L_{time_frame}"]  = df[high_col] - df[low_col]
            df[f"H-PC_{time_frame}"] = (df[high_col] - df[close_col].shift(1)).abs()
            df[f"L-PC_{time_frame}"] = (df[low_col]  - df[close_col].shift(1)).abs()
            df[f"TR_{time_frame}"]   = df[[f"H-L_{time_frame}",
                                           f"H-PC_{time_frame}",
                                           f"L-PC_{time_frame}"]].max(axis=1)
            df[atr_col] = df[f"TR_{time_frame}"].rolling(14).mean()

    # Pivot tespiti => local max
    pivot_list = []
    price_series = df[close_col]
    n= len(df)
    for i in range(left_bars, n-right_bars):
        val= price_series.iloc[i]
        left_sl= price_series.iloc[i-left_bars: i]
        right_sl= price_series.iloc[i+1: i+1+right_bars]
        if all(val> l for l in left_sl) and all(val>=r for r in right_sl):
            # ATR check
            if atr_filter>0 and atr_col in df.columns:
                pivot_atr= df[atr_col].iloc[i]
                if not pd.isna(pivot_atr):
                    c_left= price_series.iloc[i-1] if i>0 else val
                    c_right= price_series.iloc[i+1] if i+1< len(df) else val
                    if (abs(val-c_left)<(atr_filter*pivot_atr)) or (abs(val-c_right)<(atr_filter*pivot_atr)):
                        continue
            pivot_list.append( (i,val,+1) )

    results=[]
    top_pivots= pivot_list
    for i in range(len(top_pivots)-2):
        LS= top_pivots[i]
        H = top_pivots[i+1]
        RS= top_pivots[i+2]

        idxL, priceL, _ = LS
        idxH, priceH, _ = H
        idxR, priceR, _ = RS

        if not (idxL< idxH< idxR):
            continue

        if not (priceH> priceL and priceH> priceR):
            continue

        bars_LH= idxH- idxL
        bars_HR= idxR- idxH
        if bars_LH< min_distance_bars or bars_HR< min_distance_bars:
            continue
        if bars_LH> max_shoulder_width_bars or bars_HR> max_shoulder_width_bars:
            continue

        diffShoulder= abs(priceL- priceR)/(priceH+1e-9)
        if diffShoulder> shoulder_tolerance:
            continue

        # Neckline => en düşük dip L->H, ve en düşük dip H->R
        # Basit => Low_ segment
        segment_LH = df[low_col].iloc[idxL: idxH+1]
        segment_HR = df[low_col].iloc[idxH: idxR+1]
        if len(segment_LH)<1 or len(segment_HR)<1:
            continue
        dip1_idx= segment_LH.idxmin()
        dip2_idx= segment_HR.idxmin()
        dip1_price= df[low_col].iloc[dip1_idx]
        dip2_price= df[low_col].iloc[dip2_idx]

        confirmed= False
        if neckline_break:
            if dip2_idx != dip1_idx:
                slope= (dip2_price- dip1_price)/(dip2_idx- dip1_idx)
                intercept= dip1_price- slope*dip1_idx
                last_i= len(df)-1
                last_close= df[close_col].iloc[-1]
                line_y= slope* last_i + intercept
                if last_close< line_y:
                    confirmed=True

        vol_check= True
        if volume_decline and volume_col in df.columns:
            volL= df[volume_col].iloc[idxL] if idxL<len(df) else None
            volH= df[volume_col].iloc[idxH] if idxH<len(df) else None
            volR= df[volume_col].iloc[idxR] if idxR<len(df) else None
            if (volL is not None) and (volH is not None) and (volR is not None):
                if not (volH< volL and volH< volR):
                    vol_check= False

        res= {
            "L": (idxL,priceL),
            "H": (idxH,priceH),
            "R": (idxR,priceR),
            "bars_L-H": bars_LH,
            "bars_H-R": bars_HR,
            "shoulder_diff": diffShoulder,
            "neckline": ((dip1_idx,dip1_price),(dip2_idx,dip2_price)),
            "confirmed": confirmed,
            "volume_check": vol_check
        }
        results.append(res)

    return results


def detect_inverse_head_and_shoulders_advanced(
    df: pd.DataFrame,
    time_frame: str = "1m",
    left_bars: int = 10,
    right_bars: int = 10,
    min_bar_distance: int = 10,
    shoulder_tolerance: float = 0.03,
    volume_decline: bool = True,
    neckline_break: bool = True,
    max_shoulder_width_bars: int = 50,
    atr_filter: float = 0.0
) -> list:
    """
    Gelişmiş Inverse Head & Shoulders dedektörü.

    Parametreler:
    -------------
    df: pd.DataFrame
       "High_{tf}, Low_{tf}, Close_{tf}, Volume_{tf}" kolonları beklenir.
    time_frame: str
       "1m","1h" vb.
    left_bars, right_bars: int
       Local min bulma için solda/sağda bar sayısı.
    min_bar_distance: int
       Sol omuz->baş, baş->sağ omuz arası min bar farkı.
    shoulder_tolerance: float
       Sol ve sağ omuz diplerinin fiyat farkı / baş’a oranla ne kadar?
    volume_decline: bool
       Omuzlardan başa doğru hacim “farklı” ya da “başta azalan” vs. kural.
    neckline_break: bool
       Formasyon onayı => son fiyat, boyun çizgisinin üstüne çıktı mı?
    max_shoulder_width_bars: int
       Omuz->baş arası bar sayısı çok büyükse formasyonu “degrade” edebiliriz.
    atr_filter: float
       0 => kapalı; >0 => pivotların ATR eşiği.

    Dönüş:
    ------
    results: list[dict]
      [
        {
          "found": True,
          "msgs": [...],
          "L": (idxL, priceL),
          "H": (idxH, priceH),
          "R": (idxR, priceR),
          "bars_L-H": int,
          "bars_H-R": int,
          "shoulder_diff": float,
          "neckline": ((idxN1, prcN1), (idxN2, prcN2)),
          "confirmed": bool,
          "volume_check": bool
        },
        ...
      ]
    """
    # 1) Kolon isimleri
    low_col    = f"Low_{time_frame}"
    close_col  = f"Close_{time_frame}"
    volume_col = f"Volume_{time_frame}"
    atr_col    = f"ATR_{time_frame}"

    for needed in [low_col, close_col]:
        if needed not in df.columns:
            # kolon yok => direkt boş liste
            return []

    # 2) ATR hesaplaması (opsiyonel)
    if atr_filter>0:
        if atr_col not in df.columns:
            # basit ATR ekle
            high_col = f"High_{time_frame}"
            if high_col not in df.columns:
                return []
            df[f"H-L_{time_frame}"]  = df[high_col] - df[low_col]
            df[f"H-PC_{time_frame}"] = (df[high_col] - df[close_col].shift(1)).abs()
            df[f"L-PC_{time_frame}"] = (df[low_col]  - df[close_col].shift(1)).abs()
            df[f"TR_{time_frame}"]   = df[[f"H-L_{time_frame}",
                                           f"H-PC_{time_frame}",
                                           f"L-PC_{time_frame}"]].max(axis=1)
            df[atr_col] = df[f"TR_{time_frame}"].rolling(14).mean()

    # 3) Dip pivot tespiti => local min (left_bars,right_bars)
    pivot_list = []
    price_series = df[close_col]  # DIP => close price al
    n = len(df)

    for i in range(left_bars, n-right_bars):
        val= price_series.iloc[i]
        left_slice = price_series.iloc[i-left_bars : i]
        right_slice= price_series.iloc[i+1 : i+1+ right_bars]
        if all(val< l for l in left_slice) and all(val<= r for r in right_slice):
            # ATR check => eğer atr_filter>0 ise, pivotun 
            if atr_filter>0 and atr_col in df.columns:
                pivot_atr= df[atr_col].iloc[i]
                if not pd.isna(pivot_atr):
                    c_left= price_series.iloc[i-1] if i>0 else val
                    c_right= price_series.iloc[i+1] if i+1< n else val
                    # basit => val (dip) c_left c_right difference > atr_filter * pivot_atr
                    # Kafanıza göre kural
                    diff_left= abs(val- c_left)
                    diff_right= abs(val- c_right)
                    if diff_left< atr_filter*pivot_atr and diff_right< atr_filter*pivot_atr:
                        continue
            pivot_list.append((i, val, -1))

    # 4) Ardışık 3 dip => L, H, R => inverse H&S => Head en dip
    results=[]
    if len(pivot_list)<3:
        return results

    # degrade approach => eğer bir kural fail => msgs’e ekle, formasyonu iptal etmemek
    # (her formasyon dict’de “msgs” tutarız)
    for i in range(len(pivot_list)-2):
        L= pivot_list[i]
        H= pivot_list[i+1]
        R= pivot_list[i+2]

        idxL, priceL, _= L
        idxH, priceH, _= H
        idxR, priceR, _= R

        # msg list
        msgs= []
        found= True  # degrade

        # Sıra => L< H< R
        if not (idxL< idxH< idxR):
            # degrade => continue
            continue

        # Head => en dip => priceH < priceL, priceH< priceR
        if not (priceH< priceL and priceH< priceR):
            msgs.append("Head is not the lowest among the 3 dips.")
            # degrade => skip
            continue

        # bar mesafesi
        bars_LH= idxH- idxL
        bars_HR= idxR- idxH
        if bars_LH< min_bar_distance or bars_HR< min_bar_distance:
            msgs.append("bar distance fail => L->H or H->R < min_bar_distance")
            # degrade => skip
            continue
        if bars_LH> max_shoulder_width_bars or bars_HR> max_shoulder_width_bars:
            msgs.append("bar distance is too large => might degrade (shoulder width).")

        # Shoulder tolerance => (|L - R| / Head)
        diffShoulder= abs(priceL- priceR)/ ((priceL+ priceR)/2+1e-9)
        # ya da (|L-R| / priceH) de yapılabilir
        if diffShoulder> shoulder_tolerance:
            msgs.append(f"Shoulder tolerance fail => diff={diffShoulder:.3f}")
            # degrade => skip
            continue

        # Volume check => volume_decline => sol omuz ve sağ omuz > head
        vol_check= True
        if volume_decline and volume_col in df.columns:
            volL= df[volume_col].iloc[idxL]
            volH= df[volume_col].iloc[idxH]
            volR= df[volume_col].iloc[idxR]
            # basit => Head hacmi < omuz hacimleri
            if not (volH< volL and volH< volR):
                msgs.append("volume check fail => Head volume not the lowest")
                vol_check= False

        # Neckline => bulmak için => L->H segmentte local max? + H->R segmentte local max?
        # Basit => L->H arasındaki en yüksek bar => T1, H->R arasındaki en yüksek => T2
        # boyun => T1->T2
        segment_LH = df[close_col].iloc[idxL : idxH+1]
        segment_HR = df[close_col].iloc[idxH : idxR+1]
        if len(segment_LH)<1 or len(segment_HR)<1:
            # degrade => skip
            continue
        T1_idx = segment_LH.idxmax()
        T2_idx = segment_HR.idxmax()

        T1_price = df[close_col].iloc[T1_idx]
        T2_price = df[close_col].iloc[T2_idx]
        # (T1_idx, T1_price), (T2_idx, T2_price)

        # confirm => last price above that line?
        confirmed= False
        if neckline_break and close_col in df.columns:
            from math import isclose
            def line_eq(xA,yA,xB,yB):
                if (xB-xA)==0:
                    return None,None
                m_= (yB-yA)/(xB-xA)
                b_= yA - m_*xA
                return m_,b_

            mT, bT = line_eq(T1_idx, T1_price, T2_idx, T2_price)
            if mT is not None:
                last_close= df[close_col].iloc[-1]
                last_i= len(df)-1
                line_y= mT* last_i + bT
                if last_close> line_y:
                    confirmed= True
                else:
                    msgs.append("neckline break fail => last_close below neckline")

        # Kaydet
        res= {
          "found": found,
          "msgs": msgs,
          "L": (idxL, priceL),
          "H": (idxH, priceH),
          "R": (idxR, priceR),
          "bars_L-H": bars_LH,
          "bars_H-R": bars_HR,
          "shoulder_diff": diffShoulder,
          "volume_check": vol_check,
          "confirmed": confirmed,
          # boyun çizgisi => (T1, T2)
          "neckline": ((T1_idx, T1_price), (T2_idx, T2_price))
        }
        results.append(res)

    return results
