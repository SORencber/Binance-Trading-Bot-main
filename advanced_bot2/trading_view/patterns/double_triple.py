# patterns/double_triple.py
import pandas as pd
from core.logging_setup import log

def get_col_name(base_col: str, time_frame: str) -> str:
    """
    Örnek: get_col_name("High", "5m") -> "High_5m"
    """
    return f"{base_col}_{time_frame}"


def detect_double_top(
    pivots,
    time_frame="1m",
    tolerance=0.01,
    min_distance_bars=20,
    triple_variation=True,
    volume_check=False,
    neckline_break=False,
    df=None
):
    """
    Double (ve triple) top dedektörü.
    Artık time_frame bazlı kolonlar (Volume_{time_frame}, Close_{time_frame}) kullanıyoruz.
    """
    top_pivots = [p for p in pivots if p[2]== +1]
    if len(top_pivots)<2:
        return []

    volume_col = get_col_name("Volume", time_frame)
    close_col  = get_col_name("Close", time_frame)

    results = []
    i=0
    while i< len(top_pivots)-1:
        t1= top_pivots[i]
        t2= top_pivots[i+1]

        idx1, price1 = t1[0], t1[1]
        idx2, price2 = t2[0], t2[1]
        avgp = (price1+ price2)/2
        pdiff = abs(price1-price2)/(avgp+1e-9)
        bar_diff = idx2- idx1

        if pdiff< tolerance and bar_diff>= min_distance_bars:
            found_pattern = {
                "tops": [(idx1,price1),(idx2,price2)],
                "neckline": None,
                "confirmed":False,
                "start_bar":idx1,
                "end_bar": idx2,
                "pattern":"double_top"
            }
            used_third = False
            # triple
            if triple_variation and (i+2< len(top_pivots)):
                t3= top_pivots[i+2]
                idx3, price3= t3[0], t3[1]
                pdiff3= abs(price2- price3)/ ((price2+price3)/2+1e-9)
                bar_diff3= idx3- idx2
                if pdiff3< tolerance and bar_diff3>= min_distance_bars:
                    found_pattern["tops"] = [(idx1,price1),(idx2,price2),(idx3,price3)]
                    found_pattern["end_bar"] = idx3
                    found_pattern["pattern"] = "triple_top"
                    used_third=True

            # volume check
            if volume_check and df is not None and volume_col in df.columns:
                try:
                    vol1= df[volume_col].iloc[idx1]
                    vol2= df[volume_col].iloc[idx2]
                    if not (vol2< vol1):
                        i += (2 if used_third else 1)
                        continue
                except:
                    pass

            # Neckline => en alçak dip pivotu (idx1< dip< idx2)
            seg_end = t2[0] if not used_third else top_pivots[i+2][0]
            dip_segment = [p for p in pivots if p[2]== -1 and p[0]> idx1 and p[0]< seg_end]
            if dip_segment:
                dip_sorted = sorted(dip_segment, key=lambda x: x[1])
                neck = dip_sorted[0]
                found_pattern["neckline"] = (neck[0], neck[1])

            # Confirm => "Close_{time_frame}" altına indi mi
            if neckline_break and found_pattern["neckline"] is not None and df is not None and close_col in df.columns:
                dip_idx, dip_price = found_pattern["neckline"]
                last_close = df[close_col].iloc[-1]
                if last_close< dip_price:
                    found_pattern["confirmed"] = True
                    found_pattern["end_bar"] = len(df)-1

            results.append(found_pattern)

            i += (2 if used_third else 1)
        else:
            i+=1

    return results


def detect_double_bottom(
    pivots,
    time_frame="1m",
    tolerance=0.01,
    min_distance_bars=20,
    triple_variation=True,
    volume_check=False,
    neckline_break=False,
    df=None
):
    """
    Double (ve triple) bottom dedektörü.
    """
    bottom_pivots = [p for p in pivots if p[2]== -1]
    if len(bottom_pivots)<2:
        return []

    volume_col = get_col_name("Volume", time_frame)
    close_col  = get_col_name("Close", time_frame)

    results=[]
    i=0
    while i< len(bottom_pivots)-1:
        b1= bottom_pivots[i]
        b2= bottom_pivots[i+1]
        idx1, price1= b1[0], b1[1]
        idx2, price2= b2[0], b2[1]
        avgp = (price1+ price2)/2
        pdiff= abs(price1- price2)/(avgp+1e-9)
        bar_diff= idx2- idx1

        if pdiff< tolerance and bar_diff>= min_distance_bars:
            found= {
                "bottoms": [(idx1,price1),(idx2,price2)],
                "neckline": None,
                "confirmed": False,
                "start_bar": idx1,
                "end_bar": idx2,
                "pattern":"double_bottom"
            }
            used_third=False
            # triple?
            if triple_variation and (i+2< len(bottom_pivots)):
                b3= bottom_pivots[i+2]
                idx3, price3 = b3[0], b3[1]
                pdiff3= abs(price2- price3)/ ((price2+price3)/2+1e-9)
                bar_diff3= idx3- idx2
                if pdiff3< tolerance and bar_diff3>= min_distance_bars:
                    found["bottoms"] = [(idx1,price1),(idx2,price2),(idx3,price3)]
                    found["end_bar"] = idx3
                    found["pattern"] = "triple_bottom"
                    used_third= True

            # volume check => ...
            # neckline break => ...
            # (Ekleyebilirsiniz)

            results.append(found)
            i+= (2 if used_third else 1)
        else:
            i+=1

    return results


