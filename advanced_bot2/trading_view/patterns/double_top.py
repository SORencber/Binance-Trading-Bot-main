############################
# DOUBLE TOP / BOTTOM
############################
import pandas as pd
def detect_double_top(
    df: pd.DataFrame,
    pivots,
    time_frame:str="1m",
    tolerance: float=0.01,
    min_distance_bars: int=20,
    triple_variation: bool=True,
    volume_check: bool=False,
    neckline_break: bool=False,
    check_retest: bool=False,
    retest_tolerance: float=0.01
):
    """
    Double veya opsiyonel triple-top varyasyonu.
    """
    top_pivots= [p for p in pivots if p[2]== +1]
    if len(top_pivots)<2:
        return []

    volume_col= get_col_name("Volume", time_frame)
    close_col = get_col_name("Close", time_frame)

    results=[]
    i=0
    while i< len(top_pivots)-1:
        t1= top_pivots[i]
        t2= top_pivots[i+1]
        idx1,price1= t1[0], t1[1]
        idx2,price2= t2[0], t2[1]

        avgp= (price1+ price2)/2
        pdiff= abs(price1- price2)/(avgp+1e-9)
        bar_diff= idx2- idx1

        if pdiff< tolerance and bar_diff>= min_distance_bars:
            found= {
              "pattern": "double_top",
              "tops": [(idx1,price1),(idx2,price2)],
              "neckline": None,
              "confirmed": False,
              "start_bar": idx1,
              "end_bar": idx2,
              "retest_info": None
            }
            used_third= False
            if triple_variation and (i+2< len(top_pivots)):
                t3= top_pivots[i+2]
                idx3,price3= t3[0], t3[1]
                pdiff3= abs(price2- price3)/ ((price2+price3)/2+1e-9)
                bar_diff3= idx3- idx2
                if pdiff3< tolerance and bar_diff3>= min_distance_bars:
                    found["tops"]= [(idx1,price1),(idx2,price2),(idx3,price3)]
                    found["end_bar"]= idx3
                    found["pattern"]= "triple_top"
                    used_third= True

            if volume_check and volume_col in df.columns:
                vol1= df[volume_col].iloc[idx1]
                vol2= df[volume_col].iloc[idx2]
                if vol2 > (vol1*0.8):
                    i+= (2 if used_third else 1)
                    continue

            if neckline_break and close_col in df.columns:
                seg_end= found["end_bar"]
                dips_for_neck = [pp for pp in pivots if pp[2]== -1 and (pp[0]> idx1 and pp[0]< seg_end)]
                if dips_for_neck:
                    dips_sorted= sorted(dips_for_neck, key=lambda x: x[1])  # ascending price
                    neck = dips_sorted[0]
                    found["neckline"]= (neck[0], neck[1])
                    last_close= df[close_col].iloc[-1]
                    if last_close< neck[1]:
                        found["confirmed"]= True
                        found["end_bar"]= len(df)-1
                        if check_retest:
                            retest_info= _check_retest_doubletop(
                                df, time_frame,
                                neckline_price= neck[1],
                                confirm_bar= len(df)-1,
                                retest_tolerance= retest_tolerance
                            )
                            found["retest_info"]= retest_info

            results.append(found)
            i+= (2 if used_third else 1)
        else:
            i+=1
    return results


def _check_retest_doubletop(df: pd.DataFrame,
                            time_frame: str,
                            neckline_price: float,
                            confirm_bar: int,
                            retest_tolerance: float=0.01) -> dict:
    close_col= get_col_name("Close", time_frame)
    if close_col not in df.columns:
        return {"retest_done": False, "retest_bar": None}

    n= len(df)
    if confirm_bar>= n-1:
        return {"retest_done": False, "retest_bar": None}

    for i in range(confirm_bar+1, n):
        c= df[close_col].iloc[i]
        dist_ratio= abs(c - neckline_price)/(abs(neckline_price)+1e-9)
        if dist_ratio<= retest_tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "retest_price": c,
                "dist_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None}


def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

