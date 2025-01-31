# patterns/harmonics.py
import pandas as pd
from core.logging_setup import log

def get_col_name(base_col: str, time_frame: str) -> str:
    """
    Örnek: get_col_name("High", "5m") -> "High_5m"
    """
    return f"{base_col}_{time_frame}"

def detect_harmonic_multiple(
    df: pd.DataFrame,
    time_frame: str = "1h",
    left_bars: int = 5,
    right_bars: int = 5,
    fib_tolerance: float = 0.02,
    check_volume: bool = False,
    patterns: list = None
):
    """
    Çoklu Harmonic Pattern Dedektörü (Gartley, Bat, Crab, Butterfly, Shark, Cipher).
    5 pivot (X->A->B->C->D) + bullish/bearish tespiti + fib aralıkları.
    Dönüş => True/False yerine basit => (bulduysa) True, yoksa False. 
    (Eski 'detect_harmonic_pattern' bool döndürüyordu. 
     Burada basit tutmak adına yine bool döndürebiliriz. 
     Ama advanced => list pattern match de yapabilir.)
    """
    # Dilerseniz "list of pattern found" dönebilirsiniz, 
    # ya da "True/False" => en az bir pattern match.
    # Burada basit => "bulduysak True, aksi halde False"

    high_col  = f"High_{time_frame}"
    low_col   = f"Low_{time_frame}"
    close_col = f"Close_{time_frame}"
    vol_col   = f"Volume_{time_frame}"

    if patterns is None:
        patterns= ["gartley","bat","crab","butterfly","shark","cipher"]

    for c in [high_col,low_col,close_col]:
        if c not in df.columns:
            return False

    price_series= df[close_col]
    n= len(df)
    if n< left_bars+ right_bars+ 5:
        return False

    # 1) pivot bul
    pivot_list= []
    def is_local_max(i):
        val= price_series.iloc[i]
        left_sl= price_series.iloc[i-left_bars: i]
        right_sl= price_series.iloc[i+1: i+1+ right_bars]
        return all(val> l for l in left_sl) and all(val>= r for r in right_sl)

    def is_local_min(i):
        val= price_series.iloc[i]
        left_sl= price_series.iloc[i-left_bars: i]
        right_sl= price_series.iloc[i+1: i+1+ right_bars]
        return all(val< l for l in left_sl) and all(val<= r for r in right_sl)

    for i in range(left_bars, n- right_bars):
        if is_local_max(i):
            pivot_list.append( (i,price_series.iloc[i], +1) )
        elif is_local_min(i):
            pivot_list.append( (i,price_series.iloc[i], -1) )

    # zigzag
    pivot_list_sorted= sorted(pivot_list, key=lambda x: x[0])
    zigzag= [pivot_list_sorted[0]]
    for i in range(1, len(pivot_list_sorted)):
        curr= pivot_list_sorted[i]
        prev= zigzag[-1]
        if curr[2]== prev[2]:
            if curr[2]== +1:
                if curr[1]> prev[1]:
                    zigzag[-1]= curr
            else:
                if curr[1]< prev[1]:
                    zigzag[-1]= curr
        else:
            zigzag.append(curr)

    # 2) pattern specs
    harmonic_specs= {
        "gartley": {
            "AB_XA": (0.618, 0.618),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.13, 1.618)
        },
        "bat": {
            "AB_XA": (0.382, 0.5),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.618, 2.618)
        },
        "crab": {
            "AB_XA": (0.382, 0.618),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (2.24, 3.618)
        },
        "butterfly": {
            "AB_XA": (0.786, 0.786), 
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.618, 2.24)
        },
        "shark": {
            "AB_XA": (0.886,1.13),
            "BC_AB": (1.13, 1.618),
            "CD_BC": (0.886,1.13)
        },
        "cipher": {
            "AB_XA": (0.382,0.618),
            "BC_AB": (1.27,2.0),
            "CD_BC": (1.13,1.414)
        }
    }

    def ratio_in_range(r, rng, tol):
        (mn,mx)= rng
        if abs(mn- mx)< 1e-9:
            return (abs(r- mn)<= mn* tol)
        else:
            lo= mn- mn* tol
            hi= mx+ mx* tol
            return (r>=lo and r<= hi)

    # 3) 5 li pivot => X->A->B->C->D
    # bull => -1,+1,-1,+1,-1
    # bear => +1,-1,+1,-1,+1
    found_any= False
    i= 0
    while i< (len(zigzag)-4):
        X= zigzag[i]
        A= zigzag[i+1]
        B= zigzag[i+2]
        C= zigzag[i+3]
        D= zigzag[i+4]

        idxX, pxX, tX= X
        idxA, pxA, tA= A
        idxB, pxB, tB= B
        idxC, pxC, tC= C
        idxD, pxD, tD= D

        if not (idxX< idxA< idxB< idxC< idxD):
            i+=1
            continue

        is_bull= (tX==-1 and tA==+1 and tB==-1 and tC==+1 and tD==-1)
        is_bear= (tX==+1 and tA==-1 and tB==+1 and tC==-1 and tD==+1)
        if not (is_bull or is_bear):
            i+=1
            continue

        def dist(a,b): return abs(b-a)
        XA= dist(pxX, pxA)
        AB= dist(pxA, pxB)
        BC= dist(pxB, pxC)
        CD= dist(pxC, pxD)

        for pat in patterns:
            if pat not in harmonic_specs:
                continue
            spec= harmonic_specs[pat]
            rngAB_XA= spec["AB_XA"]
            rngBC_AB= spec["BC_AB"]
            rngCD_BC= spec["CD_BC"]

            AB_XA= AB/(XA+1e-9)
            BC_AB= BC/(AB+1e-9)
            CD_BC= CD/(BC+1e-9)

            okAB= ratio_in_range(AB_XA, rngAB_XA, fib_tolerance)
            okBC= ratio_in_range(BC_AB, rngBC_AB, fib_tolerance)
            okCD= ratio_in_range(CD_BC, rngCD_BC, fib_tolerance)

            if okAB and okBC and okCD:
                # volume check?
                if check_volume and vol_col in df.columns:
                    if idxD< len(df):
                        v_now= df[vol_col].iloc[idxD]
                        start_= max(0, idxD-20)
                        v_mean= df[vol_col].iloc[start_: idxD].mean()
                        if v_now> 1.2* v_mean:
                            pass
                found_any= True
        i+=1

    return found_any
