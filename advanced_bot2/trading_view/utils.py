from trading_view.patterns.all_patterns import load_best_params_from_json,optimize_system_parameters,PivotScanner
def run_scanner(df, symbol, time_frame, system_params):
    """
    df:             Mevcut veri DataFrame
    symbol:         Örn. "BTCUSDT"
    time_frame:     Örn. "15m", "1h", vs.
    system_params:  Sizin ek config'leriniz (dict)
                    Örn: {
                        "pivot_left_bars": 5,
                        "pivot_right_bars":10,
                        "volume_filter": True,
                        "min_atr_factor": 0.1
                    }

    1) Dosyadan best_params yüklenir (load_best_params_from_json).
    2) Eğer best_params None dönerse:
       => optimize_parameters(...) çalışır, best_params bulunur,
          symbol.json'a kaydedilir.
    3) Okunan best_params ile system_params birleştirilir (örnek: max).
    4) PivotScanner ile pivotlar tespit edilir.
    """

    # 1) Dosyadan yükle
    best_params = load_best_params_from_json(symbol, time_frame)
    if best_params is None:
        print(f"[run_scanner] => No best_params found in JSON for {symbol}-{time_frame}.")
        print("[run_scanner] => Running optimize_parameters to create new best_params...")
        pivot_param_grid = {
        "left_bars": [5,10],
        "right_bars":[5,10],
        "volume_factor":[1.0,1.2],
        "atr_factor":[0.0,0.2]
    }
        # 2) optimize_parameters
        # Bu fonksiyon kendi içinde best_params'ı JSON'a kaydediyor.
        opt_result = optimize_system_parameters(df, symbol, time_frame)
        best_params = opt_result["best_params"]  # Sözlük: {"left_bars":..., "right_bars":..., "volume_factor":..., "atr_factor":...}
        print("best_params", symbol, best_params)
    # 3) best_params + system_params harmanlama
    #    Aşağıdaki mantık tamamen size kalmış, örnek:
    left_bars = best_params.get("left_bars", system_params["pivot_left_bars"])
    right_bars = best_params.get("right_bars", system_params["pivot_right_bars"])
    volume_factor = best_params.get("volume_factor", 0.0)
    atr_factor    = best_params.get("atr_factor", 0.0)

    if system_params.get("volume_filter", False):
        volume_factor = max(volume_factor, 1.2)

    atr_factor = max(atr_factor, system_params["min_atr_factor"])
    
    return atr_factor,volume_factor,left_bars,right_bars
    



