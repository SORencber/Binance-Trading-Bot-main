class MacroCollector:
    def evaluate_macro(self, row) -> float:
        """
        SP500, DXY, VIX verilerini baz alarak
        basit bir makro skor hesaplar.
        row içerisinde:
        - "SPX_Change" => S&P 500 günlük % değişim (ör. 1.2 => %1.2)
        - "DXY_Change" => DXY günlük % değişim
        - "VIX"        => VIX endeksi seviyesi
        """
        score = 0.0

        # ---- SP500 ----
        spx_chg = row.get("SPX_Change", 0)
        if spx_chg > 1.0:
            score += 1
        elif spx_chg < -1.0:
            score -= 1

        # ---- DXY ----
        dxy_chg = row.get("DXY_Change", 0)
        if dxy_chg > 0.3:
            score -= 1
        elif dxy_chg < -0.3:
            score += 1

        # ---- VIX ----
        vix_val = row.get("VIX", 20)
        if vix_val:
            if vix_val > 30:
                score -= 2
            elif vix_val > 20:
                score -= 1
            elif vix_val < 15:
                score += 1

        return score