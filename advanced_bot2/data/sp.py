import yfinance as yf
import datetime
import asyncio

# Basit bir global cache sözlüğü:
sp_cache = {
    "last_fetch": None,                   # datetime objesi
    "last_data": (None, None, None),      # (sp500_close, dxy_close, vix_close)
    "last_changes": (0.0, 0.0, 0.0)       # (sp500_change%, dxy_change%, vix_change%)
}

async def sp500_api_for_change():
    """
    YFinance ile S&P500 (^GSPC), DXY (DX=F), VIX (^VIX) verilerini
    son 2 KAPANIŞ arasında % değişim hesabı yapar.
    
    Dönüş formatı:
      ((sp500_close, sp500_chg_percent),
       (dxy_close,   dxy_chg_percent),
       (vix_close,   vix_chg_percent))
    """

    def get_close_and_prev(ticker_symbol, period="5d", interval="1d"):
        """
        Verilen sembolün 'period' ve 'interval' parametresiyle
        tarihsel verisini çeker, en az 2 satır kontrolü yapar,
        son iki kapanış arasındaki % değişimi döndürür.
        """
        # Veri çek
        data = yf.Ticker(ticker_symbol).history(period=period, interval=interval).dropna()
        # Yeterli satır var mı
        if len(data) < 2:
            raise IndexError(f"[{ticker_symbol}] Yetersiz veri (en az 2 satır gerekli). Gelen satır sayısı: {len(data)}")

        close_today = data["Close"].iloc[-1]
        close_prev = data["Close"].iloc[-2]
        chg_percent = (close_today - close_prev) / close_prev * 100

        return close_today, chg_percent

    # 1) S&P500 (^GSPC)
    sp500_close_today, sp500_chg_percent = get_close_and_prev("^GSPC")
    
    # 2) DXY (Dolar Endeksi Futures) => "DX=F"
    #   Eğer "DX=F" veri getirmekte zorlanırsa "DX-Y.NYB" deneyebilirsiniz.
    dxy_close_today, dxy_chg_percent = get_close_and_prev("DX=F")

    # 3) VIX (^VIX)
    vix_close_today, vix_chg_percent = get_close_and_prev("^VIX")

    return (
        (sp500_close_today, sp500_chg_percent),
        (dxy_close_today,   dxy_chg_percent),
        (vix_close_today,   vix_chg_percent),
    )

async def fetch_sp500_dxy_vix_15min():
    """
    15 dakikada bir yfinance API'sine bağlanıp:
      - sp500 kapanışı ve % değişim
      - dxy kapanışı ve % değişim
      - vix kapanışı ve % değişim
    
    Eğer 15 dakika dolmadıysa cache verisini döndürür.
    Yeni veri çekme aşamasında hata veya yetersiz veri oluşursa,
    eski cache devam eder.
    
    Dönüş:
    (sp500_val, sp500_chg, dxy_val, dxy_chg, vix_val, vix_chg)
    """
    global sp_cache
    now = datetime.datetime.utcnow()

    should_fetch = (
        sp_cache["last_fetch"] is None  # hiç çekilmemiş
        or (now - sp_cache["last_fetch"]).total_seconds() >= 15 * 60  # 15 dakika dolmuş
    )

    if should_fetch:
        print("[DEBUG] Real Yahoo Finance API call => 15 dakika doldu veya ilk kez.")
        try:
            (sp500_close, sp500_chg), (dxy_close, dxy_chg), (vix_close, vix_chg) = await sp500_api_for_change()

            # Yeni veri sorunsuz geldiyse cache'i güncelle
            sp_cache["last_fetch"]   = now
            sp_cache["last_data"]    = (sp500_close, dxy_close, vix_close)
            sp_cache["last_changes"] = (sp500_chg, dxy_chg, vix_chg)

        except IndexError as e:
            # Yeterli veri yoksa veya veriler eksikse
            print("[ERROR] Veri çekilirken yeterli satır gelmedi. Eski cache kullanılacak.")
            print(e)
            # Burada 'return' yapmıyoruz, eski cache verisi devam ediyor.

        except Exception as e:
            # Diğer olası hatalar
            print("[ERROR] API'den veri çekmede problem oldu. Eski cache kullanılacak.")
            print(e)
            # Yine 'return' yapmıyoruz, eski cache verisi geçerli.

    else:
        print("[DEBUG] Reuse cache data, not calling yfinance.")

    # Son olarak cache'ten döndürelim
    sp500_val, dxy_val, vix_val = sp_cache["last_data"]
    sp500_chg, dxy_chg, vix_chg = sp_cache["last_changes"]

    return (sp500_val, sp500_chg, dxy_val, dxy_chg, vix_val, vix_chg)

async def main():
    # Bir kere çağırıp test edelim
    val = await fetch_sp500_dxy_vix_15min()
    print("sp500/dxy/vix => ", val)

    # Tekrar çağırırsak 15 dk dolmadığı için cache'den gelecek
    val2 = await fetch_sp500_dxy_vix_15min()
    print("sp500/dxy/vix => ", val2)

if __name__ == "__main__":
    asyncio.run(main())
