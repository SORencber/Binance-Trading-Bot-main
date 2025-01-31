# data/data_fetching.py

import aiohttp
import asyncio
import pandas as pd

def handle_api_error(e, symbol, function_name):
    print(f"Hata ({function_name}) [{symbol}]: {e}")
    return None

BASE_URL = "https://fapi.binance.com"

# async def fetch_historical_open_interest(
#     session: aiohttp.ClientSession,
#     symbol: str,
#     period: str = "5m",
#     limit: int = 30,
#     start_time: int = None,
#     end_time: int = None
# ):
#     """
#     Binance Futures - Tarihsel Open Interest verisi çeker.
#     """
#     url = f"{BASE_URL}/futures/data/openInterestHist"
#     params = {
#         "symbol": symbol,
#         "period": period,
#         "limit": limit
#     }
#     if start_time is not None:
#         params["startTime"] = start_time
#     if end_time is not None:
#         params["endTime"] = end_time

#     try:
#         async with session.get(url, params=params, timeout=10) as response:
#             data = await response.json()
#             if isinstance(data, dict) and "code" in data:
#                 # Örnek hata: {"code": -1121, "msg": "Invalid symbol"}
#                 raise ValueError(f"API Error: {data}")
#             return data  # Normalde list tipinde döner
#     except Exception as e:
#         print(f"[ERROR fetch_historical_open_interest] => {e}")
#         return handle_api_error(e, symbol, "fetch_historical_open_interest")

async def fetch_funding_rate(session, symbol):
    url = f"{BASE_URL}/fapi/v1/premiumIndex"
    try:
        async with session.get(url, params={"symbol": symbol}, timeout=5) as response:
            data = await response.json()
            #print(data)
            return float(data.get("lastFundingRate", 0.0))
    except Exception as e:
        return handle_api_error(e, symbol,"fetch_funding_rate")

async def fetch_open_interest(session, symbol):
    url = f"{BASE_URL}/fapi/v1/openInterest"
    try:
        async with session.get(url, params={"symbol": symbol}, timeout=5) as response:
            data = await response.json()
           # print(data)
            return float(data.get("openInterest", 0.0))
    except Exception as e:
        return handle_api_error(e, symbol,"fetch_open_interest")

async def fetch_order_book(session, symbol, depth=50):
    url = f"https://api.binance.com/api/v3/depth?limit={depth}&symbol={symbol}"
    try:
        async with session.get(url, timeout=5) as response:
            data= await response.json()
            #print(data)
            return data
    except Exception as e:
        return handle_api_error(e, symbol,"fetch_order_book")

async def fetch_additional_data(symbol):
    """
    4 asenkron çağrı (funding, open interest, order book, historical OI)
    => tek seferde asyncio.gather(...) ile bekleyip
       4 sonucu da döndürür.
    """
    async with aiohttp.ClientSession() as session:
        try:
            funding_rate_task = fetch_funding_rate(session, symbol)
            open_interest_task = fetch_open_interest(session, symbol)
            order_book_task = fetch_order_book(session, symbol, depth=50)
            # open_interest_history_task = fetch_historical_open_interest(
            #     session,
            #     symbol,
            #     period,
            #     limit,
            #     start_time,
            #     end_time
            # )

            # BURADA => DÖRT FONKSİYONU AYNI ANDA ÇALIŞTIRIYORUZ
            funding_rate, open_interest, order_book = await asyncio.gather(
                funding_rate_task,
                open_interest_task,
                order_book_task,
                #open_interest_history_task
            )

            return funding_rate, open_interest, order_book #, open_interest_history
        except Exception as e:
            return handle_api_error(e, symbol, "fetch_additional_data")

BASE_URL = "https://fapi.binance.com"


async def fetch_historical_open_interest(
    session: aiohttp.ClientSession,
    symbol: str,
    period: str = "5m",
    limit: int = 500,
    start_time: int = None,
    end_time: int = None
):
    """
    Binance Futures - Tarihsel Open Interest verisi (max 500 satır).
    start_time / end_time => milis cinsinden
    """
    url = f"{BASE_URL}/futures/data/openInterestHist"
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit
    }
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    try:
        async with session.get(url, params=params, timeout=10) as response:
            data = await response.json()
            if isinstance(data, dict) and "code" in data:
                raise ValueError(f"API Error: {data}")
            return data if data else []
    except Exception as e:
        print(f"[ERROR fetch_historical_open_interest] => {e}")
        # handle_api_error -> Hata log vs.
        return []

async def fetch_oi_in_chunks(
    session: aiohttp.ClientSession,
    symbol: str,
    period: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    max_bars: int = 50000
):
    """
    [start_ts, end_ts] zaman aralığında, limit=500 kısıtını
    aşarak, geriye doğru chunk'lar halinde OI verisi toplar.
    
    - start_ts, end_ts => pd.Timestamp (UTC)
    - max_bars => en fazla kaç bar toplamak istediğinizi belirtir 
      (örneğin 50000).
    - Her istekte end_time parametresini geriye kaydırarak
      chunk_limit=500 veriyi alır.

    Dönüş: pd.DataFrame (timestamp ascending)
    """
    
  
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms   = int(end_ts.timestamp()   * 1000)

    all_data = []
    bars_collected = 0
    current_end = end_ms

    while True:
        if bars_collected >= max_bars:
            break

        chunk_limit = min(500, max_bars - bars_collected)

        data = await fetch_historical_open_interest(
            session=session,
            symbol=symbol,
            period=period,
            limit=chunk_limit,
            start_time=None,  # Sadece end_time bazlı geriye gidiyoruz
            end_time=current_end
        )
        if not data:
            break

        all_data.extend(data)
        got = len(data)
        bars_collected += got

        # En eski timestamp (milis)
        min_ts = min(d["timestamp"] for d in data)
        # Bir sonraki istekte buradan daha eskisini çekeceğiz
        next_end = min_ts - 1
        if next_end < start_ms:
            # Başlangıç zamanının da ötesine geçtik
            break

        current_end = next_end
        if got < chunk_limit:
            # chunk_limit=500 istediğimiz halde daha az geldiyse, veri bitti
            break

    if not all_data:
        return pd.DataFrame(columns=["timestamp","sumOpenInterest","sumOpenInterestValue"])

    df_oi = pd.DataFrame(all_data)
    df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"], unit="ms")
    df_oi["sumOpenInterest"] = pd.to_numeric(df_oi["sumOpenInterest"], errors="coerce")
    df_oi["sumOpenInterestValue"] = pd.to_numeric(df_oi["sumOpenInterestValue"], errors="coerce")

    # ascending sort
    df_oi.sort_values("timestamp", inplace=True)
    df_oi.reset_index(drop=True, inplace=True)

    # İlgili aralığın dışındaki satırları at
    df_oi = df_oi[(df_oi["timestamp"] >= start_ts) & (df_oi["timestamp"] <= end_ts)]
    df_oi.reset_index(drop=True, inplace=True)
    df_oi = pd.DataFrame(all_data)

    return df_oi

async def main():
    symbol = "BTCUSDT"
    period = "4h"
    limit = 500
    # Örnek: son 10 veri dilimini çekelim
    # startTime ve endTime'i None bırakırsak otomatik son verileri alır.

    async with aiohttp.ClientSession() as session:
        data = await fetch_historical_open_interest(
            session=session,
            symbol=symbol,
            period=period,
            limit=limit
        )
        print("Sonuç =>", data)

if __name__ == "__main__":
    asyncio.run(main())