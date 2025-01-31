# data/onchain_data.py
"""
On-chain data + Fear&Greed, with 15-min caching
"""

import requests
import datetime

# Orijinal fonksiyonlar:
def fetch_fear_greed_index():
    try:
        url= "https://api.alternative.me/fng/"
        r= requests.get(url)
        data= r.json()
        val= float(data["data"][0]["value"])
        scaled= (val-50.0)/50.0  # => -1..+1
        return scaled
    except:
        return 0.0

def fetch_onchain_symbol(symbol="BTCUSDT"):
    netflow= -1000
    scaled= netflow/10000
    if scaled>1: scaled=1
    if scaled<-1: scaled=-1
    return scaled

# Cache
fgi_onchain_cache = {
    "last_fetch": None,
    "fgi_val": 0.0,
    "onchain_val": 0.0
}

async def fetch_fgi_and_onchain_15min(symbol="BTCUSDT"):
    global fgi_onchain_cache
    now = datetime.datetime.utcnow()
    
    if (fgi_onchain_cache["last_fetch"] is None
        or (now - fgi_onchain_cache["last_fetch"]).total_seconds() >= 15*60):
        
        print("[DEBUG] Real fetch => feargreed + onchain (15m).")
        fgi_val = fetch_fear_greed_index()
        onchain_val = fetch_onchain_symbol(symbol)
        
        fgi_onchain_cache["last_fetch"] = now
        fgi_onchain_cache["fgi_val"]    = fgi_val
        fgi_onchain_cache["onchain_val"]= onchain_val
    else:
        print("[DEBUG] Reuse cache => feargreed + onchain.")
        fgi_val = fgi_onchain_cache["fgi_val"]
        onchain_val = fgi_onchain_cache["onchain_val"]
    
    return (fgi_val, onchain_val)


def get_onchain_features(symbol: str, config: dict)-> float:
    """
    Tek seferde => fear_greed + netflow => combine
    """
    fear= fetch_fear_greed_index(config.get("fear_greed_api_key",""))
    flow= fetch_onchain_symbol(symbol)
    combined= (fear + flow)/ 2.0
    return combined
