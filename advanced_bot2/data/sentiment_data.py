# data/sentiment_data.py
"""
Twitter / haber / reddit sentiment toplayıp "sentiment_score" üretme (placeholder).
Gerçekte requests + NLP + Auth vs.
"""



import datetime
import requests
from textblob import TextBlob

# global cache
news_cache = {
    "last_fetch": None,      # datetime objesi
    "last_value": 0.0
}

def fetch_news_headlines_cached(symbol="BTCUSDT", api_key="", interval_minutes=30):
    """
    1) Eğer en son fetch üzerinden 30 dakikadan az geçtiyse, cache'deki değeri döndürür.
    2) Aksi halde, gerçek 'fetch_news_headlines' fonksiyonunu çağırır,
       dönen sentiment'i cache'e kaydeder ve döndürür.
    """
    now = datetime.datetime.utcnow()
    if news_cache["last_fetch"] is not None:
        elapsed = (now - news_cache["last_fetch"]).total_seconds() / 60.0  # dakika cinsinden
        if elapsed < interval_minutes:
            print(f"[DEBUG] Using cached news sentiment (elapsed={elapsed:.1f} min).")
            return news_cache["last_value"]

    # 30 dk dolmuş veya ilk defa => gerçek API fonksiyonunu çağır
    val = fetch_news_headlines(symbol, api_key)
    news_cache["last_fetch"] = now
    news_cache["last_value"] = val
    return val

def fetch_news_headlines(symbol, api_key):
    """
    Orijinal fonksiyonunuz (NewsAPI -> sentiment).
    Burada rate limit'e yakalanmamak için
  
    """
    topic = "bitcoin"
    if symbol.startswith("ETH"):
        topic = "ethereum"
    elif symbol.startswith("BNB"):
        topic = "binance coin"

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": api_key
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "ok":
            error_code = data.get("code", "unknown_error")
            error_msg  = data.get("message", "")
            print(f"[NewsAPI Error] code={error_code}, msg={error_msg}")
            return 0.0

        articles = data.get("articles", [])
        if not articles:
            print("[INFO] No articles found for topic:", topic)
            return 0.0

        polarities = []
        for art in articles:
            title = art.get("title", "")
            blob = TextBlob(title)
            pol  = blob.sentiment.polarity
            polarities.append(pol)

        if len(polarities) == 0:
            return 0.0

        avg_sentiment = sum(polarities) / len(polarities)
        if avg_sentiment > 1.0:
            avg_sentiment = 1.0
        elif avg_sentiment < -1.0:
            avg_sentiment = -1.0

        return avg_sentiment

    except requests.RequestException as e:
        print(f"[ERROR fetch_news_headlines] RequestException => {e}")
        return 0.0
    except Exception as ex:
        print(f"[ERROR fetch_news_headlines] => {ex}")
        return 0.0

def fetch_twitter_sentiment(symbol="BTCUSDT", bearer_token=""):
    """
    Placeholder => normalde Twitter API v2 ile:
      1) requests.get("https://api.twitter.com/2/tweets/search/recent?...$BTC")
      2) NLP => ortalama pozitif/negatif skoru
    """
    if not bearer_token:
        return 0.0
    # pseudo
    # r= requests.get(..., headers={"Authorization":f"Bearer {bearer_token}"})
    # parse => sentiment average
    # return float 
    return 0.05  # yapay skor



def combine_sentiment_scores(*scores):
    """
    Basit ortalama / weighting
    """
    if not scores:
        return 0.0
    return sum(scores)/ len(scores)

def get_sentiment_for_symbol(symbol: str, config: dict)-> float:
    """
    Tek fonksiyon => Twitter/haber skoru => float
    """
    if not config.get("use_twitter"):
        return 0.0
    bearer= config.get("twitter_bearer_token","")
    tw_score= fetch_twitter_sentiment(symbol, bearer)
    news_score= fetch_news_headlines(symbol)
    combined= combine_sentiment_scores(tw_score, news_score)
    return combined
