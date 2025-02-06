# config/bot_config.py
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Artık .env içindeki değişkenleri os.getenv ile okuyabilirsiniz
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY","")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET","")
print("DEBUG KEY:", os.getenv("BINANCE_API_KEY"))

OKX_API_KEY        = os.getenv("OKX_API_KEY","")
OKX_API_SECRET     = os.getenv("OKX_API_SECRET","")
OKX_PASSPHRASE     = os.getenv("OKX_PASSPHRASE","")

BYBIT_API_KEY      = os.getenv("BYBIT_API_KEY","")
BYBIT_API_SECRET   = os.getenv("BYBIT_API_SECRET","")


BOT_CONFIG = {
    # Tek bir borsada çalışacaksanız:
    "active_exchange": "binance",
    "mode": "trading_view", 
    "paper_trading": True,
     # OCO ile Kademeli mi ? 
    "use_oco_kademeli" :False,
    "symbols": ["BTCUSDT"], # "" => advanced_strategy, "ml", "rl", ...
    # Birden fazla borsa için veriler
    "exchanges": {
        "binance": {
            "api_key": BINANCE_API_KEY,
            "api_secret": BINANCE_API_SECRET
        },
        "okx": {
            "api_key": OKX_API_KEY,
            "api_secret": OKX_API_SECRET,
            "passphrase": OKX_PASSPHRASE
        },
        "bybit": {
            "api_key": BYBIT_API_KEY,
            "api_secret": BYBIT_API_SECRET
        }
    },

  
    
    "trade_amount_usdt": 20.0,
    "min_balance_for_trading": 20.0,
    "slippage_rate": 0.001,
    "max_consecutive_losses": 3,
    "logging_level": "info",
    "max_backoff_sec": 60,
    # Kademeli
    "partial_tp_levels": [0.02,0.04],
    "partial_tp_ratio": 0.5,
   
    # Stop-Loss
    "stop_loss_pct": 0.03,   # -> entry*(1 - 0.03)

    # Trailing
    "trailing_pct": 0.03,    # -> e.g. %3 trailing
    # ML/RL paths
    "ml_model_path": "rf_behavior.pkl",
   # "rl_model_path": "model_data/ppo_behavior.zip",
    "xgboost_model_path": "models/xgboost_model.pkl",
    "lstm_model_path": "models/lstm_model.h5",
    "rl_model_path": "models/q_table.npy",
    # *Davranışsal veri parametreleri*
    "use_twitter": False,
    "use_onchain": True,
    "twitter_bearer_token": os.getenv("TWITTER_BEARER",""),
    "fear_greed_api_key": os.getenv("FEARGREED_KEY",""),
     # Logging
    "log_to_console": False,
    "log_to_file": True,
    "log_file_path": "bot.log",
    "log_max_bytes": 1_000_000,
    "log_backup_count": 2,

    # Ek bulut log parametreleri
    "aws_logging": False,
    "aws_endpoint": "...",
    "aws_token": "...",

    "gcp_logging": False,
    "gcp_endpoint": "...",
    "gcp_token": "...",

    "slack_logging": False,
    "slack_webhook_url": "...",

    "splunk_logging": False,
    "splunk_hec_endpoint": "...",
    "splunk_hec_token": "...",

  
    # Telegram commands
    "telegram_commands": True,
    "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN","7960510187:AAG2508opCLmJN0BlZAdT1FmL2Y7HxrK2i8"),  # Bot token
    "telegram_logging_chat_id": os.getenv("TELEGRAM_LOG_CHAT_ID","-1002493512673"),  
    "telegram_command_chat_id": os.getenv("TELEGRAM_COMMAND_CHAT_ID","-1002493512673"), 
    "allowed_user_ids": [799802592,1107664588],  # Telegram user id'ler
    "telegram_logging_chat_ids": [799802592,1107664588 ],



}
