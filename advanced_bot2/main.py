# main.py

import sys
import asyncio
import traceback
import os

from config.bot_config import BOT_CONFIG, BINANCE_API_KEY, BINANCE_API_SECRET
from core.context import SharedContext
from core.logging_setup import log, configure_logger

from binance import AsyncClient, BinanceSocketManager
from data.data_collector import loop_data_collector
from data.user_data import user_data_reader, user_data_processor

from strategy.manager import StrategyManager
from inference.ml_agent import MLAgent
from inference.xgboost_agent import XGBoostAgent
from inference.lstm_agent import LSTMAgent
from inference.rl_agent import RLAgent
from portfolio.portfolio_manager import PortfolioManager

import joblib

# Exchange client import
from exchange_clients.binance_spot_manager_async import BinanceSpotManagerAsync

TRADING_TASKS = {"tasks": []}


# ------------------------------
# 1) AGENT'LERİ YÜKLEME
# ------------------------------
# async def load_agents(ctx: SharedContext):
#     """
#     XGBoost, LSTM, RL vb. modelleri .model / .npy dosyalarından yükleyip
#     context'e (ctx) atar.
#     """
#     xgb_model_path = ctx.config.get("xgboost_model_path","")
#     if xgb_model_path and os.path.exists(xgb_model_path):
#         xgb = XGBoostAgent(xgb_model_path)
#         xgb.load_model()
#         ctx.xgboost_agent = xgb
#         log(f"[main] XGB model loaded => {xgb_model_path}", "info")
#     else:
#         log("[main] XGB model not found => skipping", "warning")
#         ctx.xgboost_agent = None

#     lstm_model_path = ctx.config.get("lstm_model_path","")
#     if lstm_model_path and os.path.exists(lstm_model_path):
#         lstm = LSTMAgent(lstm_model_path)
#         lstm.load_model()
#         ctx.lstm_agent = lstm
#         log(f"[main] LSTM model loaded => {lstm_model_path}", "info")
#     else:
#         log("[main] LSTM model not found => skipping", "warning")
#         ctx.lstm_agent = None

#     rl_model_path = ctx.config.get("rl_model_path","")
#     if rl_model_path and os.path.exists(rl_model_path):
#         rl = RLAgent(
#             # alpha=0.1, gamma=0.99, vb. parametreleriniz
#         )
#         rl.load_qtable(rl_model_path)  # <-- 'load_model' yerine 'load_qtable'
#         ctx.rl_agent = rl
#         rl.loaded = True
#         log(f"[main] RL Q-table loaded => {rl_model_path}", "info")
#     else:
#         rl = RLAgent()
#         rl.loaded = False
#         ctx.rl_agent = rl
#         log("[main] RL model not found => skipping", "warning")

#     ml_model_path = ctx.config.get("ml_model_path","")
#     if ml_model_path and os.path.exists(ml_model_path):
#         ctx.ml_model = joblib.load(ml_model_path)
#         log(f"[main] ML model loaded => {ml_model_path}", "info")
#     else:
#         ctx.ml_model = None
#         log("[main] ML model not found => skipping", "warning")


# ------------------------------
# 2) TRADING İŞLEMLERİNİ BAŞLATMA
# ------------------------------
async def start_trading(ctx: SharedContext):
    tasks_list = TRADING_TASKS.get("tasks", [])
    # Eğer trading zaten başlatıldıysa, tekrarlanmasın
    if any(not t.done() for t in tasks_list):
        log("Trading tasks already running => skip start_trading", "info")
        return

    log("start_trading => begin", "info")

    # 1) Binance async client (websocket) => user data, price data
    ctx.client_async = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET,requests_params={"timeout": 60})
    
    ctx.bsm = BinanceSocketManager(ctx.client_async)

    # 2) Gerçek emirler için => BinanceSpotManagerAsync
    exchange_client = BinanceSpotManagerAsync(BINANCE_API_KEY, BINANCE_API_SECRET)
    usdt_balance = await exchange_client.get_balance("USDT")

    # 3) StrategyManager => UltraPro, Behavior, Multi, vb.
    strategy_mgr = StrategyManager(
        ctx=ctx,
        exchange_client=exchange_client,
        initial_eq=usdt_balance,
        max_risk=0.4,
        max_dd=0.15
    )
    await strategy_mgr.initialize_strategies()

    # 4) Data collector
    t_dc = asyncio.create_task(loop_data_collector(ctx, strategy_mgr), name="data_collector")

    # User data tasks (Binance orders/fills, vs.)
    t_udr = asyncio.create_task(user_data_reader(ctx), name="user_data_reader")
    t_udp = asyncio.create_task(user_data_processor(ctx), name="user_data_processor")
    tasks_list = [t_dc, t_udr, t_udp]

    # 5) Price readers
    for sym in ctx.config["symbols"]:
        r = asyncio.create_task(realtime_price_reader(ctx, sym, strategy_mgr), name=f"reader_{sym}")
        p = asyncio.create_task(realtime_price_processor(ctx, sym, strategy_mgr), name=f"processor_{sym}")
        tasks_list.extend([r, p])

    TRADING_TASKS["tasks"] = tasks_list
    log("start_trading => started tasks", "info")


async def stop_trading(ctx: SharedContext):
    tasks_list = TRADING_TASKS.get("tasks", [])
    if not tasks_list:
        log("No trading tasks => skip stop_trading", "info")
        return

    log("stop_trading => canceling tasks", "info")
    for t in tasks_list:
        if not t.done():
            t.cancel()

    await asyncio.sleep(1)
    TRADING_TASKS["tasks"] = []
    log("stop_trading => canceled", "info")


# ------------------------------
# 3) WEBSOCKET'TEN GELEN FİYAT OKUMA
# ------------------------------
async def realtime_price_reader(ctx, symbol, strategy_mgr: StrategyManager):
    """
    WebSocket'ten sembolün fiyat bilgisini çekip queue'ya koyar.
    """
    while not getattr(ctx, "stop_requested", False):
        try:
            socket = ctx.bsm.symbol_ticker_socket(symbol)
            async with socket as s:
                while not getattr(ctx, "stop_requested", False):
                    msg = await s.recv()
                    if "c" in msg:
                        price = float(msg["c"])
                        if symbol not in ctx.price_queues:
                            ctx.price_queues[symbol] = asyncio.Queue()
                        await ctx.price_queues[symbol].put(price)
        except Exception as e:
            log(f"[realtime_price_reader({symbol})] => {e}\n{traceback.format_exc()}", "error")
            await asyncio.sleep(5)
    log(f"[realtime_price_reader({symbol})] stopped", "info")


async def realtime_price_processor(ctx, symbol, strategy_mgr: StrategyManager):
    """
    Kuyruktaki fiyatı alıp manager.on_price_update çağırır.
    """
    while not getattr(ctx, "stop_requested", False):
        if symbol not in ctx.price_queues:
            ctx.price_queues[symbol] = asyncio.Queue()
        try:
            price = await ctx.price_queues[symbol].get()
            await strategy_mgr.on_price_update(symbol, price)
            await asyncio.sleep(5)

        except Exception as e:
            log(f"[realtime_price_processor({symbol})] => {e}\n{traceback.format_exc()}", "error")
            await asyncio.sleep(2)
    log(f"[realtime_price_processor({symbol})] stopped", "info")


# ------------------------------
# 4) TELEGRAM BOT (OPSİYONEL)
# ------------------------------
from telegram_bot.telegram_app import TelegramBotApp


# ------------------------------
# 5) MAIN FONKSİYONU
# ------------------------------
async def main():
    # 1) Logger ayarları
    logger = configure_logger(BOT_CONFIG)
    log("main() => starting", "info")

    # 2) SharedContext oluştur
    ctx = SharedContext(BOT_CONFIG)

    # 3) RLAgent ve diğer modelleri yükle
    #await load_agents(ctx)

    # 4) Telegram bot (opsiyonel)
    tg_app = None
    tasks = []

    if BOT_CONFIG.get("telegram_commands", False):
        tg_app = TelegramBotApp(BOT_CONFIG, ctx)
        t_tg = asyncio.create_task(tg_app.start_bot(), name="telegram_bot_app")
        tasks.append(t_tg)

    # OTOMATİK TİCARETİ HEMEN BAŞLATMAK İSTİYORSANIZ (eski davranış):
    await start_trading(ctx)

    # 5) Bekleme döngüsü
    # Eğer Telegram komutlarıyla start_trading'i başlatacaksanız,
    # bu noktada sadece telegram bot beklemesi olabilir.
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for p in pending:
        p.cancel()
    for d in done:
        ex = d.exception()
        if ex:
            log(f"[main] => {ex}\n{traceback.format_exc()}", "error")

    log("main() ended", "info")
    sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log("Exit signal => main stopped", "warning")
    except Exception as e:
        log(f"Critical => {e}\n{traceback.format_exc()}", "error")
        sys.exit(1)
