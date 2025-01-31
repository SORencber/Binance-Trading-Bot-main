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

from inference.rl_agent import RLAgent
from exchange_clients.binance_spot_manager_async import BinanceSpotManagerAsync

TRADING_TASKS = {"tasks": []}


async def start_trading(ctx: SharedContext):
    tasks_list = TRADING_TASKS.get("tasks", [])
    # Eğer trading zaten başlatıldıysa, tekrarlamasını önlüyoruz
    if any(not t.done() for t in tasks_list):
        log("Trading tasks already running => skip start_trading", "info")
        return

    log("start_trading => begin", "info")

    # 1) Binance async client (websocket) => user data, price data
    ctx.client_async = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)
    ctx.bsm = BinanceSocketManager(ctx.client_async)

    # 2) Gerçek emirler için => BinanceSpotManagerAsync
    exchange_client = BinanceSpotManagerAsync(BINANCE_API_KEY, BINANCE_API_SECRET)

    # 2.1) USDT bakiyesini çekelim
    usdt_balance = await exchange_client.get_balance("USDT")
    log(f"Current USDT balance => {usdt_balance:.2f}", "info")

    # 2.2) RLAgent => offline q_table yükle
    agent = RLAgent(alpha=0.1, gamma=0.99, epsilon=0.0)
    # Örneğin q_table.json dosyası proje kökünde veya /models/ klasöründe
    try:
        agent.load_qtable("q_table.json")
        log("RLAgent Q-table loaded from q_table.json", "info")
    except Exception as e:
        log(f"Could not load q_table.json => {e}", "warning")

    # 3) StrategyManager => UltraPro, Behavior, Multi
    strategy_mgr = StrategyManager(
        ctx=ctx,
        exchange_client=exchange_client,
        initial_eq=usdt_balance,  # offline eq => gerçekte cüzdanda ne varsa
        max_risk=0.2,
        max_dd=0.15
    )
    # EĞER StrategyManager internal "ultra" stratejisinde RLAgent set edebiliyorsa:
    await strategy_mgr.initialize_strategies()
    # manager.ultra is UltraProStrategy => or any other
    if hasattr(strategy_mgr, "ultra") and strategy_mgr.ultra is not None:
        # RL agent ata
        strategy_mgr.ultra.rl_agent = agent
        log("RLAgent assigned to UltraProStrategy in StrategyManager", "info")
    else:
        log("No 'ultra' strategy found in StrategyManager or manager mode != 'ultra'", "warning")

    # 4) Data collector
    t_dc = asyncio.create_task(loop_data_collector(ctx, strategy_mgr), name="data_collector")

    # User data tasks (Binance orders/fills vs.)
    t_udr = asyncio.create_task(user_data_reader(ctx), name="user_data_reader")
    t_udp = asyncio.create_task(user_data_processor(ctx), name="user_data_processor")

    tasks_list = [t_dc, t_udr, t_udp]

    # 5) Price readers => manager.on_price_update
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
            # StrategyManager => on_price_update
            await strategy_mgr.on_price_update(symbol, price)
        except Exception as e:
            log(f"[realtime_price_processor({symbol})] => {e}\n{traceback.format_exc()}", "error")
            await asyncio.sleep(2)
    log(f"[realtime_price_processor({symbol})] stopped", "info")


from telegram_bot.telegram_app import TelegramBotApp

async def main():
    logger = configure_logger(BOT_CONFIG)
    log("main() => starting", "info")

    ctx = SharedContext(BOT_CONFIG)

    tg_app = None
    tasks = []

    if BOT_CONFIG.get("telegram_commands", False):
        tg_app = TelegramBotApp(BOT_CONFIG, ctx)
        t_tg = asyncio.create_task(tg_app.start_bot(), name="telegram_bot_app")
        tasks.append(t_tg)

    # Otomatik ticareti başlatmak isterseniz:
    await start_trading(ctx)

    # Bekleyen görevler => telegram, vb.
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
