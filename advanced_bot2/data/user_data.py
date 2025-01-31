# data/user_data.py

import asyncio
from binance import AsyncClient, BinanceSocketManager
from core.logging_setup import log
from core.context import SharedContext

async def user_data_reader(ctx: SharedContext):
    """
    user data => emir fill, cüzdan update
    """
    while True:
        try:
            #listen_key= await ctx.client_async.stream_get_listen_key()
            bsm2= BinanceSocketManager(ctx.client_async)
            #listen_key = await ctx.client_async.stream_get_listen_key()

            async with bsm2.user_socket() as s:
                while True:
                    event= await s.recv()
                    await ctx.user_data_queue.put(event)
        except Exception as e:
            log(f"[user_data_reader] Kopma => {e}\n Reconnect 5s", "error")
            await asyncio.sleep(5)

async def user_data_processor(ctx: SharedContext):
    while True:
        event= await ctx.user_data_queue.get()
        if not event:
            continue
        etype= event.get("e")
        if etype in ["executionReport","ORDER_TRADE_UPDATE"]:
            log(f"[UserData] => {event}", "info")
            # emir fill => PnL update, vs.
