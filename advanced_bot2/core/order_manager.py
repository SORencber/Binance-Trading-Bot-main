# core/order_manager.py

from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from binance.enums import SIDE_SELL
from core.logging_setup import log

class RealOrderManager:
    def __init__(self, client_async: AsyncClient, symbol: str):
        self.client= client_async
        self.symbol= symbol

    async def place_market_order(self, side:str, quantity: float):
        try:
            if side.upper()=="BUY":
                order= await self.client.order_market_buy(symbol=self.symbol, quantity=quantity)
            else:
                order= await self.client.order_market_sell(symbol=self.symbol, quantity=quantity)
            log(f"[RealOrder] Market {side} => {self.symbol}, qty={quantity}", "info")
            return order
        except BinanceAPIException as e:
            log(f"[RealOrder] Market {side} => {e}", "error")
            raise e

    async def place_oco_sell(self, quantity: float, tp_price: float, sl_price: float):
        try:
            oco= await self.client.create_oco_order(
                symbol=self.symbol,
                side= SIDE_SELL,
                quantity= float(quantity),
                price= str(tp_price),
                stopPrice= str(sl_price),
                stopLimitPrice= str(sl_price),
                stopLimitTimeInForce="GTC"
            )
            log(f"[RealOrder] OCO SELL => {self.symbol}, qty={quantity}, TP={tp_price}, SL={sl_price}", "info")
            return oco
        except BinanceAPIException as e:
            log(f"[RealOrder] OCO SELL => {e}", "error")
            raise e
