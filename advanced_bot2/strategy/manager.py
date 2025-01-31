# strategy/manager.py

from strategy.tv_strategy import TradingViewStrategy
from core.logging_setup import log


class StrategyManager:
    def __init__(self, ctx, exchange_client, initial_eq=10000.0, max_risk=0.2, max_dd=0.15):

        """
        ctx: SharedContext
        exchange_client: opsiyonel => BinanceClient / OKXClient / ByBitClient vb.
        """
        self.ctx = ctx
        self.exchange_client = exchange_client
        self.initial_eq = initial_eq
        self.max_risk = max_risk
        self.max_dd = max_dd
       
        self.tv_strategy = TradingViewStrategy(ctx, exchange_client, max_risk, initial_eq, max_dd)

    async def analyze_data(self):
        mode = self.ctx.config.get("mode","ultra")  # default 'ultra'
        if mode == "trading_view":
            await self.tv_strategy.analyze_data(self.ctx)
       

    async def on_price_update(self, symbol: str, price: float):
        mode = self.ctx.config.get("mode","trading_view")
        
        if mode == "trading_view":
            await self.tv_strategy.on_price_update(self.ctx, symbol, price)
     
   
    async def initialize_strategies(self):
        mode = self.ctx.config.get("mode","trading_view")
        if mode == "trading_view":
            self.ultra = TradingViewStrategy(
                ctx=self.ctx,
                exchange_client=self.exchange_client,
                max_risk=self.max_risk,
                initial_eq=self.initial_eq,
                max_dd=self.max_dd
            )