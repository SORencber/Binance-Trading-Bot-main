# strategy/advanced_behavior.py

import numpy as np
from binance.exceptions import BinanceAPIException
from strategy.base import IStrategy
from core.context import SharedContext
from core.logging_setup import log
from strategy.paper_order import PaperOrderManager
from core.order_manager import RealOrderManager
from core.utils import check_lot_and_notional

class BehaviorStrategy(IStrategy):
    def name(self) -> str:
        return "BehaviorStrategy"

    async def analyze_data(self, ctx: SharedContext):
        pass

    async def on_price_update(self, ctx: SharedContext, symbol: str, price: float):
        st= ctx.symbol_map[symbol]
        df= ctx.df_map.get(symbol,{}).get("15m", None)
        if df is None or len(df)<2:
            return
        row= df.iloc[-1]
        # ML model => X= [rsi, adx, sentiment, onchain]
        # RL => benzer
        rsi= row.get("rsi",50.0)
        adx= row.get("adx",20.0)
        sentiment= row.get("sentiment",0.0)
        onchain= row.get("onchain",0.0)

        try:
            if ctx.ml_model is not None:
                X= np.array([[rsi, adx, sentiment, onchain]])
                pred= ctx.ml_model.predict(X) # 0 => SELL, 1 => BUY
                if not st.has_position:
                    if pred[0] == 1:
                        await self.do_buy(ctx, symbol, price)
                else:
                    if pred[0] == 0:
                        await self.do_sell(ctx, symbol, price)
            else:
                # RL => ctx.rl_agent => obs => predict
                if ctx.rl_agent is None:
                    # RL agent yok => skip
                    return
                obs= np.array([rsi, adx, sentiment, onchain])
                action= ctx.rl_agent.predict_action(obs)  # 0 => sell, 1 => buy, 2 => hold
                if not st.has_position:
                    #print("hello")
                    if action == 1:
                        await self.do_buy(ctx, symbol, price)
                else:
                    if action == 0:
                        await self.do_sell(ctx, symbol, price)

        except Exception as e:
            log(f"[BehaviorStrategy on_price_update] Hata => {e}", "error")
            # Bu exception 'lokalde' yakalanacak, bot kapanmayacak.
            # Devam => pass
            pass

    async def do_buy(self, ctx: SharedContext, symbol: str, price: float):
        st= ctx.symbol_map[symbol]
        if st.has_position:
            return
        config= ctx.config

        async with ctx.lock:
            try:
                cost= min(config["trade_amount_usdt"], ctx.paper_positions["USDT"])
                eff= price*(1+ config["slippage_rate"])
                qty= cost/ eff

                if config["paper_trading"]:
                    pm= PaperOrderManager(symbol, ctx.paper_positions)
                    await pm.place_market_order("BUY", qty, eff)
                else:
                    # Gerçek emir => lot & notional
                    await check_lot_and_notional(ctx.client_async, symbol, qty, eff)
                    rm= RealOrderManager(ctx.client_async, symbol)
                    await rm.place_market_order("BUY", qty)

                st.has_position= True
                st.quantity= qty
                st.entry_price= eff
                log(f"[BehaviorStrategy] BUY => {symbol}, qty={qty:.6f}, px={eff:.2f}", "info")

            except ValueError as ve:
                # [LOT_SIZE] mod error, minNotional vs.
                log(f"[BehaviorStrategy do_buy] ValueError => {ve}", "warning")
                # Emir basarisiz => pass

            except BinanceAPIException as bae:
                # Binance API hatası
                log(f"[BehaviorStrategy do_buy] BinanceAPIException => {bae}", "error")

            except Exception as e:
                log(f"[BehaviorStrategy do_buy] Unexpected error => {e}", "error")

    async def do_sell(self, ctx: SharedContext, symbol: str, price: float):
        st= ctx.symbol_map[symbol]
        if not st.has_position:
            return
        config= ctx.config
        qty= st.quantity
        local_pnl= (price - st.entry_price)/ st.entry_price

        async with ctx.lock:
            try:
                if config["paper_trading"]:
                    pm= PaperOrderManager(symbol, ctx.paper_positions)
                    await pm.place_market_order("SELL", qty, price)
                else:
                    rm= RealOrderManager(ctx.client_async, symbol)
                    await rm.place_market_order("SELL", qty)

                st.pnl += local_pnl
                if local_pnl<0:
                    st.consecutive_losses += 1
                else:
                    st.consecutive_losses = 0
                log(f"[BehaviorStrategy] SELL => {symbol}, PnL={local_pnl:.2%}", "info")
                st.reset_position()

            except ValueError as ve:
                log(f"[BehaviorStrategy do_sell] ValueError => {ve}", "warning")

            except BinanceAPIException as bae:
                log(f"[BehaviorStrategy do_sell] BinanceAPIException => {bae}", "error")

            except Exception as e:
                log(f"[BehaviorStrategy do_sell] Unexpected error => {e}", "error")
