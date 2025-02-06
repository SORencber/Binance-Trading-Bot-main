# strategy/tradingview_strategy.py

import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.context import SharedContext

from strategy.base import IStrategy
from core.logging_setup import log
from collectors.macro_collector import MacroCollector
from collectors.onchain_collector import OnChainCollector
from collectors.orderbook_analyzer import OrderbookAnalyzer
from collectors.one_minute_collector import OneMinVolCollector
from portfolio.portfolio_manager import PortfolioManager
from inference.rl_agent import RLAgent
from core.trade_logger import append_trade_record, get_last_net_pnl
import asyncio

from trading_view.main_tv import generate_signals # <-- Pattern tespiti fonksiyonlarınızı (detect_elliott vb.) içeren modül
# Yukarıda 'pattern_lib.generate_signals(df)' demek için, 
#   en son paylaştığınız "generate_signals" kodunu oraya koyabilirsiniz.
from datetime import datetime
import time
import traceback
LAST_SUMMARY_TIME = 0  # Son summary gönderim zamanı (timestamp)
SUMMARY_INTERVAL = 1800  # 30 dakika = 1800 sn

class TradingViewStrategy(IStrategy):
    """
    Bir pattern + sinyal stratejisi (Elliott, Wolfe vb.), 
    gerçek Binance emirleri için exchange_client (BinanceSpotManagerAsync) kullanır.
    """

    def name(self) -> str:
        return "TradingViewStrategy"

    def __init__(self, 
                 ctx, 
                 exchange_client, 
                 max_risk=0.5,
                 initial_eq=10000.0,
                 max_dd=0.15):
        """
        ctx: SharedContext
        exchange_client: BinanceSpotManagerAsync (gerçek emir)
        max_risk: 0.5 => risk parametresi, PortfolioManager 
        initial_eq: Paper varsayılan
        max_dd: max drawdown
        """
        super().__init__()
        self.ctx = ctx
        self.exchange_client = exchange_client
        self.pm = PortfolioManager(max_risk, initial_eq, max_dd)
        self.total_equity = initial_eq

        # Bazı collector örnekleri (demo, opsiyonel)
        self.macro = MacroCollector()
        self.onchain = OnChainCollector()
        self.orderbook = OrderbookAnalyzer()
        self.one_min_vol = OneMinVolCollector(5, 0.01)

        # RL agent (opsiyonel)
        self.rl_agent = RLAgent()

        # Strateji parametreleri
        self.panic_confirm_bars = 2
        self.max_reentry = 2
        self.reentry_window_bars = 10

    async def initialize_positions(self):
        """
        Spot cüzdanda halihazırda var olan coin miktarlarını 
        "pozisyon var" şeklinde işaretlemek isterseniz.
        """
        try:
            account_info = await self.exchange_client.get_account_info()
            if not account_info or "balances" not in account_info:
                log("Binance account info yok, 'balances' bulunamadı.", "error")
                return

            for symbol in self.ctx.config["symbols"]:
                st = self.ctx.symbol_map[symbol]
                if symbol.endswith("USDT"):
                    base_asset = symbol[:-4]
                else:
                    base_asset = symbol.replace("USDT","")

                # Balances'tan bul
                bal = next((b for b in account_info["balances"] if b["asset"]==base_asset), None)
                if not bal:
                    continue

                free_amt = float(bal["free"])
                locked_amt= float(bal["locked"])
                total_amt= free_amt + locked_amt
                if total_amt>0.0:
                    st.has_position = True
                    st.quantity = total_amt
                    st.entry_price = 0.0
                    st.highest_price= 0.0
                    last_pnl = await get_last_net_pnl(symbol)
                    st.net_pnl = last_pnl
                    log(f"[TVStrategy initpos] {symbol}, coin={total_amt} => mark position", "info")
                    self.pm.update_position(symbol, True, total_amt, 0.0)
                else:
                    log(f"[TVStrategy initpos] {symbol}, 0 coin => no pos", "info")

        except Exception as e:
            log(f"[TVStrategy initpos] => {e}\n{traceback.format_exc()}", "error")

    async def sync_account_equity(self):
        """
        Basit senaryoda, USDT bakiyesini okur. 
        Multi-coin derseniz coinleri de USDT cinsinden toplayabilirsiniz.
        """
        try:
            usdt_balance = await self.exchange_client.get_balance("USDT")
            self.total_equity = usdt_balance
            log(f"[TVStrategy] sync_account_equity => {usdt_balance:.2f}", "debug")
        except Exception as e:
            log(f"[TVStrategy] sync_account_equity => {e}\n{traceback.format_exc()}", "error")

    async def analyze_data(self, ctx):
        """
        eğer isterseniz, collector'lar vb. data incelersiniz.
        """
        pass

    async def on_price_update(self, ctx, symbol: str, price: float):
        
           # 3.1) USDT equity senkron
        await self.sync_account_equity()

        # 3.2) Drawdown kontrolü
        dd = self.pm.calc_drawdown(self.total_equity)
        if dd > self.pm.max_drawdown:
            log(f"[DDCheck] => drawdown={dd:.2%}>{self.pm.max_drawdown:.2%} => closeAll", "warning")
            await self.close_all_positions(reason="drawdownStop")
            return
        
        # 3.3) Karar mantığı (multi-timeframe + RL vs.)
        df_main = ctx.df_map.get(symbol, {}).get("merged", None)
        #print(df_main)
        if df_main is None or len(df_main) < 10:
            return  # Yeterli veri yok

        row_main = df_main.iloc[-1]
        st = ctx.symbol_map[symbol]

        # MTF & RL 
        regime = self.detect_regime(row_main, symbol)
        base_p = ctx.param_for_regime.get(regime, ctx.param_for_regime["DEFAULT"])
        obs = self.rl_agent.observe_environment({"symbol": symbol, "regime": regime})
        action = self.rl_agent.select_action(obs)
        merged_param = {
            "stop_atr_mult": action.get("stop_atr_mult", base_p["stop_atr_mult"]),
            "partial_levels": action.get("partial_levels", base_p["partial_levels"]),
            "partial_ratio": action.get("partial_ratio", base_p["partial_ratio"])
        }

        # Skorlamalar
        short_sco = self.eval_short_term(row_main, merged_param, symbol)
        med_sco = self.eval_medium_term(row_main, merged_param, symbol)
        long_sco = self.eval_long_term(row_main, merged_param, symbol)
        adv_s_s = self.eval_advanced_indicators_short(row_main, merged_param, symbol)
        adv_s_m = self.eval_advanced_indicators_medium(row_main, merged_param, symbol)
        adv_s_l = self.eval_advanced_indicators_long(row_main, merged_param, symbol)
        adv_score = adv_s_s + adv_s_m + adv_s_l
        macro_s = self.macro.evaluate_macro(row_main)
        vol_sco = self.one_min_vol.evaluate_1m_volatility(df_main)
        senti = self.sentiment_onchain(row_main)

        st_score = short_sco + med_sco + long_sco + adv_score + macro_s + vol_sco + senti
        
        total_s_s= short_sco +adv_s_s +macro_s 
        # Örnek: Short/Med/Long aynı yönde ise ek puan verelim/ceza verelim.
        synergy_bonus = 0
        sum_sml = short_sco + med_sco + long_sco
        if short_sco > 0 and med_sco > 0 and long_sco > 0:
            synergy_bonus += 1  # Hepsi pozitif => alım sinyalini güçlendir
        if short_sco < 0 and med_sco < 0 and long_sco < 0:
            synergy_bonus -= 1  # Hepsi negatif => satış sinyalini güçlendir

        st_score += synergy_bonus

        final_action = self.combine_scenario_with_ensemble(regime, st_score)      
        # panik / reentry
        if self.detect_panic_signal(row_main, symbol):
            st.panic_count += 1
        else:
            st.panic_count = 0
        st.panic_mode = (st.panic_count >= self.panic_confirm_bars)
      
        reentry_allow = False
        if st.last_sell_time:
            mins_since = (datetime.utcnow() - st.last_sell_time).total_seconds() / 60.0
            if mins_since < self.reentry_window_bars and st.reentry_count < self.max_reentry:
                reentry_allow = True

        ##hgrail_1 = row_main.get("HolyGrailSignal_5m", 0)
        hgrail_2 = row_main.get("HolyGrailSignal_15m", 0)
        hgrail_3 = row_main.get("HolyGrailSignal_30m", 0)
        hgrail_4 = row_main.get("HolyGrailSignal_1h", 0)
        hgrail_5 = row_main.get("HolyGrailSignal_4h", 0)
        hgrail_6 = row_main.get("HolyGrailSignal_1d", 0)
        print(hgrail_2,hgrail_3,hgrail_4,hgrail_5,hgrail_6)
        log_msg = (f"[{symbol}] => final_act={final_action},holygrail_=,total_score={st_score},  "
                    f"panic={st.panic_mode}, reentry={reentry_allow}, netPnL={st.net_pnl:.2f}, RL={action}")
        log(log_msg, "info")      
        #df_1m = ctx.df_map.get(symbol, {}).get("1m", None)
        #df_5m = ctx.df_map.get(symbol, {}).get("5m", None)
        #df_15m = ctx.df_map.get(symbol, {}).get("15m", None)
        df_30m = ctx.df_map.get(symbol, {}).get("30m", None)
        df_1h = ctx.df_map.get(symbol, {}).get("1h", None)
        df_4h = ctx.df_map.get(symbol, {}).get("4h", None)
        df_1d = ctx.df_map.get(symbol, {}).get("1d", None)
        if df_30m is None or df_1h is None or df_4h is None:
                return

      
        await self.send_telegram_messages(price=price,df_30m=df_30m,df_1h=df_1h,
                    df_4h=df_4h,
                    df_1d=df_1d,
                    ctx=ctx,
                    row_main=row_main,
                    symbol=symbol,
                    regime=regime,
                    force_summary=False
                )
          # sig_info_30m = generate_signals(
        #     df=df_30m, 
            
        #     time_frame="30m",         # 1h parametresi

        #        ml_model=None, 
        #     max_bars_ago=300,
        #       require_confirmed=True
        # )
        # pattern_score_30m= sig_info_30m["score"]
        # reason_30m = sig_info_30m["reason"]
        # pattern_details_30m=sig_info_30m["patterns"]
        
        # log(f"[TVStrategy {symbol}] => pattern_score_30m={pattern_score_30m}, reason={reason_30m},detail_for_1h: {pattern_details_30m}", "info")
        

        # sig_info_1h = generate_signals(
        #     df=df_1h, 
            
        #     time_frame="1h",         # 1h parametresi

        #        ml_model=None, 
        #     max_bars_ago=300,
        #       require_confirmed=True
        # )
        # pattern_score_1h= sig_info_1h["score"]
        # reason_1h  = sig_info_1h["reason"]
        # pattern_details_1h=sig_info_1h["patterns"]
        
        # log(f"[TVStrategy {symbol}] => pattern_score_1h={pattern_score_1h}, reason={reason_1h},detail_for_1h: {pattern_details_1h}", "info")
        
        # #df_4h = df_4h.iloc[-5000:]

        # sig_info_4h = generate_signals(
        #     df=df_4h, 
            
        #     time_frame="4h",         # 1h parametresi

        #     ml_model=None, 
        #     max_bars_ago=200,
        #       require_confirmed=True
        #      # veya kendi model nesneniz
        # )
        # pattern_score_4h= sig_info_4h["score"]
        # reason_4h  = sig_info_4h["reason"]
        # pattern_details_4h=sig_info_4h["patterns"]
        # log(f"[TVStrategy {symbol}] => pattern_score_4h={pattern_score_4h}, reason={reason_4h},detail_for_4h: {pattern_details_4h}", "info")

        # pattern_score_1h = sig_info_1h["score"]
        # pattern_score_4h = sig_info_4h["score"]

        # p_score_total = (1.0 * pattern_score_1h) + (1.5 * pattern_score_4h)
 # --- DEĞİŞİKLİK YAPILAN KISIMLAR (2): Eşikleri arttırarak sinyalleri güçlendir ---
        # final_action = 2  # 2 => HOLD
        # buy_threshold = 5   # önceki 4 yerine 5
        # sell_threshold = -5 # önceki -4 yerine -5
        
        #total_score=p_score_total + st_score 
       
        # reentry_allow = False
        # if st.last_sell_time:
        #     mins_since = (datetime.utcnow() - st.last_sell_time).total_seconds() / 60.0
        #     if mins_since < self.reentry_window_bars and st.reentry_count < self.max_reentry:
        #         reentry_allow = True 
       
   # 3.4) Pozisyon yok => BUY?
        # if total_score >= buy_threshold:
        #     final_action = "BUY"
        # elif total_score <= sell_threshold:
        #     final_action = "SELL"
        # else:
        #     final_action = "HOLD"

 
   
        if not st.has_position:
            if (not st.panic_mode) and (final_act ==  "BUY"):
                if retest_info is not None:
                # Bu, "kırılan neckline'a yakın" demek => alım
                    await self.do_buy(ctx, df_30m.iloc[-1], symbol)
                else:
                # retest yok => belki "daha agresif" trade
                # veya retest bekleme => bir opsiyon
                    pass
        else:
            # 3.5) Pozisyon var => SELL sinyali mi yoksa risk mgmt mi?
            if final_act == "SELL":
                await self.do_full_close(ctx, df_30m.iloc[-1], symbol)
            else:
                await self.handle_risk_management(ctx, df_30m, df_30m.iloc[-1], symbol, merged_param)



    def combine_scenario_with_ensemble(self, scenario, total_score) -> int:
        """
        Örnek:
          scenario="TREND_UP" => momentum => eger ensemble=-1 => belki hold? (zayıflat)
        """#        #
        print(scenario)
        print(total_score)

        if scenario=="TREND_UP":
            if total_score>5:
                return 1  # BUY
            else:
                # belki hold => 2
                return 2
        elif scenario=="TREND_DOWN":
            if total_score<-5:
                return 0  # SELL
            else:
                return 2
        elif scenario=="RANGE":
            # eger ensemble=+1 => Buy alt bant
            # eger ensemble=-1 => Sell üst bant
            return total_score if total_score!=-999 else 2
        elif scenario=="BREAKOUT_SOON":
            # eger ensemble=+1 => breakout buy
            return total_score
        else:
            # default
            return total_score
  
    # -------------------------------------------------
    # 4) Tüm pozisyonları kapatma
    # -------------------------------------------------
    async def close_all_positions(self, reason: str):
        for s in self.ctx.config["symbols"]:
            st = self.ctx.symbol_map[s]
            if st.has_position:
                df_main = self.ctx.df_map.get(s, {}).get("1m", None)
                if df_main is not None and len(df_main) > 0:
                    row_main = df_main.iloc[-1]
                    await self.do_full_close(self.ctx, row_main, s, reason=reason)

    # -------------------------------------------------
    # 5) BUY - FULL_CLOSE - PARTIAL_SELL
    # -------------------------------------------------
    async def do_buy(self, ctx: SharedContext, row_main, symbol):
        st = ctx.symbol_map[symbol]
        if st.has_position:
            return
        px = row_main["Close_30m"]
        log(f"[{symbol}] do_buy => px={px:.4f}", "info")
        # Miktar
        risk_usdt = 50.0
        raw_qty = risk_usdt / px

        # (Varsayım) stop => px * 0.97 => %3 alt
        # Bu sizin pattern dip altına dayandırabilirsiniz
        st.sl_price = px * 0.97
        st.tp1_price = px * 1.05  # basit
        st.tp2_price = px * 1.10  # 2. hedef

        #raw_qty = ctx.config.get("trade_amount_usdt", 20.0) / px
        st.has_position = True
        st.quantity = raw_qty
        st.entry_price = px
        st.highest_price = px

        # Portföy kaydı
        self.pm.update_position(symbol, True, raw_qty, px)

        # Emir
        try:
            if ctx.config.get("paper_trading", True):
                log(f"[{symbol}] Paper BUY => qty={raw_qty:.4f}", "info")
            else:
                order = await self.exchange_client.place_market_order(symbol, "BUY", raw_qty)
                if not order:
                    st.has_position = False
                    st.quantity = 0.0
                    st.entry_price = 0.0
                    self.pm.update_position(symbol, False, 0.0, 0.0)
        except Exception as e:
            log(f"[{symbol}] do_buy => {e}\n{traceback.format_exc()}", "error")
            st.has_position = False
            st.quantity = 0.0
            st.entry_price = 0.0
            self.pm.update_position(symbol, False, 0.0, 0.0)

    async def do_full_close(self, ctx: SharedContext, row_main, symbol, reason=""):
        st = ctx.symbol_map[symbol]
        if not st.has_position:
            return
        px = row_main["Close_1m"]
        entry = st.entry_price
        qty = st.quantity

        realized = (px - entry) * qty
        old_npl = st.net_pnl
        new_npl = old_npl + realized
        log(f"[{symbol}] => FULL CLOSE => px={px:.2f}, reason={reason}, "
            f"realizedPnL={realized:.2f}, oldNPL={old_npl:.2f}, newNPL={new_npl:.2f}", "info")

        try:
            if ctx.config.get("paper_trading", True):
                log(f"[{symbol}] Paper SELL => qty={qty:.4f}", "info")
            else:
                order = await self.exchange_client.place_market_order(symbol, "SELL", qty)
                if not order:
                    log(f"[{symbol}] do_full_close SELL fail => revert", "error")
                    return
        except Exception as e:
            log(f"[{symbol}] do_full_close => {e}\n{traceback.format_exc()}", "error")
            return

        st.net_pnl = new_npl
         # Kayıt ekle
        side = "SELL" if qty>0 else "BUY"
        await append_trade_record(
            symbol=symbol,
            side=side,
            qty=qty,
            price=px,
            realized_pnl=realized,
            net_pnl=new_npl
         )
        # RL => reward
        obs = self.rl_agent.observe_environment({"symbol": symbol, "close_reason": reason})
        next_obs = self.rl_agent.observe_environment({"symbol": symbol, "done": True})
        action = self.rl_agent.get_policy()
        self.rl_agent.update(obs, action, realized, next_obs)

        # State reset
        st.has_position = False
        st.quantity = 0.0
        st.entry_price = 0.0
        st.highest_price = 0.0
        st.last_sell_time = datetime.utcnow()
        st.reentry_count = 0

        # Portföy
        self.pm.update_position(symbol, False, 0.0, 0.0)

    async def do_partial_sell(self, ctx: SharedContext, symbol: str, qty: float, px: float):
        st = ctx.symbol_map[symbol]
        realizedPnL = (px - st.entry_price) * qty
        old_pnl = st.net_pnl
        new_pnl = old_pnl + realizedPnL

        log(f"[{symbol}] partial SELL => qty={qty:.4f}, px={px:.4f}, realizedPnL={realizedPnL:.4f}, "
            f"oldPNL={old_pnl:.4f}, newPNL={new_pnl:.4f}", "info")
        try:
            if ctx.config.get("paper_trading", True):
                log(f"[{symbol}] Paper partial SELL => {qty:.4f}", "info")
            else:
                order = await self.exchange_client.place_market_order(symbol, "SELL", qty)
                if not order:
                    log(f"[{symbol}] partial SELL fail => revert", "error")
                    return
        except Exception as e:
            log(f"[{symbol}] do_partial_sell => {e}\n{traceback.format_exc()}", "error")
            return

        st.net_pnl = new_pnl
        # (2) trade_logger
        await append_trade_record(
        symbol=symbol,
        side="SELL",
        qty=qty,
        price=px,
        realized_pnl=realizedPnL,
        net_pnl=new_pnl
         )
        # RL => partial
        obs = self.rl_agent.observe_environment({"symbol": symbol, "partial": True})
        next_obs = self.rl_agent.observe_environment({"symbol": symbol, "partial_done": True})
        action = self.rl_agent.get_policy()
        self.rl_agent.update(obs, action, realizedPnL, next_obs)

    # -------------------------------------------------
    # 6) Risk Yönetimi (Trailing / Partial TP)
    # -------------------------------------------------
    async def handle_risk_management(self, ctx: SharedContext, df_main, row_main, symbol: str, param: dict):
        """
        HOLD => partial / trailing stop
        param = {
        "stop_atr_mult": float,
        "partial_levels": [float, float, ...],
        "partial_ratio": float
        }
        """
        st= ctx.symbol_map[symbol]
        if not st.has_position:
            return

        px = row_main["Close_30m"]
        # 1) Highest price güncelle => trailing stop
        if px > st.highest_price:
            st.highest_price = px

        # 2) partial satış => RL param
        gain_ratio = (px - st.entry_price)/(st.entry_price+1e-9)
        for i, lvl in enumerate(param["partial_levels"]):
            # partial_done => SymbolState içinde bir dizi (True/False)
            if i>= len(st.partial_done):
                break
            if (not st.partial_done[i]) and gain_ratio> lvl:
                part_qty = st.quantity * param["partial_ratio"]
                if part_qty>0:
                    # do_partial_sell => Market SELL
                    await self.do_partial_sell(ctx, symbol, part_qty, px)
                    st.partial_done[i] = True
                    st.quantity -= part_qty
                    if st.quantity<=0:
                        break  # Tüm pozisyon bitti ise

        # 3) ATR Tabanlı trailing stop => RL param
        #    Basit => ref_px * 0.02 * param["stop_atr_mult"]
        trailing_stop = st.highest_price - self.calc_atr_stop_dist(st.highest_price, param["stop_atr_mult"])
        if px < trailing_stop:
            log(f"[{symbol}] => trailingStop => closeAll", "warning")
            await self.do_full_close(ctx, row_main, symbol, reason="trailingStop")

    # -------------------------------------------------
    # 7) Bazı skor / sinyal fonksiyonları
    # -------------------------------------------------
    def sentiment_onchain(self, row_main) -> int:
        cnt = 0
        fgi = row_main.get("Fear_Greed_Index", 0.5)
        news = row_main.get("News_Headlines", 0.0)
        funding = row_main.get("Funding_Rate", 0.0)
        ob = row_main.get("Order_Book_Num", 0.0)
        mvrv = row_main.get("MVRV_Z_1d", 0)

        if mvrv:
            if mvrv < -0.5:
                cnt +=2
            elif mvrv>0.7:
                cnt -=2
            
        if funding:
            if funding > 0.01:
                cnt += 1
            else:
                cnt -=1
        if fgi:
            if fgi < 0.3:
                cnt += 1
            elif fgi > 0.7:
                cnt -= 1
        if news:
            if news < -0.2:
                cnt -= 1
            elif news > 0.2:
                cnt += 1
        if ob:
            if ob < 0:
                cnt -= 1
            elif ob > 0:
                cnt += 1
        return cnt

    def detect_panic_signal(self, row_main, symbol) -> bool:
        """
        Daha gerçekçi bir 'panik' sinyali için basit yaklaşım:
        - FGI < 0.3  => Aşırı korku
        - News < -0.2 => Negatif haber akışı
        - Funding < 0 => Düşüş beklentisi ağır basıyor
        - OrderBook < 0 => Satış baskısı
        
        2 veya daha fazla koşul karşılanırsa 'True' (panik).
        """
        panic_count = 0

        # Metrikleri al
        fgi = row_main.get("Fear_Greed_Index", 0.5)   # Varsayılan 0.5 (nötr)
        news = row_main.get("News_Headlines", 0.0)   # Varsayılan 0.0 (nötr)
        funding = row_main.get("Funding_Rate", 0.0)  # Varsayılan 0.0
        orderbook_val = row_main.get("Order_Book_Num", 0.0)  # Varsayılan 0.0
        
        # FGI < 0.3 => Aşırı korku
        if fgi < 0.3:
            panic_count += 1

        # Negatif haber akışı
        if news < -0.2:
            panic_count += 1

        # Funding negatif => shortlar ödüyor, piyasa düşüş beklentili
        if funding:
            if  funding < 0:
                panic_count += 1

        # Order Book negatif => net satış baskısı
        if orderbook_val < 0:
            panic_count += 1

        # Eşik: 2 veya üzeri 'panik' sinyali olarak kabul
        return (panic_count >= 2)
    def detect_regime(self, row_main, symbol) -> str:
        """
        1 saatlik (1h) veriler üzerinden trend, range veya yaklaşan breakout senaryolarını döndüren fonksiyon.
        Daha büyük zaman dilimlerinden (4h, 1D) de teyit alır.
        """

        # =============================
        # 1) GÖSTERGELERİ AL
        # =============================
        # --- 1H ---
        adx_1h        = row_main.get("ADX_1h", 20.0)
        rsi_1h        = row_main.get("RSI_1h", 50.0)
        macd_1h       = row_main.get("MACD_1h", 0.0)
        macd_signal_1h= row_main.get("MACD_Signal_1h", 0.0)
        volume_1h     = row_main.get("Volume_1h", 0.0)

        # Bollinger
        bb_up_1h      = row_main.get("BBUp_1h", 999999)
        bb_low_1h     = row_main.get("BBLow_1h", 0)
        band_width    = bb_up_1h - bb_low_1h
        mid_price     = (bb_up_1h + bb_low_1h) / 2 if (bb_up_1h + bb_low_1h) != 0 else 1
        bb_ratio      = band_width / mid_price  # daralma için oran

        # --- Daha büyük TF (4h, 1d) teyit ---
        adx_4h = row_main.get("ADX_4h", 20.0)
        rsi_4h = row_main.get("RSI_4h", 50.0)
        adx_1d = row_main.get("ADX_1d", 20.0)
        rsi_1d = row_main.get("OI_RSI_1d", 50.0)  # Burada 'OI_RSI_1d' kolonu olduğu varsayılmış

        # Ek metrikler
        funding   = row_main.get("Funding_Rate", 0.0)
        ob        = row_main.get("Order_Book_Num", 0.0)
        oi_1h     = row_main.get("Open_Interest_1h", 0.0)
        cvd_1h    = row_main.get("CVD_1h", 0.0)

        # =============================
        # 2) EŞİK DEĞERLER
        # =============================
        ADX_STRONG       = 25
        ADX_MEDIUM       = 20
        ADX_VERYLOW      = 15

        RSI_UPPER_MED    = 60
        RSI_LOWER_MED    = 40

        MACD_POS_THRESH  = 0.0
        MACD_NEG_THRESH  = 0.0

        VOLUME_THRESHOLD = 100_000

        BB_SQUEEZE       = 0.05

        FUNDING_POS_THRESH = 0.01
        FUNDING_NEG_THRESH = -0.01
        OB_POS_THRESH      = 0.0
        OB_NEG_THRESH      = 0.0

        OI_THRESHOLD    = 100_000
        CVD_POS_THRESH  = 0
        CVD_NEG_THRESH  = 0

        # =============================
        # 3) HIZLI YARDIMCI İFADELER
        # =============================
        macd_positive = (macd_1h > MACD_POS_THRESH and macd_1h > macd_signal_1h)
        macd_negative = (macd_1h < MACD_NEG_THRESH and macd_1h < macd_signal_1h)

        bullish_4h = (adx_4h > ADX_MEDIUM and rsi_4h > 55)
        bullish_1d = (adx_1d > ADX_MEDIUM and rsi_1d > 55)
        bearish_4h = (adx_4h > ADX_MEDIUM and rsi_4h < 45)
        bearish_1d = (adx_1d > ADX_MEDIUM and rsi_1d < 45)

        # =============================
        # 4) STRONG UP / STRONG DOWN
        # =============================
        # STRONG UP TRENDi
        if (adx_1h > ADX_STRONG and
            rsi_1h > RSI_UPPER_MED and
            macd_positive and
            volume_1h > VOLUME_THRESHOLD):
            
            # OI + CVD teyidi
            if (oi_1h > OI_THRESHOLD and cvd_1h > CVD_POS_THRESH):
                if bullish_4h or bullish_1d:
                    return "STRONG_UP_TREND"
                else:
                    return "UP_TREND"
            else:
                return "UP_TREND"

        # STRONG DOWN TRENDi
        if (adx_1h > ADX_STRONG and
            rsi_1h < RSI_LOWER_MED and
            macd_negative and
            volume_1h > VOLUME_THRESHOLD):
            
            if (oi_1h > OI_THRESHOLD and cvd_1h < CVD_NEG_THRESH):
                if bearish_4h or bearish_1d:
                    return "STRONG_DOWN_TREND"
                else:
                    return "DOWN_TREND"
            else:
                return "DOWN_TREND"

        # =============================
        # 5) UP / DOWN (NORMAL)
        # =============================
        # Orta seviye trend
        if (adx_1h >= ADX_MEDIUM and rsi_1h > 55 and macd_positive):
            return "UP_TREND"
        if (adx_1h >= ADX_MEDIUM and rsi_1h < 45 and macd_negative):
            return "DOWN_TREND"

        # =============================
        # 6) BOLLINGER KISITLI / BREAKOUT SOON
        # =============================
        # Bollinger dar -> (bb_ratio < 0.05), ADX düşük -> (adx_1h < 20)
        # Funding/OB/OI/CVD'ye bakarak yön tahmini
        if (adx_1h < ADX_MEDIUM) and (bb_ratio < BB_SQUEEZE):
            if(funding) and ob and oi_1h :
                bullish_break = (
                    funding > FUNDING_POS_THRESH and
                    ob      > OB_POS_THRESH      and
                    oi_1h   > OI_THRESHOLD       and
                    cvd_1h  > CVD_POS_THRESH
                )
                bearish_break = (
                    funding < FUNDING_NEG_THRESH and
                    ob      < OB_NEG_THRESH      and
                    oi_1h   > OI_THRESHOLD       and
                    cvd_1h  < CVD_NEG_THRESH
                )

                if bullish_break:
                    return "BREAKOUT_SOON_UP"
                elif bearish_break:
                    return "BREAKOUT_SOON_DOWN"
                else:
                    return "BREAKOUT_SOON_UNKNOWN"
            

        # =============================
        # 7) RANGE (YATAY)
        # =============================
        if adx_1h < ADX_VERYLOW:
            return "RANGE"

        # =============================
        # 8) SON ÇARE
        # =============================
        return "WEAK_TREND"

    def detect_regime_15m(row_main) -> str:
        """
        15 dakikalık zaman dilimi (15m) için basit trend / range tespiti.
        Büyük zaman dilimi (1h) trendini de hafifçe göz önüne alır.
        """

        # 15m metrikler
        adx_15m = row_main.get("ADX_15m", 20.0)
        rsi_15m = row_main.get("RSI_15m", 50.0)

        # Daha büyük bir timeframe'den (örn. 1h) trend teyidi almak için
        adx_1h = row_main.get("ADX_1h", 20.0)
        rsi_1h = row_main.get("RSI_1h", 50.0)

        # Eşik değerleri (temsili)
        ADX_THRESHOLD = 25
        RSI_UPPER = 60
        RSI_LOWER = 40

        # 15m'de ana trend gücünü ölç
        if adx_15m > ADX_THRESHOLD:
            # Trend güçlü ise RSI'a bakarak yön belirle
            if rsi_15m > RSI_UPPER:
                # 1h trendine de bakıyoruz; eğer 1h da yüksekse "TREND_UP" daha güvenilir
                if rsi_1h > 55 and adx_1h > 20:
                    return "TREND_UP_STRONG"
                else:
                    return "TREND_UP"
            elif rsi_15m < RSI_LOWER:
                # 1h trendi de düşük mü?
                if rsi_1h < 45 and adx_1h > 20:
                    return "TREND_DOWN_STRONG"
                else:
                    return "TREND_DOWN"
            else:
                return "TREND_WEAK"  # ADX yüksek ama RSI nötr -> zayıf trend
        else:
            # Trend güçlü değil -> Range ya da sıkışma
            return "RANGE_OR_SIDEWAYS"


    def eval_short_term(self, row_s, param, symbol) -> float:
        rsi_5m = row_s.get("OI_RSI_5m", 50.0)
        stoch_5m = row_s.get("StochK_5m", 50.0)
        sc = 0
        if rsi_5m > 60 and stoch_5m > 80:
            sc += 1
        elif rsi_5m < 40 and stoch_5m < 20:
            sc -= 1
        return sc

    def eval_medium_term(self, row_m, param, symbol) -> float:
        st_dir = row_m.get("SuperTD_1h", 0)
        ichiA = row_m.get("Ichi_SpanA_1h", 0)
        c_m = row_m.get("Close_1h", 0)
        sc = st_dir
         # Candle Engulf => +1/-1
        cdl_ = row_m.get(f"CDL_ENGULFING_1h", 0)
        if cdl_>0:
            sc+=1
        if c_m > ichiA:
            sc += 1
        return sc

    def eval_long_term(self, row_l, param, symbol) -> float:
        t3_4h = row_l.get("T3_4h", None)
        macd_4h = row_l.get("MACD_4h", 0.0)
        c_l = row_l.get("Close_4h", 0)
        sc = 0
        if t3_4h and c_l > t3_4h:
            sc += 1
        if macd_4h > 0:
            sc += 1
        return sc

    def eval_advanced_indicators_short(self, row_s, param, symbol) -> float:
        sc = 0
        pivot_5m = row_s.get("Pivot_5m", None)
        c_s = row_s.get("Close_5m", 0)
        if pivot_5m and c_s > pivot_5m:
            sc += 0.5
        cdl_e = row_s.get("CDL_ENGULFING_5m", 0)
        if cdl_e > 0:
            sc += 0.5
        return sc

    def eval_advanced_indicators_medium(self, row_m, param, symbol) -> float:
        sc = 0
        bbmid = row_m.get("BBMid_1h", None)
        c_m = row_m.get("Close_1h", 0)
        if bbmid and c_m > bbmid:
            sc += 0.5
        stoch_rsi = row_m.get("StochRSI_1h", 0.5)
        if stoch_rsi > 0.8:
            sc += 0.5
        return sc

    def eval_advanced_indicators_long(self, row_l, param, symbol) -> float:
        sc = 0
        fib618 = row_l.get("Fibo_61.8_4h", None)
        c_l = row_l.get("Close_4h", 0)
        if fib618 and c_l > fib618:
            sc += 0.5
        mvrv = row_l.get("MVRV_Z_4h", 0.0)
        if mvrv < -0.5:
            sc += 0.5
        return sc

    def calc_atr_stop_dist(self, ref_px, mult):
        ##mvrv = ctx.get("MVRV_Z_4h", 0.0)
        base = ref_px * 0.02
        return base * mult


    async def decide_trade_mtf_with_pattern(self,df_30m, df_1h, df_4h,df_1d, 
                     symbol, 
                     min_score=5,
                     retest_tolerance=0.05,
                     ctx=None):
        """
        Çoklu TF sinyalleri toplayarak final kararı veren ÖRNEK fonksiyon.
        - df_30m, df_1h, df_4h => bu TF'lerin DataFrame'leri
        - min_score => MTF sinyallerin toplamında en az bu puan olmalı
        - retest_tolerance => (ör. %0.5) retest mesafesi
        - ctx => strateji context veya logger, vs.

        Dönüş:
        {
            "final_decision": "BUY"/"SELL"/"HOLD",
            "retest_info": {...} veya None,
            "score_30m": ...,
            "score_1h": ...,
            "score_4h": ...,
            "patterns_30m": {...},
            "patterns_1h": {...},
            "patterns_4h": {...},
            ...
        }
        """

        # 1) MTF Sinyal Çağrıları
       # sig_15m = await generate_signals(df_15m, symbol,time_frame="15m",ml_model=None,max_bars_ago=100, retest_tolerance=0.005, require_confirmed=True)
        sig_30m =  await generate_signals(df_30m, symbol,time_frame="30m",ml_model=None,max_bars_ago=150,retest_tolerance=0.02,  require_confirmed=True)
        sig_1h  =  await generate_signals(df_1h, symbol, time_frame="1h", ml_model=None,max_bars_ago=150,retest_tolerance=0.05,  require_confirmed=True)
        sig_4h  =  await generate_signals(df_4h, symbol, time_frame="4h",ml_model=None ,max_bars_ago=300, retest_tolerance=0.05, require_confirmed=True)
        sig_1d  =  await generate_signals(df_1d, symbol, time_frame="1d",ml_model=None ,max_bars_ago=300, retest_tolerance=0.05, require_confirmed=True)

        #print(sig_30m)
       # score_15m = sig_15m["score"]
        score_30m = sig_30m["score"]
        score_1h  = sig_1h["score"]
        score_4h  = sig_4h["score"]
        score_1d = sig_1d["score"]

        #print("15m results:",sig_15m["pattern_trade_levels"])
        #print("30m results:",sig_30m["pattern_trade_levels"])
        #print("1h results:",sig_1h["pattern_trade_levels"])
       # print("4h results:",sig_4h["pattern_trade_levels"])
        #print("1d results:",sig_1d["pattern_trade_levels"])


        # MTF kombine skor (basit örnek => 30m + 1h + 4h)
        # Dilerseniz 30m'ye 1x, 1h'ye 1.5x, 4h'ye 2x ağırlık verebilirsiniz.
        combined_score =  score_30m + score_1h*2 + score_4h*3

        # Basit: eğer combined_score >= min_score => BUY, <= -min_score => SELL, else HOLD
        final_decision = "HOLD"
        if combined_score >= min_score:
            final_decision = "BUY"
        elif combined_score <= -min_score:
            final_decision = "SELL"

        
        retest_data = None
    
       
        
        return {
            "final_decision": final_decision,
           # "score_15m": score_15m,

            "score_30m": score_30m,
            "score_1h":  score_1h,
            "score_4h":  score_4h,
            "score_1d": score_1d,
            
            #"patterns_15m": sig_15m["pattern_trade_levels"],

            "patterns_30m": sig_30m["pattern_trade_levels"],
            "patterns_1h":  sig_1h["pattern_trade_levels"],
            "patterns_4h":  sig_4h["pattern_trade_levels"],
            "patterns_1d":  sig_1d["pattern_trade_levels"],

            "retest_info": retest_data,
            "combined_score": combined_score
        }

    async def format_pattern_results(self, row_main,mtf_dict: dict) -> str:
        """
        mtf_dict şu formatta bir sözlüktür:
        {
        "15m": {
            "double_top": [ {...}, {...} ],
            "triple_top_advanced": [ {...}, ...],
            ...
        },
        "30m": { ... },
        "1h": { ... },
        "4h": { ... },
        "1d": { ... }
        }

        Her pattern listesi, 'detect_all_patterns_v2' veya 'generate_signals' çıktısındaki
        'patterns' dict'ine benzer:
        [
            {
            "entry_price": 3053.51,
            "stop_loss": 3215.23,
            "take_profit": 2954.83,
            "direction": "SHORT",
            "pattern_raw": {
                "confirmed": True,
                "pattern": "double_top",
                ...
            }
            },
            ...
        ]

        Döndürdüğümüz metin => multiline string (Telegram'a gönderilecek).
        """
        lines = []
      
      
        # Sadece bu TF'leriniz varsa sabit olarak tanımlayabilirsiniz.
        # Yoksa sorted(mtf_dict.keys()) diyerek de sıralayabilirsiniz.
        timeframes = ["30m","1h","4h","1d"]
        
        for tf in timeframes:
            # Her timeframe dictionary'sini al
            tf_data = mtf_dict.get(tf, None)
            if not tf_data:
                # Pattern listesi yoksa / skip
                lines.append(f"\n--- {tf} => No pattern data ---")
                continue

            lines.append(f"\n--- {tf} Pattern Results ---")

            # tf_data: ör. { "double_top": [...], "double_bottom": [...], ... }
            # Her pattern ismi ve listesini dolaşalım:
            for pattern_name, p_list in tf_data.items():
                if not p_list:
                    # Boş liste => bu pattern bulunmamış
                    continue

                # Kaç adet pattern bulundu
                lines.append(f"* {pattern_name} => {len(p_list)} adet")

                # Tek tek parse
                for idx, pat in enumerate(p_list, start=1):
                    ep   = pat.get("entry_price")
                    sl   = pat.get("stop_loss")
                    tp   = pat.get("take_profit")
                    dire = pat.get("direction", "N/A")

                    # pattern_raw içinden ek bilgi istersek:
                    raw  = pat.get("pattern_raw", {})
                    conf = raw.get("confirmed", False)
                    patn = raw.get("pattern", "?")

                    # Format -> 2 decimal
                    ep_s  = f"{ep:.2f}" if ep else "N/A"
                    sl_s  = f"{sl:.2f}" if sl else "N/A"
                    tp_s  = f"{tp:.2f}" if tp else "N/A"

                    lines.append(
                        f"   [{idx}] Dir={dire}, Entry={ep_s}, Stop={sl_s}, TP={tp_s}, "
                        f"Confirmed={conf}, name={patn}"
                    )
          
        final_text = "\n".join(lines)
        if not final_text.strip():
            final_text = "No pattern results found."
        return final_text
   
    # -------------------------------------------------------
    # 1) Pattern Puanlamasını Yapan Fonksiyon
    # -------------------------------------------------------
    def find_close_patterns(self,results_dict: dict, current_price, lower_threshold=5, upper_threshold=10):
        """
        results_dict   : (dict)  Pattern verilerinin tutulduğu sözlük
        current_price  : (float) Anlık gelen coin fiyatı
        lower_threshold: (float) Yüzde olarak alt eşik (default=5)
        upper_threshold: (float) Yüzde olarak üst eşik (default=10)
        
        TP değeri ile current_price arasındaki yüzdesel farkın 
        lower_threshold <= fark <= upper_threshold olduğu pattern’leri
        puanlayarak döndürür.
        """
        close_patterns = []

        for timeframe, pattern_types in results_dict.items():
            for pattern_type, pattern_list in pattern_types.items():
                for pattern in pattern_list:
                    direction = pattern.get("direction")
                    tp = pattern.get("take_profit")

                    if tp is not None:
                        fark_yuzdesi = abs(tp - current_price) / current_price * 100
                        if lower_threshold <= fark_yuzdesi <= upper_threshold:
                            # Örnek puan hesabı (dilediğiniz gibi değiştirebilirsiniz)
                            puan = round(upper_threshold - fark_yuzdesi, 2)
                            
                            close_patterns.append({
                                "timeframe": timeframe,
                                "pattern_type": pattern_type,
                                "direction": direction,
                                "take_profit": tp,
                                "current_price": current_price,
                                "fark_yuzdesi": round(fark_yuzdesi, 2),
                                "puan": puan
                            })
        return close_patterns


    # -------------------------------------------------------
    # 2) Telegram Mesajlarını Gönderen Fonksiyon
    #    -> Her 30 dakikada bilgi mesajı (summary)
    #    -> Pattern skoru varsa alert mesajı
    # -------------------------------------------------------
    #  30 dakikada bir "bilgi mesajı" atıp atmadığımızı kontrol için
    #  global ya da class seviyesinde sakladığımız bir değişken kullanıyoruz:

    async def send_telegram_messages(self,price,
                                     df_30m,df_1h,
                    df_4h,
                    df_1d, ctx: SharedContext, row_main, symbol, regime, force_summary=False):
        """
        force_summary=True  -> Bilgi mesajını her halükarda gönder.
        force_summary=False -> Son gönderimden bu yana 30 dk geçtiyse gönder, yoksa gönderme.
        
        1) Bilgi (Özet) Mesajı (Summary):
        - Korku Endeksi, Fonlama Oranı, vb. gibi genel bilgileri içerir.
        - 30 dakikada bir veya force_summary=True ise gönderilir.
        2) Pattern Alert Mesajı:
        - 'find_close_patterns' ile elde edilen listede bir şey varsa gönderilir.
        - Yoksa alert mesajı gönderilmez.
        """
        global LAST_SUMMARY_TIME
        results_dict={}
        #print(LAST_SUMMARY_TIME)
        # Telegram objesini al
        telegram_app = getattr(ctx, "telegram_app", None)
        if telegram_app is None:
            return  # Telegram app tanımlı değilse hiçbir şey yapma

        chat_id = ctx.config.get("telegram_logging_chat_id", None)
        if not chat_id:
            return  # chat_id yoksa çık
        #print(chat_id)
        now = time.time()
        need_summary = False

        # force_summary=True ise veya 30 dakikadan fazla geçmişse summary mesajı at
        if force_summary or (now - LAST_SUMMARY_TIME) > SUMMARY_INTERVAL:
            need_summary = True

        # row_main içindeki değerleri örnek olarak alalım
        fgi = row_main.get("Fear_Greed_Index", 0.5)   # Korku endeksi
        news = row_main.get("News_Headlines", 0.0)    # Haberler
        funding = row_main.get("Funding_Rate", 0.0)   # Fonlama oranı
        ob = row_main.get("Order_Book_Num", 0.0)      # Emir defteri dengesi
        oi_1h = row_main.get("Open_Interest", 0.0)    # 1 saatlik Açık Pozisyon
        close_30m = row_main.get("Close_30m", 0.0)    # Son 30 dak. kapanış

        # Basit yorumlar/etiketler
        ob_result = "Satıcı Ağırlıklı" if ob <= 0 else "Alıcı Ağırlıklı"
        
        if funding > 0.01:
            funding_result = "Negatif"
        else:
            funding_result = "Pozitif"

        if fgi < 0.3:
            fgi_result = "İyi"
        elif fgi > 0.7:
            fgi_result = "Kötü"
        else:
            fgi_result = "Nötr"

        if news < -0.2:
            news_result = "Kötü"
        elif news > 0.2:
            news_result = "İyi"
        else:
            news_result = "Nötr"

        # 1) Summary Mesajı
        if need_summary:
              # MTF kararı
            mtf_decision = await self.decide_trade_mtf_with_pattern(
                    #df_15m=df_15m,
                    df_30m=df_30m,
                    df_1h=df_1h,
                    df_4h=df_4h,
                    df_1d=df_1d,
                    symbol=symbol,
                    min_score=5,
                    retest_tolerance=0.005,
                    ctx=ctx
                )
            final_act = mtf_decision["final_decision"]
            retest_info = mtf_decision["retest_info"]
            
            log_msg = (f"[{symbol}] => final={final_act}, "
                        f"30m={mtf_decision['score_30m']},1h={mtf_decision['score_1h']},4h={mtf_decision['score_4h']},1d={mtf_decision['score_1d']}, "
                        f"combined={mtf_decision['combined_score']}, retest={retest_info}")
            log(log_msg, "info")
            try:
                txt_summary = (
                f"Symbol: {symbol}\n"
                f"İndikatör (1 saat): {regime}\n"
                f"Son 30 dak. Kapanış: {close_30m}\n"
                f"Korku Endeksi: {fgi_result} ({fgi:.2f})\n"
                f"Haber Skoru: {news_result} ({news:.2f})\n"
                f"Fonlama Oranı: {funding:.4f} ({funding_result})\n"
                f"Emir Defteri: {ob:.2f} ({ob_result})\n"
                f"1 Saatlik Açık Pozisyon: {oi_1h}\n"
            )
                await telegram_app.bot.send_message(chat_id=chat_id, text=txt_summary)
                #LAST_SUMMARY_TIME = now  # son gönderim zamanını güncelle
                await asyncio.sleep(5)

        # 2) Pattern Kontrolü
        # -------------------------------------------------------------------
        # mtf_decision içinden pattern bilgilerini çekip, results_dict oluşturma
                results_dict = {
                    "30m": mtf_decision["patterns_30m"],
                    "1h":  mtf_decision["patterns_1h"],
                    "4h":  mtf_decision["patterns_4h"],
                    "1d":  mtf_decision["patterns_1d"]
                }
            
                txt_report = await self.format_pattern_results(row_main, results_dict)

                await telegram_app.bot.send_message(chat_id=chat_id, text=txt_report)
                LAST_SUMMARY_TIME = now  # son gönderim zamanını güncelle
                log(f"Message sent to Telegram:", "info")
                await asyncio.sleep(5)

            except Exception as e:
                    # log fonksiyonu projenizde farklı olabilir, burada örnek.
                log(f"Telegram send error (Summary): {e}", "error")

        # Örnek olarak current_price = close_30m kabul ediyoruz
        print(price)
        # Pattern’lerden puanlı olanları bul
        close_pattern_list = self.find_close_patterns(results_dict, price)     
       
        # 3) Alert Mesajı (Puanlı pattern varsa)
        if close_pattern_list:
            alert_text = "ALERT: TP değerine %5 - %10 yakın pattern'ler:\n\n"
            for cp in close_pattern_list:
                alert_text += (
                    f"- PRICE: {price}\n"

                    f"- Timeframe: {cp['timeframe']}\n"
                    f"  Pattern: {cp['pattern_type']}\n"
                    f"  Yön: {cp['direction']}\n"
                    f"  TP: {cp['take_profit']}\n"
                    f"  Fark Yüzdesi: {cp['fark_yuzdesi']}%\n"
                    f"  Puan: {cp['puan']}\n\n"
                )
            try:
                await telegram_app.bot.send_message(chat_id=chat_id, text=alert_text)
            except Exception as e:
                log(f"Telegram send error (Alert): {e}", "error")
        # Puanlı pattern yoksa alert mesajı gönderilmez.

 