# strategy/tradingview_strategy.py

import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.context import SharedContext
from core.openai import openai_connect
import json

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
from .detect_regime import get_all_regimes,combine_regime_and_pattern_signals,analyze_multi_tf_alignment,produce_realistic_signal
from trading_view.main_tv import generate_signals # <-- Pattern tespiti fonksiyonlarınızı (detect_elliott vb.) içeren modül
# Yukarıda 'pattern_lib.generate_signals(df)' demek için, 
#   en son paylaştığınız "generate_signals" kodunu oraya koyabilirsiniz.
from datetime import datetime
import time
import traceback
#from advanced_bot2.data_analytics.v2 import optimized_detection_pipeline
LAST_SUMMARY_TIME = 0  # Son summary gönderim zamanı (timestamp)
SUMMARY_INTERVAL = 3600  # 30 dakika = 1800 sn

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
        # await self.sync_account_equity()

        # # 3.2) Drawdown kontrolü
        # dd = self.pm.calc_drawdown(self.total_equity)
        # if dd > self.pm.max_drawdown:
        #     log(f"[DDCheck] => drawdown={dd:.2%}>{self.pm.max_drawdown:.2%} => closeAll", "warning")
        #     await self.close_all_positions(reason="drawdownStop")
        #     return
        
        # # 3.3) Karar mantığı (multi-timeframe + RL vs.)
        #df_main = ctx.df_map.get(symbol, {}).get("merged", None)
        #print(df_main.coloumns)
        # #print(df_main)
        # if df_main is None or len(df_main) < 10:
        #     return  # Yeterli veri yok

        # row_main = df_main.iloc[-1]
        st = ctx.symbol_map[symbol]
        # # MTF & RL 
        # regime = self.detect_regime(row_main, symbol)
        # base_p = ctx.param_for_regime.get(regime, ctx.param_for_regime["DEFAULT"])
        # obs = self.rl_agent.observe_environment({"symbol": symbol, "regime": regime})
        # action = self.rl_agent.select_action(obs)
        # merged_param = {
        #     "stop_atr_mult": action.get("stop_atr_mult", base_p["stop_atr_mult"]),
        #     "partial_levels": action.get("partial_levels", base_p["partial_levels"]),
        #     "partial_ratio": action.get("partial_ratio", base_p["partial_ratio"])
        # }

        # # Skorlamalar
        # short_sco = self.eval_short_term(row_main, merged_param, symbol)
        # med_sco = self.eval_medium_term(row_main, merged_param, symbol)
        # long_sco = self.eval_long_term(row_main, merged_param, symbol)
        # adv_s_s = self.eval_advanced_indicators_short(row_main, merged_param, symbol)
        # adv_s_m = self.eval_advanced_indicators_medium(row_main, merged_param, symbol)
        # adv_s_l = self.eval_advanced_indicators_long(row_main, merged_param, symbol)
        # adv_score = adv_s_s + adv_s_m + adv_s_l
        # macro_s = self.macro.evaluate_macro(row_main)
        # vol_sco = self.one_min_vol.evaluate_1m_volatility(df_main)
        # senti = self.sentiment_onchain(row_main)

        # st_score = short_sco + med_sco + long_sco + adv_score + macro_s + vol_sco + senti
        
        # total_s_s= short_sco +adv_s_s +macro_s 
        # # Örnek: Short/Med/Long aynı yönde ise ek puan verelim/ceza verelim.
        # synergy_bonus = 0
        # sum_sml = short_sco + med_sco + long_sco
        # if short_sco > 0 and med_sco > 0 and long_sco > 0:
        #     synergy_bonus += 1  # Hepsi pozitif => alım sinyalini güçlendir
        # if short_sco < 0 and med_sco < 0 and long_sco < 0:
        #     synergy_bonus -= 1  # Hepsi negatif => satış sinyalini güçlendir

        # st_score += synergy_bonus

        # final_action = self.combine_scenario_with_ensemble(regime, st_score)      
        # # panik / reentry
        # if self.detect_panic_signal(row_main, symbol):
        #     st.panic_count += 1
        # else:
        #     st.panic_count = 0
        # st.panic_mode = (st.panic_count >= self.panic_confirm_bars)
      
        # reentry_allow = False
        # if st.last_sell_time:
        #     mins_since = (datetime.utcnow() - st.last_sell_time).total_seconds() / 60.0
        #     if mins_since < self.reentry_window_bars and st.reentry_count < self.max_reentry:
        #         reentry_allow = True

       
        # log_msg = (f"[{symbol}] => final_act={final_action},holygrail_=,total_score={st_score},  "
        #             f"panic={st.panic_mode}, reentry={reentry_allow}, netPnL={st.net_pnl:.2f}, RL={action}")
        # log(log_msg, "info")      
        #df_1m = ctx.df_map.get(symbol, {}).get("1m", None)
        # df_5m = ctx.df_map.get(symbol, {}).get("5m", None)
        # df_15m = ctx.df_map.get(symbol, {}).get("15m", None)
        # df_30m = ctx.df_map.get(symbol, {}).get("30m", None)
        # df_1h = ctx.df_map.get(symbol, {}).get("1h", None)
        # df_4h = ctx.df_map.get(symbol, {}).get("4h", None)
        # df_1d = ctx.df_map.get(symbol, {}).get("1d", None)
        # df_1w = ctx.df_map.get(symbol, {}).get("1w", None)

        # force_summary=False
        # command_source=ctx.config["command_source"]
        # if command_source=="telegram": 
        #     force_summary=True
        #     ctx.config["command_source"]="app"


        # result = await self.send_telegram_messages(price=price,df_5m=df_5m,df_15m=df_15m, df_30m=df_30m, df_1h=df_1h, df_4h=df_4h, df_1d=df_1d,df_1w=df_1w, ctx=ctx, row_main=row_main, symbol=symbol, regime=regime, force_summary=force_summary)
        
        # if result:
        #     mtf_decision, summ_patterns = result
        #     #(".....",summ_patterns)
        
        #     short_closest_entry=summ_patterns["short_closest_entry"]
        #     short_min_tp=summ_patterns["short_min_tp"]
        #     short_max_sl=summ_patterns["short_max_sl"]
        #     long_closest_entry=summ_patterns["long_closest_entry"]
        #     long_min_tp=summ_patterns["long_min_tp"]
        #     long_max_sl=summ_patterns["long_max_sl"]      
        
        #     final_act = mtf_decision["final_decision"]
                
        #     log_msg = (f"[{symbol}] => final={final_act}, "
        #                     f"30m={mtf_decision['score_30m']},1h={mtf_decision['score_1h']},4h={mtf_decision['score_4h']},1d={mtf_decision['score_1d']}, "
        #                     f"combined={mtf_decision['combined_score']}")
        #     log(log_msg, "info")    
        # # log(f"[TVStrategy {symbol}] => pattern_score_30m={pattern_score_30m}, reason={reason_30m},detail_for_1h: {pattern_details_30m}", "info")
        # else:
        #     log("Info: Beklem süresi basladi", "info")
           

        #DEĞİŞİKLİK YAPILAN KISIMLAR (2): Eşikleri arttırarak sinyalleri güçlendir ---
        # final_action = 2  # 2 => HOLD
        # buy_threshold = 5   # önceki 4 yerine 5
        # sell_threshold = -5 # önceki -4 yerine -5
        
        #total_score=p_score_total + st_score 
       
        # reentry_allow = False
        # if st.last_sell_time:
        #      mins_since = (datetime.utcnow() - st.last_sell_time).total_seconds() / 60.0
        #      if mins_since < self.reentry_window_bars and st.reentry_count < self.max_reentry:
        #          reentry_allow = True 
       
   # 3.4) Pozisyon yok => BUY?
        # if total_score >= buy_threshold:
        #     final_action = "BUY"
        # elif total_score <= sell_threshold:
        #     final_action = "SELL"
        # else:
        #     final_action = "HOLD"

        # result = optimized_detection_pipeline(symbol, df_main)
        
        # print("\n⚡ Optimize Sonuçlar ⚡")
        # if 'error' in result:
        #     print(f"❌ Hata: {result['error']}")
        # else:
        #     print(f"🔮 Tahmin Olasılığı: {result['probability']}%")
        #     print(f"📈 Pozisyon Büyüklüğü: {result['position_size']:.4f}")
        #     print("\n📊 Performans Metrikleri:")
        #     for k, v in result['metrics'].items():
        #         print(f"▸ {k.upper():<10}: {v:.4f}")
    
        if not st.has_position:
            # if (not st.panic_mode) and (final_act ==  "BUY"):
            #     if retest_info is not None:
            #     # Bu, "kırılan neckline'a yakın" demek => alım
            #         await self.do_buy(ctx, df_30m.iloc[-1], symbol)
            #     else:
            #     # retest yok => belki "daha agresif" trade
            #     # veya retest bekleme => bir opsiyon
                pass
        else:
            # 3.5) Pozisyon var => SELL sinyali mi yoksa risk mgmt mi?
            # if final_act == "SELL":
            #     await self.do_full_close(ctx, df_30m.iloc[-1], symbol)
            # else:
            #     await self.handle_risk_management(ctx, df_30m, df_30m.iloc[-1], symbol, merged_param)
             pass


    def combine_scenario_with_ensemble(self, scenario, total_score) -> int:
        """
        Örnek:
          scenario="TREND_UP" => momentum => eger ensemble=-1 => belki hold? (zayıflat)
        """#        #
        #print(scenario)
        #print(total_score)

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
            if (oi_1h > OI_THRESHOLD):
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
                    oi_1h   > OI_THRESHOLD       
                    
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


    async def decide_trade_mtf_with_pattern(self,df_5m,df_15m,df_30m, df_1h, df_4h,df_1d, df_1w,
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
        get_regime_info = get_all_regimes(df_5m=df_5m,df_15m=df_15m,df_30m=df_30m,df_1h=df_1h,df_4h=df_4h,df_1d=df_1d,df_1w=df_1w)
        regime_info_5m= get_regime_info["5m"]               

        regime_info_15m= get_regime_info["15m"]               
        regime_info_30m= get_regime_info["30m"]
        regime_info_1h= get_regime_info["1h"]
        regime_info_4h= get_regime_info["4h"]
        regime_info_1d= get_regime_info["1d"]
        synergy_intraday = analyze_multi_tf_alignment(get_regime_info, combo_name="intraday")
        synergy_scalping = analyze_multi_tf_alignment(get_regime_info, combo_name="scalping")
        synergy_swing = analyze_multi_tf_alignment(get_regime_info, combo_name="swing")
        synergy_position = analyze_multi_tf_alignment(get_regime_info, combo_name="position")

       # final_side = synergy_intraday["final_side"]
       # score_details = synergy_intraday["score_details"]
       # break_out_note = synergy_intraday["break_out_note"]
        #total_score = synergy_intraday["total_score"]
        #main_regime = synergy_intraday["main_regime"]
        #patterns_used = synergy_intraday["patterns_used"]
       # alignment = synergy_intraday["alignment"]
       # print(synergy_intraday["alignment"])

        # print(synergy_intraday)
        # print(synergy_scalping)
        # print(synergy_swing)

        # 1) MTF Sinyal Çağrıları
        sig_5m  =  await generate_signals(df_5m, symbol,time_frame="5m",ml_model=None,max_bars_ago=80, retest_tolerance=0.001, require_confirmed=True)
        #result_5m=combine_regime_and_pattern_signals(regime_info_5m,sig_5m["pattern_trade_levels"])

        
        sig_15m =  await generate_signals(df_15m, symbol,time_frame="15m",ml_model=None,max_bars_ago=80, retest_tolerance=0.005, require_confirmed=True)
       # result_15m=combine_regime_and_pattern_signals(regime_info_15m,sig_15m["pattern_trade_levels"])
        result_15m=produce_realistic_signal(regime_info_15m,sig_15m["pattern_trade_levels"],"15m", "intraday")
        
        sig_30m =  await generate_signals(df_30m, symbol,time_frame="30m",ml_model=None,max_bars_ago=90,retest_tolerance=0.005,  require_confirmed=True)
        #result_30m=combine_regime_and_pattern_signals(regime_info_30m,sig_30m["pattern_trade_levels"])
        result_30m= produce_realistic_signal(regime_info_30m,sig_30m["pattern_trade_levels"],"30m", "intraday")

        sig_1h  =  await generate_signals(df_1h, symbol, time_frame="1h", ml_model=None,max_bars_ago=300,retest_tolerance=0.01,  require_confirmed=True)
       # result_1h=combine_regime_and_pattern_signals(regime_info_1h,sig_1h["pattern_trade_levels"])
        result_1h=produce_realistic_signal(regime_info_1h,sig_1h["pattern_trade_levels"],"1h", "intraday")

        sig_4h  =  await generate_signals(df_4h, symbol, time_frame="4h",ml_model=None ,max_bars_ago=300, retest_tolerance=0.01, require_confirmed=True)
       # result_4h=combine_regime_and_pattern_signals(regime_info_4h,sig_4h["pattern_trade_levels"])
        result_4h=produce_realistic_signal(regime_info_4h,sig_4h["pattern_trade_levels"],"4h", "intraday")

        sig_1d  =  await generate_signals(df_1d, symbol, time_frame="1d",ml_model=None ,max_bars_ago=300, retest_tolerance=0.01, require_confirmed=True)
        #result_1d=combine_regime_and_pattern_signals(regime_info_1d,sig_1d["pattern_trade_levels"])
        result_1d=produce_realistic_signal(regime_info_1d,sig_1d["pattern_trade_levels"],"4h", "intraday")

        #print(result_1h["patterns_used"])
      
        #print(sig_30m)
        score_5m = sig_5m["score"]
        score_15m = sig_15m["score"]

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
            "score_5m": score_5m,
                        "score_15m": score_15m,


            "score_30m": score_30m,
            "score_1h":  score_1h,
            "score_4h":  score_4h,
            "score_1d": score_1d,
            #"patterns_5m": result_5m["patterns_used"],

            "patterns_15m": result_15m["patterns_used"],

            "patterns_30m": result_30m["patterns_used"],
            "patterns_1h":  result_1h["patterns_used"],
            "patterns_4h":  result_4h["patterns_used"],
            "patterns_1d":  result_1d["patterns_used"],

            "break_out_note": retest_data,
            "combined_score": combined_score,
            "synergy_intraday":synergy_intraday,
            "synergy_scalping": synergy_scalping ,
       "synergy_swing": synergy_swing ,
        "synergy_position" : synergy_position,
        "get_regime_info":get_regime_info

        }

        
        # return {
        #     "final_decision": final_decision,
        #     "score_5m": score_5m,
        #                 "score_15m": score_15m,


        #     "score_30m": score_30m,
        #     "score_1h":  score_1h,
        #     "score_4h":  score_4h,
        #     "score_1d": score_1d,
        #     "patterns_5m": sig_5m["pattern_trade_levels"],

        #     "patterns_15m": sig_15m["pattern_trade_levels"],

        #     "patterns_30m": sig_30m["pattern_trade_levels"],
        #     "patterns_1h":  sig_1h["pattern_trade_levels"],
        #     "patterns_4h":  sig_4h["pattern_trade_levels"],
        #     "patterns_1d":  sig_1d["pattern_trade_levels"],

        #     "retest_info": retest_data,
        #     "combined_score": combined_score
        # }

    async def format_pattern_results(self,mtf_dict: dict,price) -> str:
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
        timeframes = ["15m","30m","1h","4h","1d"]
        
        for tf in timeframes:
            # Her timeframe dictionary'sini al
            tf_data = mtf_dict.get(tf, None)
            if not tf_data:
                # Pattern listesi yoksa / skip
                lines.append(f"\n--- {tf} => Desen bulunamadi ---")
                continue

            lines.append(f"\n--- {tf} Desen Sonuclari ---")

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
                    breakout_note   = pat.get("breakout_note")


                    # pattern_raw içinden ek bilgi istersek:
                    raw  = pat.get("pattern_raw", {})
                    #conf = raw.get("confirmed", False)
                    #patn = raw.get("pattern", "?")
                    

                        # Format -> 2 decimal
                    ep_s  = f"{ep:.2f}" if ep else "N/A"
                    sl_s  = f"{sl:.2f}" if sl else "N/A"
                    tp_s  = f"{tp:.2f}" if tp else "N/A"

                    lines.append(
                    f"<b>[{idx}] YÖN:</b> {dire}- {breakout_note}\n"
                    f"<b>Giris:</b> {ep_s}\n"
                    f"<b>Stop-loss:</b> {sl_s}\n"
                    f"<b>Kâr hedefi:</b> {tp_s}\n"
                    f"----------------\n"
                )

        final_text = "\n".join(lines)
        if not final_text.strip():
            final_text = "No pattern results found."
        return final_text
   
    # -------------------------------------------------------
    # 1) Pattern Puanlamasını Yapan Fonksiyon
    # -------------------------------------------------------
    def find_close_patterns(self, results_dict: dict, current_price, lower_threshold=5, upper_threshold=10):
        close_patterns = []
        
        for timeframe, pattern_list in results_dict.items():
            # Artık pattern_list bir dict değil, list:
            if not isinstance(pattern_list, list):
                # Hata veya uyarı
                continue
            
            for pattern_obj in pattern_list:
                direction = pattern_obj.get("direction")
                tp = pattern_obj.get("take_profit")

                if tp is not None:
                    fark_yuzdesi = abs(tp - current_price) / current_price * 100
                    if lower_threshold <= fark_yuzdesi <= upper_threshold:
                        puan = round(upper_threshold - fark_yuzdesi, 2)
                        close_patterns.append({
                            "timeframe": timeframe,
                            "pattern_type": pattern_obj.get("pattern_type"),
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
                                     df_5m,
                                     df_15m,
                                     df_30m,df_1h,
                    df_4h,
                    df_1d,df_1w, ctx: SharedContext, row_main, symbol, regime, force_summary=False):
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

        
        # 1) Summary Mesajı
        if need_summary:
            # row_main içindeki değerleri örnek olarak alalım
            fgi = row_main.get("Fear_Greed_Index", 0.5)   # Korku endeksi
            news = row_main.get("News_Headlines", 0.0)    # Haberler
            funding = row_main.get("Funding_Rate", 0.0)   # Fonlama oranı
            ob = row_main.get("Order_Book_Num", 0.0)      # Emir defteri dengesi
            oi_1h = row_main.get("Open_Interest", 0.0)    # 1 saatlik Açık Pozisyon
            close_5m = row_main.get("Close_5m", 0.0)    # Son 30 dak. kapanış
            resistance_15m=row_main.get("Resistance_15m",0.0)
            support_15m = row_main.get("Support_15m",0.0)
            resistance_30m=row_main.get("Resistance_30m",0.0)
            support_30m = row_main.get("Support_30m",0.0)
            resistance_1h=row_main.get("Resistance_1h",0.0)
            support_1h = row_main.get("Support_1h",0.0)
            resistance_4h=row_main.get("Resistance_4h",0.0)
            support_4h = row_main.get("Support_4h",0.0)
            resistance_1d=row_main.get("Resistance_1d",0.0)
            support_1d = row_main.get("Support_1d",0.0)
   
            # Basit yorumlar/etiketler
            # Order Book Yorumu (OB)
            # "ob" değeri, emir defterinde alıcı ve satıcı yoğunluğunu temsil eder.
            #  - ob > 0  => emir defterinde alıcılar daha baskın (pozitif, yukarı yönlü baskı)
            #  - ob <= 0 => emir defterinde satıcılar daha baskın (negatif, aşağı yönlü baskı)

            # if ob <= 0:
            #     ob_result = "NEGATİF - Emir defteri satıcı baskılı (düşüş potansiyeli)"
            # else:
            #     ob_result = "POZİTİF - Emir defteri alıcı baskılı (yükseliş potansiyeli)"

            funding_result=""     
            fgi_result = ""

            if funding :
                if funding > 0.01:
                    funding_result = "Pozitif (Long'lar ödüyor, short avantajı)"
                elif funding < -0.01:
                    funding_result = "Negatif (Short'lar ödüyor, long avantajı)"
                else:
                    funding_result = "Nötr (Short ve Long lar arasi avantaj yok)"
            # ------------------------------------------
            # Fear & Greed Index (fgi)
            # ------------------------------------------
            if fgi < 0.3:
                fgi_result = "Negatif (Piyasa korku halinde)"
            elif fgi > 0.7:
                fgi_result = "Pozitif (Piyasa açgözlü)"
            else:
                fgi_result = "Nötr"
           # ------------------------------------------
            # News (haberler)
            # ------------------------------------------
            if news < -0.2:
                news_result = "Negatif (Kötü haber akışı)"
            elif news > 0.2:
                news_result = "Pozitif (İyi haber akışı)"
            else:
                news_result = "Nötr"

              # MTF kararı
            mtf_decision = await self.decide_trade_mtf_with_pattern(
                df_5m=df_5m,
                    df_15m=df_15m,
                    df_30m=df_30m,
                    df_1h=df_1h,
                    df_4h=df_4h,
                    df_1d=df_1d,df_1w=df_1w,
                    symbol=symbol,
                    min_score=5,
                    retest_tolerance=0.005,
                    ctx=ctx
                )
            
            final_act = mtf_decision["final_decision"]
            #retest_info = mtf_decision["retest_info"]
            synergy_intraday =mtf_decision["synergy_intraday"]
            synergy_scalping =mtf_decision["synergy_scalping"]
            synergy_swing =mtf_decision["synergy_swing"]
            synergy_position =mtf_decision["synergy_position"]


       
            log_msg = (f"[{symbol}] => final={final_act}, "
                        f"5m={mtf_decision['score_5m']},15m={mtf_decision['score_15m']},30m={mtf_decision['score_30m']},1h={mtf_decision['score_1h']},4h={mtf_decision['score_4h']},1d={mtf_decision['score_1d']}, "
                        f"combined={mtf_decision['combined_score']}")
            log(log_msg, "info")
            try:
                txt_summary = (
                f"<b>Coin:</b> {symbol}\n"
                # f"----------------\n"
                # f"<b>Kisa Vadeli Islemler[5 dakika, 15 dakika,  1 saat]:</b> {synergy_scalping}\n"
                # f"----------------\n"
                # f"<b>Gün İçi  Islemler[15 dakika, 1 Saat,  4 saat]:</b> {synergy_intraday}\n"
                # f"----------------\n"
                # f"<b>Orta Vadeli Islemler[1 saat, 4 Saat,  1 gün]:</b> {synergy_swing}\n"
                # f"----------------\n"
                # f"<b>Uzun Vadeli Islemler[1 günlük, 1 Haftalik]:</b> {synergy_position}\n"
                # f"----------------\n"

                f"<b>son 5 dak. kapanış:</b> {close_5m:.4f}\n"
                f"----------------\n"
               # f"<b>i̇ndikatör YÖN (1 saat):</b> {regime}\n"
                f"----------------\n"
                f"<b>i̇ndikatör ort. destek-direnç (15 dakika):</b> {support_15m:.4f}, {resistance_15m:.4f}\n"
                f"----------------\n"
              
                f"<b>i̇ndikatör ort. destek-direnç (30 dakika):</b> {support_30m:.4f}, {resistance_30m:.4f}\n"
                f"----------------\n"
                f"<b>i̇ndikatör ort. destek-direnç (1 saat):</b> {support_1h:.4f}, {resistance_1h:.4f}\n"
                f"----------------\n"
                f"<b>i̇ndikatör ort. destek-direnç (4 saat):</b> {support_4h:.4f}, {resistance_4h:.4f}\n"
                f"----------------\n"
                f"<b>korku endeksi:</b> {fgi_result} ({fgi:.2f})\n"
                f"----------------\n"
                f"<b>Coin haberleri :</b> {news_result} ({news})\n"
                f"----------------\n"
                f"<b>Fonlama oranı:</b> {funding:.2f} ({funding_result})\n"
                f"----------------\n"
                #f"<b>emir defteri:</b> {ob:.2f} ({ob_result})\n"
                f"----------------\n"
                f"<b>1 saatlik açık pozisyon:</b> {oi_1h:.2f}\n"
                f"----------------\n"

                )
                await telegram_app.bot.send_message(chat_id=chat_id, text=txt_summary, parse_mode="HTML")
                # mtf_decision içinden pattern bilgilerini çekip, results_dict oluşturma
                results_dict = {
                    #"5m": mtf_decision["patterns_5m"],

                    "15m": mtf_decision["patterns_15m"],

                    "30m": mtf_decision["patterns_30m"],
                    "1h":  mtf_decision["patterns_1h"],
                    "4h":  mtf_decision["patterns_4h"],
                    "1d":  mtf_decision["patterns_1d"]
                }
                
                
                txt_report = await self.format_pattern_results(results_dict,price)
                time_frame_infos=mtf_decision["get_regime_info"]
                await telegram_app.bot.send_message(chat_id=chat_id, text=txt_report, parse_mode="HTML")

                # Open AI baglantisi ve yorumlari alinir.
                time_frame_infos_str = json.dumps(time_frame_infos, ensure_ascii=False)  # JSON string formatına çevir  
                prompt = f"Verilen tüm analiz ve pattern sonuclarini birlikte degerlendir ve gercekci bir Short ve Long onerisi yap. {time_frame_infos_str} {txt_report}{txt_summary}"              
                open_ai= ctx.config.get("open_ai", False)
                print(prompt)
                if  open_ai:
                    openai_model = ctx.config.get("openai_model", None)
                    response_text=await openai_connect(prompt,openai_model)
                    formatted_text = f"<b>ChatGPT Yorumu:</b>\n\n{response_text}"  # Kalın başlık ekleyerek gönderiyoruz
                    # Mesaj uzunluğu 4096 karakteri geçiyorsa bölerek gönder
                    max_length = 4096
                    for i in range(0, len(formatted_text), max_length):
                        await telegram_app.bot.send_message(chat_id=chat_id, text=formatted_text[i:i+max_length], parse_mode="HTML")
                    LAST_SUMMARY_TIME = now 
                    log(f"OPenAI Message sent to Telegram:", "info")
                    await asyncio.sleep(5)
                else :
                    await telegram_app.bot.send_message(chat_id=chat_id, text=txt_report, parse_mode="HTML")
                    LAST_SUMMARY_TIME = now 
                    log(f"Pattern Message sent to Telegram:", "info")
                    await asyncio.sleep(5)


                # 2) Pattern Kontrolü
                # -------------------------------------------------------------------
               # print(txt_report)
                #await telegram_app.bot.send_message(chat_id=chat_id, text=txt_report, parse_mode="HTML")
                #LAST_SUMMARY_TIME = now  # son gönderim zamanını güncelle
                #log(f"Message sent to Telegram:", "info")
                #await asyncio.sleep(5)
                summ_patterns=await self.summarize_patterns(results_dict, price)
                return mtf_decision,summ_patterns

            except Exception as e:
                log(f"[Telegram send ilk error ): => {e}\n{traceback.format_exc()}", "error")

        # Örnek olarak current_price = close_30m kabul ediyoruz
        #print(price)
        # Pattern’lerden puanlı olanları bul
        close_pattern_list = self.find_close_patterns(results_dict, price)     
       
        # 3) Alert Mesajı (Puanlı pattern varsa)
        if close_pattern_list:
            alert_text = "ALERT: TP değerine %5 - %10 yakın pattern'ler:\n\n"
            for cp in close_pattern_list:
                alert_text += (
                    f"- PRICE: {price}\n"

                    f"- Zaman araligi: {cp['timeframe']}\n"
                    f"  Desen: {cp['pattern_type']}\n"
                    f"  Yön: {cp['direction']}\n"
                    f"  TP: {cp['take_profit']}\n"
                    f"  Fark Yüzdesi: {cp['fark_yuzdesi']}%\n"
                    f"  Puan: {cp['puan']}\n\n"
                                    f"----------------\n"

                )
            try:
               pass #await telegram_app.bot.send_message(chat_id=chat_id, text=alert_text, parse_mode="HTML")
            except Exception as e:
                #log(f"Telegram send error (Alert): {e}", "error")
                log(f"[Telegram send error (Alert): => {e}\n{traceback.format_exc()}", "error")
        # Puanlı pattern yoksa alert mesajı gönderilmez.

    async def summarize_patterns(self, results_dict, current_price):
        """
        Summarizes pattern results from results_dict.
        """
        # SHORT & LONG pattern lists
        short_patterns = []
        long_patterns  = []
        #print(results_dict)
        
        # Iterate over each timeframe's dictionary
        for timeframe, patterns_by_type in results_dict.items():
            # Ensure that we have a dictionary for pattern types
            if not isinstance(patterns_by_type, dict):
                log(f"Expected dict for patterns in timeframe {timeframe}, got {type(patterns_by_type)}", "error")
                continue

            # Now iterate over each pattern type and its associated list
            for pattern_type, pattern_list in patterns_by_type.items():
                if not isinstance(pattern_list, list):
                    log(f"Expected list for patterns in timeframe {timeframe}, pattern type {pattern_type}, got {type(pattern_list)}", "error")
                    continue

                # Process each pattern in the list
                for pattern in pattern_list:
                    # Ensure the pattern is a dictionary before processing
                    if not isinstance(pattern, dict):
                        log(f"Unexpected pattern type in timeframe {timeframe} for pattern type {pattern_type}: {pattern} (type {type(pattern)})", "error")
                        continue

                    # Now safe to access dictionary keys
                    direction = pattern.get("direction", None)
                    entry_price = pattern.get("entry_price", None)

                    # Collect patterns based on direction
                    if direction == "SHORT":
                        short_patterns.append(pattern)
                    elif direction == "LONG":
                        long_patterns.append(pattern)
                    else:
                        log(f"Pattern without valid direction in timeframe {timeframe} for pattern type {pattern_type}: {pattern}", "debug")
        
        # Analyze SHORT patterns
        short_closest_entry = None
        short_min_tp = None
        short_max_sl = None

        if short_patterns:
            # Get pattern with entry price closest to current_price
            short_patterns_sorted = sorted(
                short_patterns,
                key=lambda p: abs(p.get("entry_price", float("inf")) - current_price)
            )
            short_closest_entry = short_patterns_sorted[0].get("entry_price")

            # Get lowest take_profit among valid patterns
            valid_tps = [p.get("take_profit") for p in short_patterns if p.get("take_profit") is not None]
            if valid_tps:
                short_min_tp = min(valid_tps)

            # Get highest stop_loss among valid patterns
            valid_sls = [p.get("stop_loss") for p in short_patterns if p.get("stop_loss") is not None]
            if valid_sls:
                short_max_sl = max(valid_sls)

        # Analyze LONG patterns
        long_closest_entry = None
        long_min_tp = None
        long_max_sl = None

        if long_patterns:
            long_patterns_sorted = sorted(
                long_patterns,
                key=lambda p: abs(p.get("entry_price", float("inf")) - current_price)
            )
            long_closest_entry = long_patterns_sorted[0].get("entry_price")

            valid_tps = [p.get("take_profit") for p in long_patterns if p.get("take_profit") is not None]
            if valid_tps:
                long_min_tp = min(valid_tps)

            valid_sls = [p.get("stop_loss") for p in long_patterns if p.get("stop_loss") is not None]
            if valid_sls:
                long_max_sl = max(valid_sls)

        return {
            "short_closest_entry": short_closest_entry,
            "short_min_tp": short_min_tp,
            "short_max_sl": short_max_sl,
            "long_closest_entry": long_closest_entry,
            "long_min_tp": long_min_tp,
            "long_max_sl": long_max_sl
        }
