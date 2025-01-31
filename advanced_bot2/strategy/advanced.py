import numpy as np
import traceback
from datetime import datetime, timedelta
import os
from core.context import SharedContext
from core.logging_setup import log
from strategy.base import IStrategy
from strategy.paper_order import PaperOrderManager
from core.order_manager import RealOrderManager
from collectors.macro_collector import MacroCollector
from collectors.onchain_collector import OnChainCollector
from collectors.one_minute_collector import OneMinVolCollector
from collectors.orderbook_analyzer import OrderbookAnalyzer
from inference.rl_agent import RLAgent
from exchange_clients.binance_spot_manager_async import BinanceSpotManagerAsync
from portfolio.portfolio_manager import PortfolioManager
from core.trade_logger import append_trade_record, get_last_net_pnl

class UltraProStrategyV3_4_6(IStrategy):
    """
    v3.4.6 => RL + MTF + Spot emir yönetimi.
    Örnek strateji altyapısı.

    Tek coin + USDT senaryosunda, pozisyon kapandığında
    tüm bakiye tekrar USDT'ye döneceğinden, 'sync_account_equity'
    fonksiyonunda sadece USDT bakiyesini okumak yeterli olur.
    """

    def name(self) -> str:
        return "UltraProStrategyV3_4_6"

    def __init__(
        self,
        ctx: SharedContext,
        exchange_client: BinanceSpotManagerAsync,
        max_risk=0.4,
        initial_eq=10000.0,
        max_dd=0.15
    ):
        super().__init__()
        self.ctx = ctx
        self.exchange_client = exchange_client
        self.pm = PortfolioManager(max_risk, initial_eq, max_dd)
        self.total_equity = initial_eq

        # Collector ve ek bileşenler
        self.macro = MacroCollector()
        self.onchain = OnChainCollector()
        self.orderbook = OrderbookAnalyzer()
        self.one_min_vol = OneMinVolCollector(5, 0.01)

        # RLAgent => load_agents'da yüklenmiş halini kullanıyoruz:
        self.rl_agent = self.ctx.rl_agent

        # Strateji parametreleri
        self.panic_confirm_bars = 2
        self.max_reentry = 2
        self.reentry_window_bars = 10

    # --------------------------------------------------
    # 1) Mevcut pozisyonları senkronize et
    # --------------------------------------------------
    async def initialize_positions(self):
        try:
            account_info = await self.exchange_client.get_account_info()
            if not account_info or "balances" not in account_info:
                log("Hesap bilgisi çekilemedi veya 'balances' yok.", "error")
                return

            for symbol in self.ctx.config["symbols"]:
                st = self.ctx.symbol_map[symbol]
                if symbol.endswith("USDT"):
                    base_asset = symbol[:-4]
                else:
                    base_asset = symbol.replace("USDT", "")

                bal = next((b for b in account_info["balances"] if b["asset"] == base_asset), None)
                if not bal:
                    continue

                free_amt = float(bal["free"])
                locked_amt = float(bal["locked"])
                total_amt = free_amt + locked_amt

                if total_amt > 0.0:
                    st.has_position = True
                    st.quantity = total_amt
                    st.entry_price = 0.0
                    st.highest_price = 0.0
                    log(f"[initialize_positions] {symbol} => cüzdanda {total_amt} {base_asset} var. Pozisyon olarak işaretleniyor.", "info")
                    last_pnl = await get_last_net_pnl(symbol)
                    st.net_pnl = last_pnl
                    log(f"[initialize_positions] => {symbol} netPnL from CSV => {last_pnl}", "info")
                    self.pm.update_position(symbol, True, total_amt, 0.0)
                else:
                    log(f"[initialize_positions] {symbol} => cüzdanda {base_asset} yok (0).", "info")

        except Exception as e:
            log(f"[initialize_positions] Hata => {e}\n{traceback.format_exc()}", "error")

    # --------------------------------------------------
    # 2) USDT bakiyeyi güncelle
    # --------------------------------------------------
    async def sync_account_equity(self):
        try:
            usdt_balance = await self.exchange_client.get_balance("USDT")
            self.total_equity = usdt_balance
            log(f"[sync_account_equity] => total_equity={usdt_balance:.2f}", "info")
        except Exception as e:
            log(f"[sync_account_equity] => {e}\n{traceback.format_exc()}", "error")

    async def analyze_data(self, ctx: SharedContext):
        pass

    # --------------------------------------------------
    # 3) Her fiyat güncellemesinde çağrılan ana fonksiyon
    # --------------------------------------------------
    async def on_price_update(self, ctx: SharedContext, symbol: str, price: float):
        # 3.1) Hesap bakiyesi & drawdown kontrol
        await self.sync_account_equity()
        dd = self.pm.calc_drawdown(self.total_equity)
        if dd > self.pm.max_drawdown:
            log(f"[DDCheck] => drawdown={dd:.2%}>{self.pm.max_drawdown:.2%} => closeAll", "warning")
            await self.close_all_positions(reason="drawdownStop")
            return

        df_main = ctx.df_map.get(symbol, {}).get("1m", None)
        if df_main is None or len(df_main) < 10:
            return  # Yeterli veri yok

        row_main = df_main.iloc[-1]
        st = ctx.symbol_map[symbol]

        # 3.2) Rejim tespiti
        regime = self.detect_regime(row_main, symbol)
        self.rsi_bins = 5
        self.stoch_bins = 5
        self.drawdown_bins = 5

        
             # (1) Rejimi integer olarak al
        regime_bin = self.detect_regime(row_main, symbol)  
        # detect_regime => 0 veya 1 döndürüyor

        # (2) RSI bin
        rsi_val = row_main.get("RSI_5m", 50.0)
        rsi_bin = min(int(rsi_val // (100 / 5)), 4)  # 0..4

        # (3) Stoch bin
        stoch_val = row_main.get("StochK_5m", 50.0)
        stoch_bin = min(int(stoch_val // (100 / 5)), 4)

        # (4) haspos bin
        haspos_bin = 1 if st.has_position else 0

        # (5) dd bin
        dd_ratio = min(dd / self.pm.max_drawdown, 1.0)
        dd_bin = int(dd_ratio * (5 - 1))  # 0..4

        current_state = (regime_bin, rsi_bin, stoch_bin, haspos_bin, dd_bin)

        action = self.rl_agent.select_action(current_state)

        # Diğer skorlamalar (sizce lazımsa)
        base_p = ctx.param_for_regime.get(regime, ctx.param_for_regime["DEFAULT"])
        short_sco = self.eval_short_term(row_main, base_p, symbol)
        med_sco   = self.eval_medium_term(row_main, base_p, symbol)
        long_sco  = self.eval_long_term(row_main, base_p, symbol)
        adv_s_s   = self.eval_advanced_indicators_short(row_main, base_p, symbol)
        adv_s_m   = self.eval_advanced_indicators_medium(row_main, base_p, symbol)
        adv_s_l   = self.eval_advanced_indicators_long(row_main, base_p, symbol)
        adv_score = adv_s_s + adv_s_m + adv_s_l
        macro_s   = self.macro.evaluate_macro(row_main)
        vol_sco   = self.one_min_vol.evaluate_1m_volatility(df_main)
        senti     = self.sentiment_onchain(row_main)

        total_score = short_sco + med_sco + long_sco + adv_score + macro_s + vol_sco + senti

        # final_action => Skor mantığı + RL aksiyonu
        X = 2  # default HOLD
        if total_score >= 4:
            X = 1  # BUY
        elif total_score <= -4:
            X = 0  # SELL

        # RL eylemiyle birleştir (İSTEĞE BAĞLI)
        # Örneğin, RL SELL (0) derse final_action=0 vs. RL BUY (1) derse final_action=1
        # Bu tasarım tamamen size kalmış
        # Aşağıda, basitçe RL aksiyonu her şeyin üstünde tutan bir yaklaşım
         #action == 0 => SELL, 1 => BUY, 2 => HOLD, 3 => PARTIAL gibi.
        final_action = action

        # 3.3) panik / reentry hesapları
        if self.detect_panic_signal(row_main, symbol):
            st.panic_count += 1
        else:
            st.panic_count = 0
        st.panic_mode = (st.panic_count >= self.panic_confirm_bars)

        reentry_allow = False
        # if st.last_sell_time:
        #     mins_since = (datetime.utcnow() - st.last_sell_time).total_seconds() / 60.0
        #     if mins_since < self.reentry_window_bars and st.reentry_count < self.max_reentry:
        #         reentry_allow = True

        log_msg = (
            f"[{symbol}] => RL_action={action}, total_score={total_score}, "
            f"final_action={X}, panic={st.panic_mode}, "
            f"reentry={reentry_allow}, netPnL={st.net_pnl:.2f}"
        )
        log(log_msg, "info")

        # 3.4) İşlem mantığı
        if not st.has_position:
            if (not st.panic_mode) and (final_action == 1):
                propose_usd = ctx.config.get("trade_amount_usdt", 20)
                print(propose_usd,self.total_equity)

                if self.pm.check_portfolio_risk(symbol, propose_usd, self.total_equity):
                    if st.last_sell_time is None or reentry_allow:
                        await self.do_buy(ctx, row_main, symbol, base_p)

                        if reentry_allow:
                            st.reentry_count += 1
                else:
                    log(f"[{symbol}] => skip buy => port risk", "info")
        else:
            if final_action == 0:
                await self.do_full_close(ctx, row_main, symbol, reason="final SELL")
            elif final_action == 3:
                # PARTIAL => bir miktar sat, mesela %30
                part_qty = st.quantity * 0.3
                if part_qty > 0:
                    await self.do_partial_sell(ctx, symbol, part_qty, price)
            else:
                # 2 => HOLD, risk mgmt
                await self.handle_risk_management(ctx, df_main, row_main, symbol, base_p)

    # --------------------------------------------------
    # 4) Tüm pozisyonları kapatma
    # --------------------------------------------------
    async def close_all_positions(self, reason: str):
        for s in self.ctx.config["symbols"]:
            st = self.ctx.symbol_map[s]
            if st.has_position:
                df_main = self.ctx.df_map.get(s, {}).get("1m", None)
                if df_main is not None and len(df_main) > 0:
                    row_main = df_main.iloc[-1]
                    await self.do_full_close(self.ctx, row_main, s, reason=reason)

    # --------------------------------------------------
    # 5) BUY - FULL_CLOSE - PARTIAL_SELL
    # --------------------------------------------------
    async def do_buy(self, ctx: SharedContext, row_main, symbol, param):
        st = ctx.symbol_map[symbol]
        if st.has_position:
            return
        px = row_main["Close"]
        log(f"[{symbol}] do_buy => px={px:.4f}", "info")
        
        raw_qty = ctx.config.get("trade_amount_usdt", 20.0) / px
        st.has_position = True
        st.quantity = raw_qty
        st.entry_price = px
        st.highest_price = px
        self.pm.update_position(symbol, True, raw_qty, px)
        print(st.has_position)

        try:
            if ctx.config.get("paper_trading", True):
                log(f"[{symbol}] Paper BUY => qty={raw_qty:.4f}", "info")
                print("paper_trading")

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
        print("hello")

    async def do_full_close(self, ctx: SharedContext, row_main, symbol, reason=""):
        st = ctx.symbol_map[symbol]
        if not st.has_position:
            return
        px = row_main["Close"]
        entry = st.entry_price
        qty = st.quantity

        realized = (px - entry) * qty
        old_npl = st.net_pnl
        new_npl = old_npl + realized
        log(
            f"[{symbol}] => FULL CLOSE => px={px:.2f}, reason={reason}, "
            f"realizedPnL={realized:.2f}, oldNPL={old_npl:.2f}, newNPL={new_npl:.2f}",
            "info"
        )

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
        side = "SELL" if qty > 0 else "BUY"
        await append_trade_record(
            symbol=symbol,
            side=side,
            qty=qty,
            price=px,
            realized_pnl=realized,
            net_pnl=new_npl
        )
        st.has_position = False
        st.quantity = 0.0
        st.entry_price = 0.0
        st.highest_price = 0.0
        st.last_sell_time = datetime.utcnow()
        st.reentry_count = 0
        self.pm.update_position(symbol, False, 0.0, 0.0)
        # RLAgent'e ödül ekle (örnek)
        # Q update => state=(1,?), action=0, reward=realized, next_state=(0,?), done=False
        self.rl_agent.update(
            state=(1, 1),   # Örneğin: (regime_bin, haspos_bin=1) => Tamamen size bağlı
            action=0,       # SELL
            reward=realized,
            next_state=(1, 0),
            done=False
        )
        self.rl_agent.decay_epsilon()
        self.rl_agent.save_qtable("q_table.npy")

        

    async def do_partial_sell(self, ctx: SharedContext, symbol: str, qty: float, px: float):
        st = ctx.symbol_map[symbol]
        if not st.has_position or qty <= 0:
            return

        realizedPnL = (px - st.entry_price) * qty
        old_pnl = st.net_pnl
        new_pnl = old_pnl + realizedPnL
        log(
            f"[{symbol}] partial SELL => qty={qty:.4f}, px={px:.4f}, realizedPnL={realizedPnL:.2f}, "
            f"oldPNL={old_pnl:.2f}, newPNL={new_pnl:.2f}",
            "info"
        )

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
        await append_trade_record(
            symbol=symbol,
            side="SELL",
            qty=qty,
            price=px,
            realized_pnl=realizedPnL,
            net_pnl=new_pnl
        )

        # RLAgent tablo güncellemesi
        self.rl_agent.update(
            state=(1, 1),  # Örnek => (regime_bin=1, haspos_bin=1)
            action=3,      # PARTIAL
            reward=realizedPnL,
            next_state=(1, 1),
            done=False
        )
        self.rl_agent.decay_epsilon()
        self.rl_agent.save_qtable("q_table.npy")

        st.quantity -= qty
        if st.quantity <= 0:
            st.has_position = False
            st.quantity = 0.0
            st.entry_price = 0.0
            st.highest_price = 0.0

        self.pm.update_position(symbol, st.has_position, st.quantity, st.entry_price)

    # --------------------------------------------------
    # 6) Risk Yönetimi
    # --------------------------------------------------
    async def handle_risk_management(self, ctx: SharedContext, df_main, row_main, symbol: str, param: dict):
        st = ctx.symbol_map[symbol]
        if not st.has_position:
            return
        px = row_main["Close"]
        if px > st.highest_price:
            st.highest_price = px

        gain_ratio = (px - st.entry_price) / (st.entry_price + 1e-9)
        for i, lvl in enumerate(param["partial_levels"]):
            if i >= len(st.partial_done):
                break
            if (not st.partial_done[i]) and gain_ratio > lvl:
                part_qty = st.quantity * param["partial_ratio"]
                if part_qty > 0:
                    await self.do_partial_sell(ctx, symbol, part_qty, px)
                    st.partial_done[i] = True
                    st.quantity -= part_qty
                    if st.quantity <= 0:
                        break

        trailing_stop = st.highest_price - self.calc_atr_stop_dist(st.highest_price, param["stop_atr_mult"])
        if px < trailing_stop:
            log(f"[{symbol}] => trailingStop => closeAll", "warning")
            await self.do_full_close(ctx, row_main, symbol, reason="trailingStop")

    # --------------------------------------------------
    # 7) İndikatör Fonksiyonları
    # --------------------------------------------------
    def sentiment_onchain(self, row_main) -> int:
        cnt = 0
        fgi = row_main.get("Fear_Greed_Index", 0.5)
        news = row_main.get("News_Headlines", 0.0)
        funding = row_main.get("Funding_Rate", 0.0)
        ob = row_main.get("Order_Book_Num", 0.0)
        if funding > 0.01:
            cnt += 1
        if fgi < 0.3:
            cnt += 1
        elif fgi > 0.7:
            cnt -= 1
        if news < -0.2:
            cnt -= 1
        elif news > 0.2:
            cnt += 1
        if ob < 0:
            cnt -= 1
        elif ob > 0:
            cnt += 1
        return cnt

    def detect_panic_signal(self, row_main, symbol) -> bool:
        cnt = 0
        fgi = row_main.get("Fear_Greed_Index", 0.5)
        news = row_main.get("News_Headlines", 0.0)
        funding = row_main.get("Funding_Rate", 0.0)
        orderbook_val = row_main.get("Order_Book_Num", 0.0)
        if funding > 0.01:
            cnt += 1
        if fgi > 0.7:
            cnt += 1
        if news < -0.2:
            cnt += 1
        if orderbook_val < 0:
            cnt += 1
        return (cnt >= 2)

    def detect_regime(self, row_main, symbol) -> str:
        adx_4h = row_main.get("ADX_4h", 20.0)
        if adx_4h > 25:
            return "TREND"
        else:
            return "RANGE"

    def eval_short_term(self, row_s, param, symbol) -> float:
        rsi_5m = row_s.get("RSI_5m", 50.0)
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
        c_m = row_m.get("Close", 0)
        sc = st_dir
        if c_m > ichiA:
            sc += 1
        return sc

    def eval_long_term(self, row_l, param, symbol) -> float:
        t3_4h = row_l.get("T3_4h", None)
        macd_4h = row_l.get("MACD_4h", 0.0)
        c_l = row_l.get("Close", 0)
        sc = 0
        if t3_4h and c_l > t3_4h:
            sc += 1
        if macd_4h > 0:
            sc += 1
        return sc

    def eval_advanced_indicators_short(self, row_s, param, symbol) -> float:
        sc = 0
        pivot_5m = row_s.get("Pivot_5m", None)
        c_s = row_s.get("Close", 0)
        if pivot_5m and c_s > pivot_5m:
            sc += 0.5
        cdl_e = row_s.get("CDL_ENGULFING_5m", 0)
        if cdl_e > 0:
            sc += 0.5
        return sc

    def eval_advanced_indicators_medium(self, row_m, param, symbol) -> float:
        sc = 0
        bbmid = row_m.get("BBMid_1h", None)
        c_m = row_m.get("Close", 0)
        if bbmid and c_m > bbmid:
            sc += 0.5
        stoch_rsi = row_m.get("StochRSI_1h", 0.5)
        if stoch_rsi > 0.8:
            sc += 0.5
        return sc

    def eval_advanced_indicators_long(self, row_l, param, symbol) -> float:
        sc = 0
        fib618 = row_l.get("Fibo_61.8_4h", None)
        c_l = row_l.get("Close", 0)
        if fib618 and c_l > fib618:
            sc += 0.5
        mvrv = row_l.get("MVRV_Z_4h", 0.0)
        if mvrv < -0.5:
            sc += 0.5
        return sc

    def calc_atr_stop_dist(self, ref_px, mult):
        base = ref_px * 0.02
        return base * mult


async def load_agents(ctx: SharedContext):
    rl_model_path = ctx.config.get("rl_model_path", "")
    if rl_model_path and os.path.exists(rl_model_path):
        rl = RLAgent(
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            state_shape=(2, 2),  # Örnek => (regime_bin, haspos_bin)
            n_actions=4,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        rl.load_qtable(rl_model_path)
        ctx.rl_agent = rl
        rl.loaded = True
        log(f"[main] RL Q-tablosu yüklendi => {rl_model_path}", "info")
    else:
        rl = RLAgent(
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            state_shape=(2, 2),  # (regime_bin, haspos_bin)
            n_actions=4,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        rl.loaded = False
        ctx.rl_agent = rl
        log("[main] RL Q-tablosu bulunamadı => Geçiliyor", "warning")
import json
import random
import numpy as np

class RLAgent:
    """
    Tablo tabanlı Q-learning ajanı.
    Eylem seti: 0=SELL, 1=BUY, 2=HOLD, 3=PARTIAL
    State shape: (2, 5, 5, 2, 5) => (regime, rsi_bin, stoch_bin, haspos, dd_bin)
    """

    def __init__(
        self,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        state_shape=(2, 5, 5, 2, 5),
        n_actions=4,
        epsilon_min=0.01,
        epsilon_decay=0.990
    ):
        """
        :param state_shape: (2,5,5,2,5) => 2 * 5 * 5 * 2 * 5 = 1000 state
        :param n_actions: 4
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.state_shape = state_shape
        self.n_actions = n_actions

        # Q tablosu, boyutu: (*state_shape, n_actions)
        self.q_table = np.zeros((*self.state_shape, self.n_actions), dtype=np.float32)

    def select_action(self, state):
        """
        Epsilon-greedy aksiyon seçimi.
        state: (regime, rsi_bin, stoch_bin, haspos, dd_bin) gibi bir tuple
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            q_vals = self.q_table[state]
            return int(np.argmax(q_vals))

    def update(self, state, action, reward, next_state, done=False):
        """
        Q(s,a) = Q(s,a) + alpha * (r + gamma*max(Q(s',:)) - Q(s,a))
        done=True ise next_q_max=0 olarak alınır.
        """
        current_q = self.q_table[state][action]
        next_q_max = 0.0 if done else np.max(self.q_table[next_state])

        new_q = current_q + self.alpha * (reward + self.gamma * next_q_max - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """
        Epsilon’u her adımda biraz azaltarak ajan zamanla daha az rastgele, 
        daha çok öğrendiği değerlere göre karar vermeye başlar.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_qtable(self, filename="q_table.npy"):
        """
        Tüm Q tablosunu .npy formatında kaydedin.
        """
        np.save(filename, self.q_table)

    def load_qtable(self, filename="q_table.npy"):
        """
        Daha önce kaydedilmiş Q tablosunu diskteki .npy dosyasından yükleyin.
        """
        self.q_table = np.load(filename)
