# strategy/advanced_ensemble_strategy.py

import numpy as np
from binance.exceptions import BinanceAPIException
from strategy.base import IStrategy
from core.context import SharedContext
from core.logging_setup import log
from strategy.paper_order import PaperOrderManager
from core.order_manager import RealOrderManager
from core.utils import check_lot_and_notional

class MultiSignalStrategy(IStrategy):
    def name(self) -> str:
        return "MultiSignalStrategy"

    async def analyze_data(self, ctx: SharedContext):
        """
        (Opsiyonel) Offline veri analizi burada yapılabilir.
        """
        pass

    async def on_price_update(self, ctx: SharedContext, symbol: str, price: float):
        """
        1) "Çoklu Sinyal" (XGBoost, LSTM, RL) => final_action (0=SELL,1=BUY,2=HOLD).
        2) Kademeli risk yönetimi => 
           - use_oco_kademeli=True => OCO emirler
           - use_oco_kademeli=False => bot tabanlı partial/trailing
        """
        log(f"on_price_update => Entered, symbol={symbol}, price={price}", "debug")
        
        st = ctx.symbol_map[symbol]
        df = ctx.df_map.get(symbol, {}).get("1m", None)
        if df is None or len(df) < 2:
            return
        #print(df)
        # 1) Çoklu sinyal => xgb, lstm, rl => oylama => final_action
        final_action = self.calc_final_action(ctx, df)
        # (final_action= 0=>SELL, 1=>BUY, 2=>HOLD)

        # 2) "Tam" al/sat => 
        if not st.has_position and final_action == 1:
            await self.do_buy(ctx, symbol, price)
            return
        if st.has_position and final_action == 0:
            await self.do_sell(ctx, symbol, price)
            return

        # 3) EĞER OCO => kademeli emirler borsada
        #    EĞER Bot => handle_risk_management => partial/trailing
        if st.has_position:
            use_oco = ctx.config.get("use_oco_kademeli", False)
            if not use_oco:
                # Bot tabanlı partial/trailing
                await self.handle_risk_management(ctx, symbol, price)
            # else => OCO tabanlı => sabit emir, trailing yoksa pass

    # def calc_final_action(self, ctx: SharedContext, df) -> int:
    #     """
    #     Örneğin son bar verisinden RSI, ADX, sentiment vs. çekip
    #     XGB, LSTM, RL agent'ları çağırıp => basit oylama
    #     """
    #     # En son bar
    #     row = df.iloc[-1]
    #     numeric_cols = []
    #     for col in row.index:
    #             val = row[col]
    #             if isinstance(val, (int, float, np.number)):
    #                 numeric_cols.append(col)
            
    #         # 2) Gözlem vektörü => shape (1, N)
    #     observation = row[numeric_cols].values.astype(np.float32).reshape(1, -1)
    #     rsi = row.get("rsi", 50.0)
    #     adx = row.get("adx", 20.0)
    #     sentiment = row.get("sentiment", 0.0)
    #     onchain = row.get("onchain", 0.0)
    #     #print(observation)
    #     # XGBoost => 0=SELL, 1=BUY
    #     xgb_signal = 0
    #     if getattr(ctx, "xgboost_agent", None) is not None:
    #         X = observation
    #         xgb_signal = ctx.xgboost_agent.predict_signal(X)  # 0 veya 1

    #     # LSTM => 0=SELL, 1=BUY
    #     lstm_signal = 0
    #     if getattr(ctx, "lstm_agent", None) is not None:
    #         # L shape => (1,1,4)
    #         L = observation
    #         lstm_signal = ctx.lstm_agent.predict_signal(L)

    #     # RL => 0=SELL, 1=BUY, 2=HOLD
    #     rl_signal = 2
    #     if getattr(ctx, "rl_agent", None) is not None:
    #         # Eğer eğitimde PriceEnv, "tüm numeric kolonları" gözlem olarak kullandıysa:
    #         # row birçok kolon içeriyor olabilir. Onları float32 array'e çevirin.
            
    #         # 1) Tüm numeric kolonların listesi:
    #         #    (Bu örnekte, df her adımda tüm numeric kolonlara sahip. 
    #         #     Tek bir satır 'row' Series nesnesi => numeric değerleri ayıklıyoruz.)
           
    #         # 3) RL agent ile tahmin
    #         rl_signal = ctx.rl_agent.predict_action(observation)
    #     # Basit oylama
    #     signals = []

    #     # xgb => 0 => -1, 1 => +1
    #     signals.append(-1 if xgb_signal == 0 else +1)

    #     # lstm => 0 => -1, 1 => +1
    #     signals.append(-1 if lstm_signal == 0 else +1)

    #     # rl => 0=>-1,1=>+1,2=>0
    #     if rl_signal == 0:
    #         signals.append(-1)
    #     elif rl_signal == 1:
    #         signals.append(+1)
    #     else:
    #         signals.append(0)

    #     total_score = sum(signals)  # -3..+3
    #     final_action = 2  # hold
    #     if total_score >= 1:
    #         final_action = 1  # BUY
    #     elif total_score <= -1:
    #         final_action = 0  # SELL
    #     log(f"[on_price_update] xgb={xgb_signal}, lstm={lstm_signal}, rl={rl_signal}, final_action={final_action}", "info")

    #     return final_action
    
    def calc_final_action(self, ctx: SharedContext, df) -> int:
        """
        0 => SELL, 1 => BUY, 2 => HOLD
        """
          # En son bar
        row = df.iloc[-1]
      

        # 1) Piyasa Durumu => Trend mi Range mi?
        #    (Örn: 1h adx + 4h RSI)
        # => "TREND_UP", "TREND_DOWN", "RANGE", "BREAKOUT", ...
        scenario = self.detect_market_scenario(ctx, row)

        # 2) XGBoost/LSTM/RL => raw_action => -1(sell), +1(buy), 0(hold)
        raw_ensemble = self.ensemble_signals(ctx, row)

        # 3) final logic => eger scenario="range" => reversion => ...
        #    eger scenario="trend_up" => momentum => ...
        final_action = self.combine_scenario_with_ensemble(scenario, raw_ensemble, row)
        #print(final_action)
        return final_action
 
    def ensemble_signals(self, ctx: SharedContext, row) -> int:
        """
        xgb=0 => -1,1=>+1
        lstm=0 => -1,1=>+1
        rl => 0=>-1,1=>+1,2=>0
        => sum => total_score
        => -3..+3
        """
        numeric_cols = []
        for col in row.index:
                val = row[col]
                if isinstance(val, (int, float, np.number)):
                    numeric_cols.append(col)
               # 2) Gözlem vektörü => shape (1, N)
        observation = row[numeric_cols].values.astype(np.float32).reshape(1, -1)
        #rsi = row.get("rsi", 50.0)
       # adx = row.get("adx", 20.0)
        #sentiment = row.get("sentiment", 0.0)
        #onchain = row.get("onchain", 0.0)
        fgi = row.get("Fear_Greed_Index", 0.5)  # 0..1 arası
        news = row.get("News_Headlines", 0.0)   # -1..+1
        fundamentals_boost = 0
        price_rsi_4h = row.get("RSI_4h", 50.0)
        oi_rsi_4h    = row.get("4h_OI_RSI", 50.0)
        oi_roc_4h    = row.get("4h_OI_ROC", 0.0)
        oi_up_4h     = row.get("4h_OI_BBUp", 0.0)
        current_oi   = row.get("Open_Interest", 0.0)

        # Basit mantık:
        if oi_rsi_4h > 70:
            # OI çok yüksek => potansiyel büyük hareket
            fundamentals_boost += 1
        if price_rsi_4h > 70:
            # Fiyat RSI da overbought => belki short sinyali
            fundamentals_boost -= 1
        if oi_roc_4h > 10:
            # OI hızla artıyor => momentum
            fundamentals_boost += 1
        if fgi < 0.3:
            fundamentals_boost += 1  # korku => dip al
        elif fgi > 0.7:
            fundamentals_boost -= 1  # açgözlülük => risk

        if news < -0.2:
            fundamentals_boost -= 1  # kötü haber
        elif news > 0.2:
            fundamentals_boost += 1  # pozitif haber
        #print(observation)
        # XGBoost => 0=SELL, 1=BUY
        xgb_signal = 0
        if getattr(ctx, "xgboost_agent", None) is not None:
            X = observation
            xgb_signal = ctx.xgboost_agent.predict_signal(X)  # 0 veya 1

        # LSTM => 0=SELL, 1=BUY
        lstm_signal = 0
        if getattr(ctx, "lstm_agent", None) is not None:
            # L shape => (1,1,4)
            L = observation
            lstm_signal = ctx.lstm_agent.predict_signal(L)

        # RL => 0=SELL, 1=BUY, 2=HOLD
        rl_signal = 2
        if getattr(ctx, "rl_agent", None) is not None:
            # Eğer eğitimde PriceEnv, "tüm numeric kolonları" gözlem olarak kullandıysa:
            # row birçok kolon içeriyor olabilir. Onları float32 array'e çevirin.
            
            # 1) Tüm numeric kolonların listesi:
            #    (Bu örnekte, df her adımda tüm numeric kolonlara sahip. 
            #     Tek bir satır 'row' Series nesnesi => numeric değerleri ayıklıyoruz.)
           
            # 3) RL agent ile tahmin
            rl_signal = ctx.rl_agent.predict_action(observation)
        # Basit oylama
        signals = []

        # xgb => 0 => -1, 1 => +1
        signals.append(-1 if xgb_signal == 0 else +1)

        # lstm => 0 => -1, 1 => +1
        signals.append(-1 if lstm_signal == 0 else +1)

        # rl => 0=>-1,1=>+1,2=>0
        if rl_signal == 0:
            signals.append(-1)
        elif rl_signal == 1:
            signals.append(+1)
        else:
            signals.append(0)
        signals.append(fundamentals_boost)

        total_score = sum(signals)  # -3..+3
        
        final_action = 2  # hold
        if total_score >= 1:
            final_action = 1  # BUY
        elif total_score <= -1:
            final_action = 0  # SELL
        
        log(f"[on_price_update] xgb={xgb_signal}, lstm={lstm_signal}, rl={rl_signal}, final_action={final_action}", "info")
        return total_score


    def detect_market_scenario(self, ctx: SharedContext, row) -> str:
        """
        Örnek:
         - 1h_ADX_14 > 25 => trend
         - eger 4h_RSI_14>55 => up
         - yoksa 1h_BB_width < X => breakout
         - vs...
        """
        adx_1h = row.get("ADX_1h", 20.0)
        rsi_4h = row.get("RSI_4h", 50.0)  # eğer 4h verisi asof merge ile df'ye eklendiyse

        if adx_1h>25:
            if rsi_4h>55:
                return "TREND_UP"
            elif rsi_4h<45:
                return "TREND_DOWN"
            else:
                return "TREND_FLAT"
        else:
            # belki Boll width
            bb_up_1h = row.get("BBUp_1h", 999999)
            bb_low_1h= row.get("BBLow_1h", 0)
            mid = (bb_up_1h + bb_low_1h)/2
            band_width= bb_up_1h - bb_low_1h
            if band_width/mid < 0.05:
                return "BREAKOUT_SOON"
            else:
                return "RANGE"

    def generate_signals(self, ctx: SharedContext,df) -> int:
        """
        df içinde, en azından:
        - 4h_ADX_14, 4h_RSI_14
        - 1h_ADX_14, 1h_RSI_14, 1h_BBUp, ...
        - 15m_RSI_14, 15m_STOCHK_14, ...
        vs. olmalı.
        Bu fonksiyon, df'e 'Signal' sütunu (0=hold,1=buy,-1=sell) ekleyebilir.
        """
        df = df.copy()

        # Örnek kural: Makro (4H) RSI > 50 => up bias
        # Orta (1H) ADX>25 => trend
        # Kısa (15m) RSI<30 => buy
        condition_buy = (
            (df["RSI_4h"]>50) &
            (df["ADX_1h"]>25) &
            (df["RSI_15m"]<30)
        )
        condition_sell = (
            (df["RSI_4h"]<50) &
            (df["ADX_1h"]>25) &
            (df["RSI_15m"]>70)
        )
        df["Signal"] = 0
        df.loc[condition_buy,  "Signal"] = 1
        df.loc[condition_sell, "Signal"] = -1
        
        return df

    def combine_scenario_with_ensemble(self, scenario, ensemble_raw, row) -> int:
        """
        Örnek:
          scenario="TREND_UP" => momentum => eger ensemble=-1 => belki hold? (zayıflat)
        """#        #
        print(scenario)
        print(ensemble_raw)

        if scenario=="TREND_UP":
            if ensemble_raw>0:
                return 1  # BUY
            else:
                # belki hold => 2
                return 2
        elif scenario=="TREND_DOWN":
            if ensemble_raw<0:
                return 0  # SELL
            else:
                return 2
        elif scenario=="RANGE":
            # eger ensemble=+1 => Buy alt bant
            # eger ensemble=-1 => Sell üst bant
            return ensemble_raw if ensemble_raw!=-999 else 2
        elif scenario=="BREAKOUT_SOON":
            # eger ensemble=+1 => breakout buy
            return ensemble_raw
        else:
            # default
            return ensemble_raw
  
  
  
  
  
    async def do_buy(self, ctx: SharedContext, symbol: str, price: float):
        st = ctx.symbol_map[symbol]
        if st.has_position:
            return
        config = ctx.config

        async with ctx.lock:
            try:
                cost = min(config["trade_amount_usdt"], ctx.paper_positions["USDT"])
                eff = price*(1+ config.get("slippage_rate",0.001))
                qty = cost/eff

                # Market BUY
                if config.get("paper_trading",True):
                    pm = PaperOrderManager(symbol, ctx.paper_positions)
                    await pm.place_market_order("BUY", qty, eff)
                else:
                    await check_lot_and_notional(ctx.client_async, symbol, qty, eff)
                    rm = RealOrderManager(ctx.client_async, symbol)
                    await rm.place_market_order("BUY", qty)

                # Pozisyon update
                st.has_position = True
                st.quantity = qty
                st.entry_price = eff
                st.highest_price = eff
                st.pnl = 0.0
                st.consecutive_losses = 0

                log(f"[MultiSignalStrategy] BUY => {symbol}, qty={qty:.4f}, px={eff:.2f}", "info")

                # eğer use_oco => Kademeli OCO emirleri (2 kademe vs.)
                use_oco = config.get("use_oco_kademeli",False)
                if use_oco and not config.get("paper_trading",True):
                    rm= RealOrderManager(ctx.client_async, symbol)
                    stop_px= eff*(1- config.get("stop_loss_pct",0.03))

                    # Kademe1 => yarı qty => +2% limit
                    half= qty*0.5
                    tp1= eff*(1+0.02)
                    await rm.place_oco_order("SELL", half, tp1, stop_px)

                    # Kademe2 => kalan => +4% limit
                    half2= qty - half
                    tp2= eff*(1+0.04)
                    await rm.place_oco_order("SELL", half2, tp2, stop_px)

                    log(f"[OCO Kademeli] => kad1=+2%, kad2=+4%, stop={stop_px:.2f}", "info")
                else:
                    # Bot tabanlı partial => init
                    st.partial_tp_done = [False,False]
                    st.stop_loss = eff*(1- config.get("stop_loss_pct",0.03))

            except (ValueError, BinanceAPIException) as e:
                log(f"[MultiSignalStrategy do_buy] => {e}", "error")
            except Exception as ex:
                log(f"[MultiSignalStrategy do_buy] => {ex}", "error")

    async def do_sell(self, ctx: SharedContext, symbol: str, price: float):
        """
        Tam kapat => SELL
        """
        st= ctx.symbol_map[symbol]
        if not st.has_position:
            return
        config= ctx.config
        qty= st.quantity
        local_pnl= (price - st.entry_price)/ st.entry_price

        async with ctx.lock:
            try:
                if config.get("paper_trading", True):
                    pm= PaperOrderManager(symbol, ctx.paper_positions)
                    await pm.place_market_order("SELL", qty, price)
                else:
                    rm= RealOrderManager(ctx.client_async, symbol)
                    await rm.place_market_order("SELL", qty)

                st.pnl += local_pnl
                st.reset_position()
                log(f"[MultiSignalStrategy] SELL => {symbol}, PnL={local_pnl:.2%}", "info")
            except Exception as e:
                log(f"[MultiSignalStrategy do_sell] => {e}", "error")

    async def handle_risk_management(self, ctx: SharedContext, symbol: str, price: float):
        """
        Bot tabanlı partial/trailing risk yönetimi.
        + Ek olarak mikro OI & Funding & OrderBook analizleri.
        """
        st = ctx.symbol_map[symbol]
        config = ctx.config

        # 1) Trailing stop (aynı önceki mantık)
        if price > st.highest_price:
            st.highest_price = price
        trailing_stop = st.highest_price * (1 - config.get("trailing_pct", 0.03))
        if st.highest_price > 0 and price <= trailing_stop:
            log("[MultiSignalStrategy-bot] trailing => SELL all", "warning")
            await self.do_sell(ctx, symbol, price)
            return

        # 2) Kademeli => partial TP (aynı önceki mantık)
        if hasattr(st, "partial_tp_done"):
            partial_levels = config.get("partial_tp_levels", [0.02, 0.04])
            for i, level in enumerate(partial_levels):
                if i >= len(st.partial_tp_done):
                    continue
                if not st.partial_tp_done[i]:
                    tp_price = st.entry_price * (1 + level)
                    if price >= tp_price:
                        part_qty = st.quantity * config.get("partial_tp_ratio", 0.5)
                        await self.do_partial_sell(ctx, symbol, price, part_qty)
                        st.partial_tp_done[i] = True

        # 3) StopLoss
        if st.stop_loss > 0 and price < st.stop_loss:
            log("[MultiSignalStrategy-bot] STOP => SELL", "warning")
            await self.do_sell(ctx, symbol, price)
            return

        # =======================
        # 4) EK MİKRO ANALİZ (OI, Funding, OrderBook)
        # =======================
        df_1m = ctx.df_map[symbol].get("1m", None)
        if df_1m is None or len(df_1m) < 1:
            return

        row = df_1m.iloc[-1]

        # OI ROC, Funding Rate, vs.
        oi_roc_1m = row.get("OI_ROC_1m", 0.0)      # 1 dakikada OI değişimi yüzdesi
        funding = row.get("fundingRate", 0.0)      # anlık funding rate (örn. 0.0006 = 0.06%)
        total_bids = row.get("OrderBook_BidVol", 0.0)  # order book bid toplam hacmi
        total_asks = row.get("OrderBook_AskVol", 0.0)  # order book ask toplam hacmi

        # (A) Funding > 0.01 and OI ROC > 10 => aşırı long baskısı + OI artışı => momentum
        # Örnek: trailing stop'u hafif genişletme veya partial kâr alma eşiğini öteleme
        if funding > 0.01 and oi_roc_1m > 10:
            # Bu tamamen örnek. Burada 'st.stop_loss'u %1 aşağı çekebilirsiniz vs.
            old_stop = st.stop_loss
            new_stop = old_stop * 0.99  # örn. stop'u biraz daha geniş tut
            st.stop_loss = new_stop
            log(f"[RISK] funding>0.01 & OI_ROC>10 => momentum up, stop_loss widened from {old_stop:.2f} to {new_stop:.2f}", "info")

        # (B) Aşırı Sell Wall => total_asks >> total_bids (ör. ratio>2.0)
        ratio = (total_asks + 1e-9) / (total_bids + 1e-9)
        if ratio > 2.0:
            # Satıcılar baskın => belki partial SELL
            part_qty = st.quantity * 0.25
            if part_qty > 0:
                log("[RISK] Sell wall in OrderBook => partial SELL 25%", "warning")
                await self.do_partial_sell(ctx, symbol, price, part_qty)

        # (C) OI ROC < -10 => OI hızla düşüyor => momentum kaybı => tam exit
        if oi_roc_1m < -10:
            log("[RISK] OI dropping fast => SELL all position", "warning")
            await self.do_sell(ctx, symbol, price)

    async def do_partial_sell(self, ctx: SharedContext, symbol: str, price: float, part_qty: float):
            st= ctx.symbol_map[symbol]
            if part_qty> st.quantity:
                part_qty= st.quantity

            change= (price- st.entry_price)/ st.entry_price
            ratio= part_qty/ st.quantity if st.quantity>0 else 1.0
            partial_pnl= change* ratio

            config= ctx.config
            async with ctx.lock:
                try:
                    if config.get("paper_trading", True):
                        pm= PaperOrderManager(symbol, ctx.paper_positions)
                        await pm.place_market_order("SELL", part_qty, price)
                    else:
                        rm= RealOrderManager(ctx.client_async, symbol)
                        await rm.place_market_order("SELL", part_qty)

                    st.pnl+= partial_pnl
                    st.quantity-= part_qty
                    log(f"[MultiSignalStrategy-bot] partial SELL => {part_qty}, px={price}, PNL~{partial_pnl:.2%}", "info")

                    if st.quantity<=0:
                        st.reset_position()
                except Exception as e:
                    log(f"[MultiSignalStrategy do_partial_sell] => {e}", "error")
