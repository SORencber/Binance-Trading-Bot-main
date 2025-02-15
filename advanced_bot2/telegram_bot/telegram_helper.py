# strategy/telegram_helper.py

import traceback
from core.context import SharedContext
from core.openai import openai_connect
import json
from exchange_clients.binance_spot_manager_async import BinanceSpotManagerAsync
from config.bot_config import BOT_CONFIG, BINANCE_API_KEY, BINANCE_API_SECRET

from core.logging_setup import log

import asyncio
from data_analytics.detect_regime import get_all_regimes,analyze_multi_tf_alignment,produce_realistic_signal
from trading_view.main_tv import generate_signals
from data.data_collector import update_data

from datetime import datetime
import time
import traceback
LAST_SUMMARY_TIME = 0  # Son summary gönderim zamanı (timestamp)
SUMMARY_INTERVAL = 60  # 30 dakika = 1800 sn

import pandas as pd

def calculate_timeframe_score(tf_data):
    """Tek bir timeframe için LONG/SHORT skorunu hesaplar"""
    print(tf_data)  # Check the data before parsing
    #tf_data = json.loads(tf_data)

    # Eğer tf_data bir string ise, JSON verisini parse et
    if isinstance(tf_data, str):
        tf_data = json.loads(tf_data)
    elif not isinstance(tf_data, dict):
        raise ValueError("Expected tf_data to be a dictionary or a valid JSON string.")
    
    long_score = 0
    short_score = 0
    
    # Trend ve Momentum
    trend_weights = {'bullish': 2, 'bearish': -2, 'sideways': 0, 'neutral': 0}
    
    # Safe access for trend key, with default value 'neutral'
    trend = tf_data.get('trend', 'neutral')  # Default to 'neutral' if 'trend' is missing
    long_score += trend_weights.get(trend, 0)
    short_score += trend_weights.get(trend, 0) * -1
    
    # Momentum
    momentum = tf_data.get('momentum', 'neutral')  # Default to 'neutral' if 'momentum' is missing
    long_score += 1.5 if momentum == 'bullish' else -1.5 if momentum == 'bearish' else 0
    short_score += 1.5 if momentum == 'bearish' else -1.5 if momentum == 'bullish' else 0
    
    # Göstergeler
    long_score += min(tf_data.get('rsi', 0)/70, 1)  # RSI 70'e normalize
    short_score += min((100 - tf_data.get('rsi', 0))/70, 1)
    
    macd_hist = tf_data.get('macd_hist', 0)
    long_score += macd_hist/abs(macd_hist) if macd_hist != 0 else 0
    short_score += -macd_hist/abs(macd_hist) if macd_hist != 0 else 0
    
    # Volatilite ve Momentum
    if tf_data.get('potential_breakout_note') and 'bullish' in tf_data.get('potential_breakout_note', ''):
   
        long_score += 2
    elif tf_data.get('potential_breakout_note') and 'bearish' in tf_data.get('potential_breakout_note', ''):
        short_score += 2
        
    # Zaman Dilimi Ağırlıkları
    timeframe_weights = {
        '5m': 0.7, 
        '15m': 0.8, 
        '30m': 0.9, 
        '1h': 1.0, 
        '4h': 1.2, 
        '1d': 1.5, 
        '1w': 2.0
    }
    
    # Zaman dilimi geçerli mi kontrol et
    # Check if 'timeframe' exists and is valid before calling the function
    if 'timeframe' not in tf_data or tf_data['timeframe'] not in timeframe_weights:
        raise ValueError(f"Invalid or missing timeframe in the input data: {tf_data}")

    timeframe = tf_data.get('timeframe')
    print(timeframe)
    if timeframe not in timeframe_weights:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    return {
        'long': long_score * timeframe_weights[timeframe],
        'short': short_score * timeframe_weights[timeframe]
    }


async def analyze_coins(ctx: SharedContext,list_long=20):
    # Initialize the best coin dictionaries and lists to store scores
    long_coins = []
    short_coins = []
    exchange_client = BinanceSpotManagerAsync(BINANCE_API_KEY, BINANCE_API_SECRET)

    trending_coins = await exchange_client.get_trending_coins(list=list_long)

    for coin in trending_coins:
        symbol = coin['symbol']
        print(symbol)
        await update_data(ctx=ctx, s=symbol)

        # Retrieve data for different timeframes
        df_5m = ctx.df_map.get(symbol, {}).get("5m", None)
        df_15m = ctx.df_map.get(symbol, {}).get("15m", None)
        df_30m = ctx.df_map.get(symbol, {}).get("30m", None)
        df_1h = ctx.df_map.get(symbol, {}).get("1h", None)
        df_4h = ctx.df_map.get(symbol, {}).get("4h", None)
        df_1d = ctx.df_map.get(symbol, {}).get("1d", None)
        df_1w = ctx.df_map.get(symbol, {}).get("1w", None)

        get_regime_info = get_all_regimes(symbol, df_5m=df_5m, df_15m=df_15m, df_30m=df_30m, df_1h=df_1h, df_4h=df_4h, df_1d=df_1d, df_1w=df_1w)
        
        total_long = 0
        total_short = 0

        for timeframe, data in get_regime_info.items():
            try:
                result = calculate_timeframe_score(data)
                print(f"Timeframe: {timeframe} - Long: {result['long']}, Short: {result['short']}")
                total_long += result['long']
                total_short += result['short']
            except ValueError as e:
                print(f"Error for {timeframe}: {e}")
                continue  # Skip any errors and continue processing

        # Add the coin and its scores to the respective lists
        long_coins.append({'coin': symbol, 'score': total_long})
        short_coins.append({'coin': symbol, 'score': total_short})

    # Sort coins by their scores in descending order
    top_long_coins = sorted(long_coins, key=lambda x: x['score'], reverse=True)[:10]
    top_short_coins = sorted(short_coins, key=lambda x: x['score'], reverse=True)[:10]

    # Return the top 10 coins for long and short positions
    return {
        'top_long_coins': top_long_coins,
        'top_short_coins': top_short_coins
    }


async def analyze_data(ctx:SharedContext, symbol: str):
 
        await update_data(ctx=ctx,s=symbol)

        print("ne oldu ?")

       # 3.3) Karar mantığı (multi-timeframe + RL vs.)
        df_main = ctx.df_map.get(symbol, {}).get("merged", None)
        print(df_main)
        if df_main is None or len(df_main) < 10:
            return  # Yeterli veri yok

        row_main = df_main.iloc[-1]

          #df_1m = ctx.df_map.get(symbol, {}).get("1m", None)
        df_5m = ctx.df_map.get(symbol, {}).get("5m", None)
        df_15m = ctx.df_map.get(symbol, {}).get("15m", None)
        df_30m = ctx.df_map.get(symbol, {}).get("30m", None)
        df_1h = ctx.df_map.get(symbol, {}).get("1h", None)
        df_4h = ctx.df_map.get(symbol, {}).get("4h", None)
        df_1d = ctx.df_map.get(symbol, {}).get("1d", None)
        df_1w = ctx.df_map.get(symbol, {}).get("1w", None)

        force_summary=False
        command_source=ctx.config["command_source"]
        if command_source=="telegram": 
            force_summary=True
            ctx.config["command_source"]="app"
            
        await send_telegram_messages(df_5m=df_5m,df_15m=df_15m, df_30m=df_30m, df_1h=df_1h, df_4h=df_4h, df_1d=df_1d,df_1w=df_1w, ctx=ctx, row_main=row_main, symbol=symbol, force_summary=force_summary)


async def decide_trade_mtf_with_pattern(df_5m,df_15m,df_30m, df_1h, df_4h,df_1d, df_1w,
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
        get_regime_info = get_all_regimes(symbol=symbol,df_5m=df_5m,df_15m=df_15m,df_30m=df_30m,df_1h=df_1h,df_4h=df_4h,df_1d=df_1d,df_1w=df_1w)
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

async def format_pattern_results(mtf_dict: dict,price) -> str:
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

def find_close_patterns(results_dict: dict, current_price, lower_threshold=5, upper_threshold=10):
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

async def send_telegram_messages(
                                     df_5m,
                                     df_15m,
                                     df_30m,df_1h,
                    df_4h,
                    df_1d,df_1w, ctx: SharedContext, row_main, symbol, force_summary=False):
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
        price = row_main.get("Close_1m", 0.0)    # Son 30 dak. kapanış

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
            resistance_5m=row_main.get("Resistance_5m",0.0)
            support_5m = row_main.get("Support_5m",0.0)
    
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
            mtf_decision = await decide_trade_mtf_with_pattern(
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
                f"<b>i̇ndikatör  destek-direnç (5 dakika):</b> {support_5m:.4f}, {resistance_5m:.4f}\n"
                f"----------------\n"
               f"<b>i̇ndikatör  destek-direnç (15 dakika):</b> {support_15m:.4f}, {resistance_15m:.4f}\n"
                f"----------------\n"
              
                f"<b>i̇ndikatör  destek-direnç (30 dakika):</b> {support_30m:.4f}, {resistance_30m:.4f}\n"
                f"----------------\n"
                f"<b>i̇ndikatör  destek-direnç (1 saat):</b> {support_1h:.4f}, {resistance_1h:.4f}\n"
                f"----------------\n"
                f"<b>i̇ndikatör  destek-direnç (4 saat):</b> {support_4h:.4f}, {resistance_4h:.4f}\n"
                f"----------------\n"
                f"<b>i̇ndikatör  destek-direnç (1 günlük):</b> {support_1d:.4f}, {resistance_1d:.4f}\n"
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
                
                
                txt_report = await format_pattern_results(results_dict,price)
                await telegram_app.bot.send_message(chat_id=chat_id, text=txt_report, parse_mode="HTML")
                await asyncio.sleep(5)
                time_frame_infos=mtf_decision["get_regime_info"]
                # Open AI baglantisi ve yorumlari alinir.
                time_frame_infos_str = json.dumps(time_frame_infos, ensure_ascii=False)  # JSON string formatına çevir  
                prompt = f"Verilen tüm analiz ve pattern sonuclarini birlikte degerlendir ve gercekci bir Short ve Long onerisi yap. {time_frame_infos_str} {txt_report}{txt_summary}"              
                open_ai= ctx.config.get("open_ai", False)
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
                #     await telegram_app.bot.send_message(chat_id=chat_id, text=txt_report, parse_mode="HTML")
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
                summ_patterns=await summarize_patterns(results_dict, price)
                return mtf_decision,summ_patterns

            except Exception as e:
                log(f"[Telegram send ilk error ): => {e}\n{traceback.format_exc()}", "error")

        # Örnek olarak current_price = close_30m kabul ediyoruz
        #print(price)
        # Pattern’lerden puanlı olanları bul
        close_pattern_list = find_close_patterns(results_dict, price)     
       
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

async def summarize_patterns(results_dict, current_price):
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

