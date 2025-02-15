import logging

import traceback

import asyncio
import concurrent.futures

from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException

from exchange_clients.base_exchange import BaseExchangeClient
from core.logging_setup import log 


class BinanceSpotManagerAsync(BaseExchangeClient):
    """
    python-binance senkron fonksiyonlarını asenkron sarmalayan sınıf.
    Market/Limit/Stop-Limit/OCO emirleri destekler.
    LOT_SIZE / MIN_NOTIONAL kontrolü yapar.
    Kodda log(...) ile traceback'leri de ekliyoruz.

    Bu sınıftaki fonksiyonlar:
    - Asenkron olarak hesap/bakiye bilgisi çekebilir (get_account_info, get_balance)
    - Market, Limit, Stop-Limit, OCO emirlerini gönderir (place_* fonksiyonları)
    - Emir iptali (cancel_order, cancel_oco_order) ve açık emir sorgusu (get_open_orders vs.)
    - Ek olarak get_my_trades (trade geçmişi) ve get_klines (kandil verisi) gibi fonksiyonlar
      strateji dosyasında maliyet hesaplama, sinyal üretme vs. için kullanılabilir.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client = Client(api_key, api_secret, testnet=testnet)
        if testnet:
            # Testnet için farklı base URL
            self.client.API_URL = 'https://testnet.binance.vision/api'

        # Senkron fonksiyonları çalıştırmak için bir executor
        self._executor = concurrent.futures.ThreadPoolExecutor()

        # Borsa exchange_info bilgilerini saklarız.
        self.exchange_info = {}
        self.symbol_filters_map = {}

        # Exchange info yükleyelim
        self._load_exchange_info()

    # ---------------------------------------------------------------------
    # Özel yardımıcılar (senkron fonksiyonları asenkron çağırmak)
    # ---------------------------------------------------------------------
    async def _call_sync(self, func, *args, **kwargs):
        """
        Senkron fonksiyonu, asyncio Executor üzerinden asenkron hâlde çalıştırır.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    def _load_exchange_info(self):
        """
        Borsadaki tüm semboller ve ilgili filtreleri (LOT_SIZE, PRICE_FILTER, vs.) yükler.
        """
        try:
            info = self.client.get_exchange_info()
            self.exchange_info = info
            if "symbols" in info:
                for s in info["symbols"]:
                    sym = s["symbol"]
                    filters = s["filters"]
                    self.symbol_filters_map[sym] = filters
        except BinanceAPIException as e:
            log(f"[_load_exchange_info] => {e}\n{traceback.format_exc()}", "error")
        except Exception as ex:
            log(f"[_load_exchange_info] => unknown => {ex}\n{traceback.format_exc()}", "error")

    def get_symbol_filters(self, symbol: str):
        """
        Sembole ait LOT_SIZE, PRICE_FILTER, vs. bulmak için.
        """
        sym = symbol.upper()
        fs = self.symbol_filters_map.get(sym)
        if not fs:
            raise ValueError(f"{sym} not found in exchange_info.")
        lot_size = next((f for f in fs if f["filterType"] == "LOT_SIZE"), None)
        price_filter = next((f for f in fs if f["filterType"] == "PRICE_FILTER"), None)
        notional_filter = next((f for f in fs if f["filterType"] in ["MIN_NOTIONAL", "NOTIONAL"]), None)
        return {
            "lot_size": lot_size,
            "price_filter": price_filter,
            "notional_filter": notional_filter
        }

    def _check_lot_notional(self, symbol: str, quantity: float, ref_price: float):
        """
        Binance LOT_SIZE & MIN_NOTIONAL kontrolleri.
        Pozisyon boyutu, sembol filtreleriyle uyumlu mu?
        """
        fs = self.get_symbol_filters(symbol)
        ls = fs["lot_size"]
        nn = fs["notional_filter"]
        if not ls:
            return

        step_size = float(ls["stepSize"])
        min_qty = float(ls["minQty"])
        max_qty = float(ls["maxQty"])

        min_notional = 0.0
        if nn:
            min_notional = float(nn["minNotional"])

        # Miktar kontrolü
        if quantity < min_qty or quantity > max_qty:
            raise ValueError(f"quantity {quantity} out of range ({min_qty}, {max_qty})")

        from decimal import Decimal
        dq = Decimal(str(quantity))
        ds = Decimal(str(step_size))
        rem = dq % ds
        if rem != 0:
            raise ValueError(f"quantity {quantity} not multiple of stepSize={step_size}")

        notional = quantity * ref_price
        if notional < min_notional:
            raise ValueError(f"notional {notional} < minNotional={min_notional}")

    # ---------------------------------------------------------------------
    # Temel Hesap/Bakiye Fonksiyonları
    # ---------------------------------------------------------------------
    async def get_account_info(self):
        """
        Spot hesap bilgilerini (bakiyeler dahil) döndürür.
        Strateji'de cüzdanda ne var, margin level vs. görmek için kullanılabilir.
        """
        try:
            account_info = await self._call_sync(self.client.get_account)
            return account_info
        except Exception as ex:
            logging.error(f"[get_account_info] => unknown => {ex}\n{traceback.format_exc()}")
            return {}

    async def get_balance(self, asset: str) -> float:
        """
        Tek bir asset'in (ör. 'USDT') 'free' bakiyesini döndürür.
        """
        free_amt = 0.0
        try:
            bal_info = await self._call_sync(self.client.get_asset_balance, asset=asset)
            free_amt = float(bal_info["free"])
        except (BinanceAPIException, BinanceOrderException) as e:
            log(f"[get_balance] => {e}\n{traceback.format_exc()}", "error")
        except Exception as ex:
            log(f"[get_balance] => unknown => {ex}\n{traceback.format_exc()}", "error")
        return free_amt

    # ---------------------------------------------------------------------
    # Trades / Kline
    # ---------------------------------------------------------------------
    async def get_my_trades(self, symbol: str, limit=500, from_id=None,
                            startTime=None, endTime=None):
        """
        Geçmiş trade'leri döndürür (alış/satış işlemleri).
        Örneğin, "BTCUSDT" için.
        StartTime, EndTime epoch milis cinsinden verilebilir.
        Limit, default 500 (binance max 1000).
        from_id parametresi ile belirtilen tradeId'den sonrakileri çekebilirsiniz.
        Bu, ortalama maliyet hesaplama veya trade geçmişi incelemede kullanılabilir.
        """
        try:
            trades = await self._call_sync(
                self.client.get_my_trades,
                symbol=symbol.upper(),
                limit=limit,
                fromId=from_id,
                startTime=startTime,
                endTime=endTime
            )
            return trades
        except Exception as e:
            log(f"[get_my_trades] {e}\n{traceback.format_exc()}", "error")
            return []

    async def get_klines(self, symbol: str, interval: str = Client.KLINE_INTERVAL_1MINUTE,
                         limit: int = 100, startTime=None, endTime=None):
        """
        Ham kline (candle) verisini döndürür.
        Strateji veri besleme / collector aşamasında kullanılabilir.
        interval: '1m', '5m', '1h', ...
        limit: max veri bar sayısı.
        startTime, endTime: milis cinsinden epoch.
        """
        try:
            data = await self._call_sync(
                self.client.get_klines,
                symbol=symbol.upper(),
                interval=interval,
                limit=limit,
                startTime=startTime,
                endTime=endTime
            )
            return data
        except Exception as e:
            log(f"[get_klines] {e}\n{traceback.format_exc()}", "error")
            return []

    # ---------------------------------------------------------------------
    # Emir Gönderme Fonksiyonları
    # ---------------------------------------------------------------------
    async def place_market_order(self, symbol: str, side: str, quantity: float):
        """
        Market order oluştur (BUY/SELL).
        """
        log(f"[place_market_order] => {symbol}, side={side}, qty={quantity}", "info")
        try:
            # Price öğrenip lot/notional check
            ticker = await self._call_sync(self.client.get_symbol_ticker, symbol=symbol.upper())
            px = float(ticker["price"])
            self._check_lot_notional(symbol, quantity, px)

            if side.upper() == "BUY":
                order = await self._call_sync(
                    self.client.order_market_buy,
                    symbol=symbol.upper(),
                    quantity=quantity
                )
            else:
                order = await self._call_sync(
                    self.client.order_market_sell,
                    symbol=symbol.upper(),
                    quantity=quantity
                )

            log(f"[place_market_order] => {order}", "info")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            log(f"[place_market_order] => {e}\n{traceback.format_exc()}", "error")
            return None
        except ValueError as ve:
            log(f"[place_market_order] => ParamCheck => {ve}\n{traceback.format_exc()}", "error")
            return None
        except Exception as ex:
            log(f"[place_market_order] => unknown => {ex}\n{traceback.format_exc()}", "error")
            return None

    async def place_limit_order(self, symbol: str, side: str, quantity: float,
                                limit_price: float, time_in_force="GTC"):
        """
        Limit emir.
        """
        log(f"[place_limit_order] => {symbol}, side={side}, qty={quantity}, limit={limit_price}", "info")
        try:
            self._check_lot_notional(symbol, quantity, limit_price)

            if side.upper() == "BUY":
                order = await self._call_sync(
                    self.client.order_limit_buy,
                    symbol=symbol.upper(),
                    quantity=quantity,
                    price=f"{limit_price:.8f}",
                    timeInForce=time_in_force
                )
            else:
                order = await self._call_sync(
                    self.client.order_limit_sell,
                    symbol=symbol.upper(),
                    quantity=quantity,
                    price=f"{limit_price:.8f}",
                    timeInForce=time_in_force
                )
            log(f"[place_limit_order] => {order}", "info")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            log(f"[place_limit_order] => {e}\n{traceback.format_exc()}", "error")
            return None
        except ValueError as ve:
            log(f"[place_limit_order] => ParamCheck => {ve}\n{traceback.format_exc()}", "error")
            return None
        except Exception as ex:
            log(f"[place_limit_order] => unknown => {ex}\n{traceback.format_exc()}", "error")
            return None

    async def place_stop_limit_order(self, symbol: str, side: str, quantity: float,
                                     stop_price: float, limit_price: float,
                                     time_in_force="GTC"):
        """
        Stop-limit emir (ör. stopPrice aşıldığında limit emir devreye girer).
        """
        log(f"[place_stop_limit_order] => {symbol}, side={side}, qty={quantity}, "
            f"stop={stop_price}, limit={limit_price}", "info")
        try:
            self._check_lot_notional(symbol, quantity, limit_price)

            if side.upper() == "BUY":
                order = await self._call_sync(
                    self.client.create_order,
                    symbol=symbol.upper(),
                    side=SIDE_BUY,
                    type=ORDER_TYPE_STOP_LOSS_LIMIT,
                    quantity=quantity,
                    price=f"{limit_price:.8f}",
                    stopPrice=f"{stop_price:.8f}",
                    timeInForce=time_in_force
                )
            else:
                order = await self._call_sync(
                    self.client.create_order,
                    symbol=symbol.upper(),
                    side=SIDE_SELL,
                    type=ORDER_TYPE_STOP_LOSS_LIMIT,
                    quantity=quantity,
                    price=f"{limit_price:.8f}",
                    stopPrice=f"{stop_price:.8f}",
                    timeInForce=time_in_force
                )
            log(f"[place_stop_limit_order] => {order}", "info")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            log(f"[place_stop_limit_order] => {e}\n{traceback.format_exc()}", "error")
            return None
        except ValueError as ve:
            log(f"[place_stop_limit_order] => ParamCheck => {ve}\n{traceback.format_exc()}", "error")
            return None
        except Exception as ex:
            log(f"[place_stop_limit_order] => unknown => {ex}\n{traceback.format_exc()}", "error")
            return None

    async def place_oco_order(self, symbol: str, side: str, quantity: float,
                              limit_price: float, stop_price: float,
                              stop_limit_price=None):
        """
        OCO (One Cancels Other) emir.
        Tek bir emirle hem limit hem de stop-limit koyabilirsiniz.
        """
        log(f"[place_oco_order] => {symbol}, side={side}, qty={quantity}, "
            f"limit={limit_price}, stop={stop_price}, stopLimit={stop_limit_price}", "info")
        try:
            self._check_lot_notional(symbol, quantity, limit_price)

            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "quantity": quantity,
                "price": f"{limit_price:.8f}",
                "stopPrice": f"{stop_price:.8f}",
                "stopLimitTimeInForce": "GTC"
            }
            if stop_limit_price:
                params["stopLimitPrice"] = f"{stop_limit_price:.8f}"

            oco_order = await self._call_sync(self.client.create_oco_order, **params)
            log(f"[place_oco_order] => {oco_order}", "info")
            return oco_order
        except (BinanceAPIException, BinanceOrderException) as e:
            log(f"[place_oco_order] => {e}\n{traceback.format_exc()}", "error")
            return None
        except ValueError as ve:
            log(f"[place_oco_order] => Param => {ve}\n{traceback.format_exc()}", "error")
            return None
        except Exception as ex:
            log(f"[place_oco_order] => unknown => {ex}\n{traceback.format_exc()}", "error")
            return None

    # ---------------------------------------------------------------------
    # Emir İptali & Açık Emir Sorgu
    # ---------------------------------------------------------------------
    async def cancel_order(self, symbol: str, order_id: int):
        """
        Tek bir open order'ı iptal etmek.
        """
        log(f"[cancel_order] => {symbol}, orderId={order_id}", "info")
        try:
            res = await self._call_sync(
                self.client.cancel_order,
                symbol=symbol.upper(),
                orderId=order_id
            )
            log(f"[cancel_order] => {res}", "info")
            return res
        except (BinanceAPIException, BinanceOrderException) as e:
            log(f"[cancel_order] => {e}\n{traceback.format_exc()}", "error")
            return None
        except Exception as ex:
            log(f"[cancel_order] => unknown => {ex}\n{traceback.format_exc()}", "error")
            return None

    async def cancel_oco_order(self, symbol: str, oco_order_id: int):
        """
        Belirli bir OCO order'ı iptal etmek.
        """
        log(f"[cancel_oco_order] => {symbol}, listId={oco_order_id}", "info")
        try:
            res = await self._call_sync(
                self.client.cancel_order_list,
                symbol=symbol.upper(),
                orderListId=oco_order_id
            )
            log(f"[cancel_oco_order] => {res}", "info")
            return res
        except (BinanceAPIException, BinanceOrderException) as e:
            log(f"[cancel_oco_order] => {e}\n{traceback.format_exc()}", "error")
            return None
        except Exception as ex:
            log(f"[cancel_oco_order] => unknown => {ex}\n{traceback.format_exc()}", "error")
            return None

    async def get_open_orders(self, symbol: str):
        """
        Açık emirleri (limit, stop vb.) getirir.
        """
        try:
            orders = await self._call_sync(
                self.client.get_open_orders,
                symbol=symbol.upper()
            )
            log(f"[get_open_orders] => {orders}", "info")
            return orders
        except (BinanceAPIException, BinanceOrderException) as e:
            log(f"[get_open_orders] => {e}\n{traceback.format_exc()}", "error")
            return []
        except Exception as ex:
            log(f"[get_open_orders] => unknown => {ex}\n{traceback.format_exc()}", "error")
            return []

    async def get_open_oco_orders(self, symbol: str):
        """
        Açık OCO emirleri listesini getirir. 
        Sembol eşleşirse, filtreleyip döndürür.
        """
        try:
            oco_list = await self._call_sync(
                self.client.get_open_oco_orders
            )
            log(f"[get_open_oco_orders] => {oco_list}", "info")

            filtered = []
            for oco in oco_list:
                if oco["symbol"] == symbol.upper():
                    filtered.append(oco)
            return filtered
        except (BinanceAPIException, BinanceOrderException) as e:
            log(f"[get_open_oco_orders] => {e}\n{traceback.format_exc()}", "error")
            return []
        except Exception as ex:
            log(f"[get_open_oco_orders] => unknown => {ex}\n{traceback.format_exc()}", "error")
            return []
    
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

    
    async def  get_trending_coins(self,list=20):
        X = 0.01
        tickers =  self.client.get_ticker()
        trending =  [ticker for ticker in tickers if ticker['symbol'].endswith('USDT') and float(ticker['priceChangePercent']) > X]
        return trending[:list]
