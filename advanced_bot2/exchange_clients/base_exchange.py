# exchange_clients/base_exchange.py
import abc

class BaseExchangeClient(abc.ABC):
    """
    Soyut borsa arayüzü (async).
    Tüm metodlar asenkron => 'async def ...'
    """

    @abc.abstractmethod
    async def get_balance(self, asset: str) -> float:
        pass

    @abc.abstractmethod
    async def place_market_order(self, symbol: str, side: str, quantity: float):
        pass

    @abc.abstractmethod
    async def place_limit_order(self, symbol: str, side: str, quantity: float,
                                limit_price: float, time_in_force="GTC"):
        pass

    @abc.abstractmethod
    async def place_stop_limit_order(self, symbol: str, side: str, quantity: float,
                                     stop_price: float, limit_price: float,
                                     time_in_force="GTC"):
        pass

    @abc.abstractmethod
    async def place_oco_order(self, symbol: str, side: str, quantity: float,
                              limit_price: float, stop_price: float,
                              stop_limit_price=None):
        pass

    @abc.abstractmethod
    async def cancel_order(self, symbol: str, order_id: int):
        pass

    @abc.abstractmethod
    async def cancel_oco_order(self, symbol: str, oco_order_id: int):
        pass

    @abc.abstractmethod
    async def get_open_orders(self, symbol: str):
        pass

    @abc.abstractmethod
    async def get_open_oco_orders(self, symbol: str):
        pass
