# strategy/base.py

from abc import ABC, abstractmethod
from core.context import SharedContext

class IStrategy(ABC):
    @abstractmethod
    def name(self)->str:
        pass

    @abstractmethod
    async def analyze_data(self, ctx: SharedContext):
        pass

    @abstractmethod
    async def on_price_update(self, ctx: SharedContext, price: float):
        pass
