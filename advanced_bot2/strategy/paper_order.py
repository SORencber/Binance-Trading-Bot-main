# strategy/paper_order.py
from typing import Dict
from core.logging_setup import log

class PaperOrderManager:
    def __init__(self, symbol: str, positions: Dict[str,float]):
        self.symbol= symbol
        self.positions= positions
        self.base= symbol[:-4]

    async def place_market_order(self, side: str, quantity: float, price: float):
        if side.upper()=="BUY":
            cost= quantity* price
            if self.positions["USDT"]< cost:
                log(f"[PaperOrder] Not enough USDT => need {cost}, have {self.positions['USDT']}", "warning")
                return None
            self.positions["USDT"]-= cost
            self.positions[self.base]+= quantity
            log(f"[PaperOrder] BUY => {self.symbol}, qty={quantity:.6f}, px={price:.2f}", "info")
        else:
            if self.positions[self.base]< quantity:
                log(f"[PaperOrder] Not enough {self.base} => need {quantity}, have {self.positions[self.base]}", "warning")
                return None
            self.positions[self.base]-= quantity
            self.positions["USDT"]+= quantity* price
            log(f"[PaperOrder] SELL => {self.symbol}, qty={quantity:.6f}, px={price:.2f}", "info")
        return {"status":"FILLED","side":side}

    async def place_oco_sell(self, quantity: float, tp_price: float, sl_price: float):
        # fill => user_data logic
        log(f"[PaperOrder] OCO SELL => {self.symbol}, qty={quantity}, TP={tp_price}, SL={sl_price}", "info")
        return {"oco":"paper","tp":tp_price,"sl":sl_price}
