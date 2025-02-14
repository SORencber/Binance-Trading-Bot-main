# core/context.py

import asyncio
from typing import Dict
from core.logging_setup import log

class SymbolState:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.has_position = False
        self.quantity = 0.0
        self.entry_price = 0.0
        self.net_pnl = 0.0
        self.highest_price = 0.0
        self.panic_count = 0
        self.panic_mode = False
        self.last_sell_time = None
        self.reentry_count = 0
        self.partial_done = []
        self.partial_tp_done = []
        self.stop_loss = 0.0

    def reset_position(self):
        self.has_position = False
        self.quantity = 0.0
        self.entry_price = 0.0
        self.net_pnl = 0.0
        self.highest_price = 0.0
        self.panic_count = 0
        self.panic_mode = False
        self.last_sell_time = None
        self.reentry_count = 0
        self.partial_done = []
        self.partial_tp_done = []
        self.stop_loss = 0.0

class SharedContext:
    def __init__(self, config: dict):
        self.config = config
        #print(ctx)
        # Kullanıcı verilerini işlemek için kuyruk
        self.user_data_queue = asyncio.Queue()

        # Binance asenkron client, socket manager
        self.client_async = None
        self.bsm = None

        # RL agent vb. modelleri sonradan yükleyebilirsiniz
        self.rl_agent = None

        # Sembollerin durumu (SymbolState)
        self.symbol_map: Dict[str, SymbolState] = {}
        for s in self.config["symbols"]:
            print("context........",s)
            self.symbol_map[s] = SymbolState(s)

        # Fiyatların asenkron kuyruğu (her sembol için)
        self.price_queues = {}

        # DataFrame saklamak (örn. 1m, 5m, 1h verileri)
        self.df_map = {}

        # Asenkron kilit
        self.lock = asyncio.Lock()

        # Paper trading için basit cüzdan örneği
        self.paper_positions = {
            "USDT": 1000.0
        }

        # Botun durmasını istediğimizde True yapabiliriz
        self.stop_requested = False

        # Rejim bazlı parametreler
        self.param_for_regime = {
            "TREND": {
                "stop_atr_mult": 3.0,
                "partial_levels": [0.05, 0.1, 0.15],
                "partial_ratio": 0.3
            },
            "RANGE": {
                "stop_atr_mult": 1.5,
                "partial_levels": [0.02, 0.04, 0.06],
                "partial_ratio": 0.3
            },
            "DEFAULT": {
                "stop_atr_mult": 2.0,
                "partial_levels": [0.03, 0.06, 0.1],
                "partial_ratio": 0.3
            }
        }
