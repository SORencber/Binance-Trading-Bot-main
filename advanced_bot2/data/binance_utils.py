# file: core/binance_utils.py

import math

class SymbolRule:
    def __init__(self, min_qty, max_qty, step_size, min_notional, tick_size):
        self.min_qty = float(min_qty)
        self.max_qty = float(max_qty)
        self.step_size = float(step_size)
        self.min_notional = float(min_notional)
        self.tick_size = float(tick_size)

def truncate_to_step(value: float, step: float) -> float:
    """
    step_size'a göre value'yu aşağı yuvarlar. (örn. 0.000001 step)
    """
    return math.floor(value / step) * step

def check_lot_and_notional(symbol_rule: SymbolRule, quantity: float, price: float):
    """
    Emir parametrelerini Binance kurallarına göre doğrular.
    - quantity >= min_qty ve <= max_qty
    - (price * quantity) >= min_notional
    - quantity stepSize'a uygun
    - price tickSize'a uygun  (limit emirse)
    """

    # 1) quantity step kontrolü
    truncated_qty = truncate_to_step(quantity, symbol_rule.step_size)
    if truncated_qty <= 0:
        raise ValueError(f"Quantity {quantity} is too small (stepSize={symbol_rule.step_size}).")

    # 2) min_qty / max_qty
    if truncated_qty < symbol_rule.min_qty:
        raise ValueError(f"Quantity {truncated_qty} < minQty={symbol_rule.min_qty}")
    if truncated_qty > symbol_rule.max_qty:
        raise ValueError(f"Quantity {truncated_qty} > maxQty={symbol_rule.max_qty}")

    # 3) notional kontrol => price * qty >= min_notional
    notional = price * truncated_qty
    if notional < symbol_rule.min_notional:
        raise ValueError(f"Notional {notional:.4f} < minNotional={symbol_rule.min_notional}")

    # 4) Price tickSize (limit emirler için) -> benzer truncate_to_step
    # eğer Market emirse price'ı kontrol etmeyebilirsiniz. 
    # ama OCO vs. limit price varsa:
    # truncated_price = truncate_to_step(price, symbol_rule.tick_size)
    # if truncated_price <= 0:
    #    raise ValueError(...)

    # 5) Sorun yok => geri dön
    return truncated_qty
# file: setup_binance.py

async def load_binance_rules(ctx: SharedContext):
    """
    ExchangeInfo sonucundan her symbol için SymbolRule() objesi oluşturup,
    ctx.symbol_rules_map[symbol] = SymbolRule(...) dolduruyoruz.
    """
    exchange_info = await ctx.client_async.get_exchange_info()
    ctx.symbol_rules_map = {}

    for s in exchange_info["symbols"]:
        sym_name = s["symbol"]
        if sym_name not in ctx.config["symbols"]:
            continue
        # "filters" içinde LOT_SIZE, MIN_NOTIONAL, PRICE_FILTER vb. var
        min_qty = 0
        max_qty = 9999999
        step_size = 1.0
        min_notional = 10.0
        tick_size = 0.1
        for f in s["filters"]:
            if f["filterType"] == "LOT_SIZE":
                min_qty = float(f["minQty"])
                max_qty = float(f["maxQty"])
                step_size = float(f["stepSize"])
            elif f["filterType"] == "MIN_NOTIONAL":
                min_notional = float(f["minNotional"])
            elif f["filterType"] == "PRICE_FILTER":
                tick_size = float(f["tickSize"])
        ctx.symbol_rules_map[sym_name] = SymbolRule(min_qty, max_qty, step_size, min_notional, tick_size)

    print("Binance rules loaded for:", ctx.symbol_rules_map.keys())
