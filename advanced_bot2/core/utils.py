# core/utils.py

import math
from decimal import Decimal, ROUND_DOWN
from binance import AsyncClient
from core.logging_setup import log

def adjust_quantity(qty: float, step_size: float)->float:
    """
    Miktarı step_size'in tam katı olacak şekilde 'aşağı' yuvarlar.
    Örneğin qty=0.01507, step=0.0001 => 0.0150
    8 ondalık basamağa kadar formatlanır.
    """
    if step_size <= 0:
        return float(f"{qty:.8f}")
    floored = math.floor(qty / step_size) * step_size
    return float(f"{floored:.8f}")

def format_price(price: float, tick_size: float)->float:
    """
    Fiyatı tick_size'a göre aşağı yuvarlar. 
    Örneğin 123.4567 => tick=0.01 => 123.45 
    """
    try:
        ts = Decimal(str(tick_size))
        pr = Decimal(str(price))
        # Floor
        formatted = (pr // ts)* ts
        # quantize ile son hâli
        return float(formatted.quantize(ts, rounding=ROUND_DOWN))
    except Exception as e:
        log(f"[format_price] Hata => {e}", "warning")
        return price

async def check_lot_and_notional(
    client_async: AsyncClient,
    symbol: str,
    qty: float,
    price: float,
    auto_adjust=False
):
    """
    LOT_SIZE ve NOTIONAL filtrelerini kontrol eder.
    Eğer auto_adjust=True ise, qty'yi otomatik olarak step_size'a uygun hale getirmeyi dener.
    Yine de minQty, minNotional vb. filtrelere takılırsa ValueError fırlatır.
    
    Örnek kullanım:
        try:
            await check_lot_and_notional(client, "BTCUSDT", 0.01507, 20000.0, auto_adjust=True)
        except ValueError as ve:
            # Hata => log veya skip
    """

    info = await client_async.get_exchange_info()
    s_info = next(s for s in info["symbols"] if s["symbol"]==symbol)
    
    # LOT_SIZE
    lot = next(f for f in s_info["filters"] if f["filterType"]=="LOT_SIZE")
    step_size = float(lot["stepSize"])
    min_qty = float(lot["minQty"])
    max_qty = float(lot["maxQty"])

    # NOTIONAL
    notf = [f for f in s_info["filters"] if f["filterType"] in ["NOTIONAL","MIN_NOTIONAL"]]
    min_notional = None
    if notf:
        min_notional= float(notf[0]["minNotional"])

    # Otomatik ayarlama istenirse:
    if auto_adjust:
        adj_qty = adjust_quantity(qty, step_size)
        if adj_qty != qty:
            log(f"[check_lot_and_notional] auto_adjust => {qty} -> {adj_qty} (step={step_size})", "info")
        qty = adj_qty

    # Ardından kontroller:
    if not(min_qty <= qty <= max_qty):
        raise ValueError(f"[LOT_SIZE] {qty} out of range => {min_qty}~{max_qty}")

    # Step mod check
    residual = qty / step_size - int(qty / step_size)
    if abs(residual) > 1e-8:
        raise ValueError(f"[LOT_SIZE] mod error => qty={qty}, step={step_size}")

    # Notional
    total = qty * price
    if min_notional is not None:
        if total < min_notional:
            raise ValueError(f"[NOTIONAL] {total:.6f} < {min_notional}")

#  # core/utils.py

# import math
# from decimal import Decimal, ROUND_DOWN
# from binance import AsyncClient

# async def check_lot_and_notional(client_async: AsyncClient, symbol: str, qty: float, price: float):
#     info = await client_async.get_exchange_info()
#     s_info = next(s for s in info["symbols"] if s["symbol"]==symbol)
#     lot = next(f for f in s_info["filters"] if f["filterType"]=="LOT_SIZE")
#     step_size = float(lot["stepSize"])
#     min_qty = float(lot["minQty"])
#     max_qty = float(lot["maxQty"])
#     if not(min_qty<=qty<=max_qty):
#         raise ValueError(f"[LOT_SIZE] {qty} out of range => {min_qty}~{max_qty}")

#     residual = qty/step_size - int(qty/step_size)
#     if abs(residual)>1e-8:
#         raise ValueError(f"[LOT_SIZE] mod error => qty={qty}, step={step_size}")

#     notf= [f for f in s_info["filters"] if f["filterType"] in ["NOTIONAL","MIN_NOTIONAL"]]
#     if not notf:
#         return
#     min_notional= float(notf[0]["minNotional"])
#     total= qty* price
#     if total< min_notional:
#         raise ValueError(f"[NOTIONAL] {total:.6f} < {min_notional}")

# def adjust_quantity(qty: float, step_size: float)->float:
#     return float(f"{(qty // step_size)* step_size:.8f}")

# def format_price(price: float, tick_size: float)->float:
#     try:
#         ts = Decimal(str(tick_size))
#         pr = Decimal(str(price))
#         formatted = (pr // ts)* ts
#         return float(formatted.quantize(ts, rounding=ROUND_DOWN))
#     except:
#         return price
