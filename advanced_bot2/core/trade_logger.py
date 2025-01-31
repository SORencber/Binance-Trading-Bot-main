import aiofiles
import os,csv
from datetime import datetime

CSV_FILE_NAME = "trade_history.csv"

async def append_trade_record(symbol: str, side: str, qty: float, price: float,
                              realized_pnl: float, net_pnl: float):
    """
    aiofiles ile asenkron olarak CSV'ye yeni bir satır ekler.
    Kolonlar: [timestamp, symbol, side, qty, price, realizedPnL, netPnL]
    """
    file_exists = os.path.exists(CSV_FILE_NAME)

    # Dosyayı "a" (append) modunda açıyoruz, newline='' => CSV formatı
    async with aiofiles.open(CSV_FILE_NAME, mode="a", newline="", encoding="utf-8") as f:
        # Eğer dosya yeni oluşturulmuşsa başlık satırı ekle
        if not file_exists:
            header_line = "timestamp,symbol,side,qty,price,realizedPnL,netPnL\n"
            await f.write(header_line)

        now_str = datetime.utcnow().isoformat()
        row = f"{now_str},{symbol},{side},{qty},{price},{realized_pnl},{net_pnl}\n"
        # Asenkron write
        await f.write(row)

async def get_last_net_pnl(symbol: str) -> float:
    """
    Asenkron olarak trade_history.csv'yi okuyup,
    'symbol' için en son netPnL'yi döndürür. Yoksa 0.0.
    """
    if not os.path.exists(CSV_FILE_NAME):
        return 0.0

    last_pnl = 0.0
    # csv modülüyle asenkron okumada bir "trick" gerek;
    # ya satır satır manual parse ya da "aiofiles.tempfile" gibi advanced kullanım.
    # En basit yol: Tüm satırları asenkron oku, sonra csv.reader ile parse.

    async with aiofiles.open(CSV_FILE_NAME, mode="r", encoding="utf-8") as f:
        # Tüm satırları asenkron okuyalım
        lines = await f.readlines()

    # lines => senkron python list[str]
    # CSV parse edelim
    reader = csv.DictReader(lines)
    for row in reader:
        if row["symbol"] == symbol:
            last_pnl = float(row["netPnL"])
    return last_pnl
