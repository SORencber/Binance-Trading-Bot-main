# telegram_bot/handlers.py

from telegram import Update
from telegram.ext import ContextTypes
from core.logging_setup import log
from core.trade_logger import get_last_net_pnl
from core.context import SymbolState
from telegram_bot.telegram_helper import analyze_data,analyze_coins
from telegram import ReplyKeyboardMarkup
from data.data_collector import update_data
from exchange_clients.binance_spot_manager_async import BinanceSpotManagerAsync

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start => hoşgeldin mesajı ve butonlar
    """
    keyboard = [
        ["/durum"],
        ["/goster"],
        ["/bul"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    await update.message.reply_text(
        "Merhaba! Aşağıdaki butonları kullanarak komutları çalıştırabilirsin:",
        reply_markup=reply_markup
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /status => bot durumu
    """
    shared_ctx = context.application.bot_data["shared_context"]
    msg = f"Bot su an müsait. /goster KoinAdi (Ornek, /goster BTCUSDT) komutu ile istediginizi sorabilirsiniz. "
    await update.message.reply_text(msg)

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /stop => Botu durdurma isteği
    """
    shared_ctx = context.application.bot_data["shared_context"]
    # Sadece authorized user
    user_id = update.effective_user.id
    allowed = shared_ctx.config.get("allowed_user_ids", [])
    print("sesver",update.effective_chat.id)

    log(f"DEBUG /stop => user_id={user_id}, allowed={allowed}", "info")

    if user_id not in shared_ctx.config.get("allowed_user_ids",[]):
        await update.message.reply_text("Unauthorized.")
        return

    # Basitçe shared_ctx içine "stop_requested" bayrağı
    shared_ctx.stop_requested = True
    await update.message.reply_text("Bot durdurma isteği kaydedildi.")

async def find_coin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    shared_ctx = context.application.bot_data["shared_context"]
    user_id = update.effective_user.id
    if user_id not in shared_ctx.config.get("allowed_user_ids",[]):
        await update.message.reply_text("Yetkisiz Erisim, Lütfen yonetici ile iletisime gecin..")
        return
    tokens = update.message.text.split()
   
    # = tokens[1].upper()
    #await update.message.reply_text(f"Symbol {new_sym} added.")
    await update.message.reply_text(f"ilk 20 Koin Analiz siraya alindi , Lütfen bekleyiniz.")

    # 1) (Opsiyonel) Trading'i durdur
    #from main import stop_trading, start_trading  # Örnek, proje yapınıza göre
    #await stop_trading(shared_ctx)
   # await update.message.reply_text("Trading tasks paused.")

    # 2) Symbol listeyi temizle => yeni sembol ekle
    #shared_ctx.config["symbols"] = [new_sym]
    #shared_ctx.config["command_source"] = "telegram"

    # **Burada symbol_map'i sıfırlıyoruz**
    #shared_ctx.symbol_map = {sym: SymbolState(sym) for sym in shared_ctx.config["symbols"]}
    await analyze_coins(shared_ctx)    
    
 
    #print("Yeni semboller:", shared_ctx.symbol_map.keys())  # Debug için
    #await update.message.reply_text(f"{new_sym} icin Analiz tamamlandi.")

    # 3) Tekrar trading başlat
    #await start_trading(shared_ctx)
    #await update.message.reply_text("Trading tasks resumed.")


async def setsymbol_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    shared_ctx = context.application.bot_data["shared_context"]
    user_id = update.effective_user.id
    if user_id not in shared_ctx.config.get("allowed_user_ids",[]):
        await update.message.reply_text("Yetkisiz Erisim, Lütfen yonetici ile iletisime gecin..")
        return
    tokens = update.message.text.split()
    if len(tokens) < 2:
        await update.message.reply_text("Kullanim: /goster KoinAdi (ornek. BTCUSDT)")
        return
    new_sym = tokens[1].upper()
    #await update.message.reply_text(f"Symbol {new_sym} added.")
    await update.message.reply_text(f"{new_sym} icin Analiz siraya alindi , Lütfen bekleyiniz.")

    # 1) (Opsiyonel) Trading'i durdur
    #from main import stop_trading, start_trading  # Örnek, proje yapınıza göre
    #await stop_trading(shared_ctx)
   # await update.message.reply_text("Trading tasks paused.")

    # 2) Symbol listeyi temizle => yeni sembol ekle
    #shared_ctx.config["symbols"] = [new_sym]
    #shared_ctx.config["command_source"] = "telegram"

    # **Burada symbol_map'i sıfırlıyoruz**
    #shared_ctx.symbol_map = {sym: SymbolState(sym) for sym in shared_ctx.config["symbols"]}
    await analyze_data(ctx= shared_ctx,symbol=new_sym)
    #print("Yeni semboller:", shared_ctx.symbol_map.keys())  # Debug için
    #await update.message.reply_text(f"{new_sym} icin Analiz tamamlandi.")

    # 3) Tekrar trading başlat
    #await start_trading(shared_ctx)
    #await update.message.reply_text("Trading tasks resumed.")



async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /positions => open positions
    """
    shared_ctx = context.application.bot_data["shared_context"]
    msg=""
    for sym, st in shared_ctx.symbol_map.items():
        if st.has_position:
            msg += f"{sym}: qty={st.quantity}, entry={st.entry_price}, PnL={st.pnl}\n"
    if not msg:
        msg= "No open positions."
    await update.message.reply_text(msg)

async def pnl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /pnl => total PnL
    """
    shared_ctx = context.application.bot_data["shared_context"]
    total_pnl=0
    for sym, st in shared_ctx.symbol_map.items():
        total_pnl+= st.pnl
    await update.message.reply_text(f"Total PnL= {total_pnl:.2f}")

async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    shared_ctx = context.application.bot_data["shared_context"]
    user_id= update.effective_user.id
    allowed= shared_ctx.config.get("allowed_user_ids",[user_id])
    if user_id not in allowed:
        await update.message.reply_text("Yetkisiz erisim, Lütfen Yonetici ile gorusun.")
        return

    # stop_trading => import main or store reference
    # Minimal: direct dynamic import
    from main import stop_trading
    await stop_trading(shared_ctx)
    await update.message.reply_text("Ticari gorevler durdu.")

async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    shared_ctx = context.application.bot_data["shared_context"]
    user_id= update.effective_user.id
    allowed= shared_ctx.config.get("allowed_user_ids",[user_id])
    if user_id not in allowed:
        await update.message.reply_text("Yetkisiz erisim, Lütfen yonetici ile gorusunuz..")
        return

    from main import start_trading
    await start_trading(shared_ctx)
    await update.message.reply_text("Ticari gorevler devam ediyor.")

    async def netpnl_command(self, update, context):
        symbol_list = self.ctx.config["symbols"]
        if not symbol_list:
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text="Hic bir coin ayarlanamadi.")
            return

        symbol = symbol_list[0]
        last_pnl = await get_last_net_pnl(symbol)  # asenkron CSV okuma

        msg = f"Son netPnL for {symbol} = {last_pnl:.2f} USDT"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)