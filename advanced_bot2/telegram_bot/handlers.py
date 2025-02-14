# telegram_bot/handlers.py

from telegram import Update
from telegram.ext import ContextTypes
from core.logging_setup import log
from core.trade_logger import get_last_net_pnl
from core.context import SymbolState
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start => hoşgeldin mesajı
    """
    await update.message.reply_text("Merhaba! Komutlar: /status, /pause, /resume, /setsymbol SYM(coin ismi yazarak, coin icin bilgi alirsiniz.)  , /addsymbol SYM, /positions, /pnl")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /status => bot durumu
    """
    shared_ctx = context.application.bot_data["shared_context"]
    msg = f"Bot is running. Current symbols: {shared_ctx.config['symbols']}"
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

async def addsymbol_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /addsymbol BTCUSDT => run-time symbol ekleme
    """
    shared_ctx = context.application.bot_data["shared_context"]
    user_id = update.effective_user.id
    if user_id not in shared_ctx.config.get("allowed_user_ids",[]):
        await update.message.reply_text("Unauthorized.")
        return
    
    tokens = update.message.text.split()
    if len(tokens)<2:
        await update.message.reply_text("Usage: /addsymbol SYMBOL")
        return
    new_sym = tokens[1].upper()

    if new_sym not in shared_ctx.config["symbols"]:
        #shared_ctx = SharedContext(BOT_CONFIG)

        shared_ctx.config["symbols"].append(new_sym)
        # Not: manage_tasks'te anlık ek okuması => 
        # en basit: shared_ctx.symbol_map[new_sym] = SymbolState(new_sym)
        # + price reader tasks vs. 
        # Sade bir mesaj
        await update.message.reply_text(f"Symbol {new_sym} added.")
    else:
        await update.message.reply_text(f"{new_sym} already in list.")
# telegram_bot/handlers.py
async def setsymbol_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    shared_ctx = context.application.bot_data["shared_context"]
    user_id = update.effective_user.id
    if user_id not in shared_ctx.config.get("allowed_user_ids",[]):
        await update.message.reply_text("Unauthorized.")
        return
    tokens = update.message.text.split()
    if len(tokens) < 2:
        await update.message.reply_text("Usage: /setsymbol SYMBOL")
        return
    new_sym = tokens[1].upper()
    await update.message.reply_text(f"Symbol {new_sym} added.")

    # 1) (Opsiyonel) Trading'i durdur
    from main import stop_trading, start_trading  # Örnek, proje yapınıza göre
    await stop_trading(shared_ctx)
    await update.message.reply_text("Trading tasks paused.")

    # 2) Symbol listeyi temizle => yeni sembol ekle
    shared_ctx.config["symbols"] = [new_sym]
    shared_ctx.config["command_source"] = "telegram"

    # **Burada symbol_map'i sıfırlıyoruz**
    shared_ctx.symbol_map = {sym: SymbolState(sym) for sym in shared_ctx.config["symbols"]}
    
    print("Yeni semboller:", shared_ctx.symbol_map.keys())  # Debug için

    # 3) Tekrar trading başlat
    await start_trading(shared_ctx)
    await update.message.reply_text("Trading tasks resumed.")

    await update.message.reply_text(f"Symbol changed to {new_sym}, tasks restarted.")

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
        await update.message.reply_text("Unauthorized.")
        return

    # stop_trading => import main or store reference
    # Minimal: direct dynamic import
    from main import stop_trading
    await stop_trading(shared_ctx)
    await update.message.reply_text("Trading tasks paused.")

async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    shared_ctx = context.application.bot_data["shared_context"]
    user_id= update.effective_user.id
    allowed= shared_ctx.config.get("allowed_user_ids",[user_id])
    if user_id not in allowed:
        await update.message.reply_text("Unauthorized.")
        return

    from main import start_trading
    await start_trading(shared_ctx)
    await update.message.reply_text("Trading tasks resumed.")

    async def netpnl_command(self, update, context):
        symbol_list = self.ctx.config["symbols"]
        if not symbol_list:
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text="No symbols configured.")
            return

        symbol = symbol_list[0]
        last_pnl = await get_last_net_pnl(symbol)  # asenkron CSV okuma

        msg = f"Son netPnL for {symbol} = {last_pnl:.2f} USDT"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)