# telegram_bot/telegram_app.py

import asyncio
from telegram.ext import ApplicationBuilder, CommandHandler
from .handlers import (
    start_command, status_command, stop_command,
    find_coin_command, positions_command, pnl_command,
    pause_command, resume_command,setsymbol_command
)

class TelegramBotApp:
    def __init__(self, config, shared_ctx):
        self.config = config
        self.shared_ctx = shared_ctx
        self.bot_token = config.get("telegram_token","")
        self.app = None

    async def start_bot(self):
        self.app = ApplicationBuilder().token(self.bot_token).build()
        self.app.bot_data["shared_context"] = self.shared_ctx

        # Komutlar
        self.app.add_handler(CommandHandler("basla", start_command))
        self.app.add_handler(CommandHandler("durum", status_command))
        self.app.add_handler(CommandHandler("dur", stop_command))
        self.app.add_handler(CommandHandler("bul", find_coin_command))
        self.app.add_handler(CommandHandler("pozisyonlar", positions_command))
        self.app.add_handler(CommandHandler("pnl", pnl_command))
        self.app.add_handler(CommandHandler("dur", pause_command))
        self.app.add_handler(CommandHandler("devam", resume_command))
        self.app.add_handler(CommandHandler("goster", setsymbol_command))

        # --- BURAYA YENİ KOMUT EKLEYECEĞİZ ---
        # (Aşağıda gösteriliyor)

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        
        
        # ÖNEMLİ EKLEME:
        self.shared_ctx.telegram_app = self.app  
       
       
        # Manuel while => eğer wait_for_stop yoksa
        while not getattr(self.shared_ctx,"stop_requested",False):
            await asyncio.sleep(1)
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
