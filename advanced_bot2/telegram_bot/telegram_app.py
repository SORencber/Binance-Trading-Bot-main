# telegram_bot/telegram_app.py

import asyncio
from telegram.ext import ApplicationBuilder, CommandHandler
from .handlers import (
    start_command, status_command, stop_command,
    addsymbol_command, positions_command, pnl_command,
    pause_command, resume_command
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
        self.app.add_handler(CommandHandler("start", start_command))
        self.app.add_handler(CommandHandler("status", status_command))
        self.app.add_handler(CommandHandler("stop", stop_command))
        self.app.add_handler(CommandHandler("addsymbol", addsymbol_command))
        self.app.add_handler(CommandHandler("positions", positions_command))
        self.app.add_handler(CommandHandler("pnl", pnl_command))
        self.app.add_handler(CommandHandler("pause", pause_command))
        self.app.add_handler(CommandHandler("resume", resume_command))
        self.app.add_handler(CommandHandler("resume", pnl_command))

        # --- BURAYA YENİ KOMUT EKLEYECEĞİZ ---
        # (Aşağıda gösteriliyor)

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

        # Manuel while => eğer wait_for_stop yoksa
        while not getattr(self.shared_ctx,"stop_requested",False):
            await asyncio.sleep(1)

        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
