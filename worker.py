# worker.py
import os
import asyncio
import logging

from celery_app import celery_app
from bot import MomentumScalpBot

log = logging.getLogger("WORKER")


@celery_app.task(bind=True)
def run_momentum_bot(self):
    """
    Long-running background task.
    """
    log.info("Starting MomentumScalpBot task")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    bot = MomentumScalpBot(
        symbols=["TQQQ", "SQQQ"],
        key=os.environ["ALPACA_API_KEY"],
        secret=os.environ["ALPACA_SECRET_KEY"],
    )

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        log.warning("Bot interrupted")
    finally:
        bot.stop()
        loop.stop()
        loop.close()
        log.warning("Event loop closed")