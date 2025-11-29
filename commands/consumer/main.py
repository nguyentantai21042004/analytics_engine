"""
Analytics Engine Consumer - Main entry point.
Loads config, initializes instances, and starts the consumer service.
"""

import asyncio
import logging

from core.config import settings
from core.logger import logger
from internal.consumers.main import consume_messages


async def main():
    """Entry point for the Analytics Engine consumer."""
    try:
        logger.info(
            f"========== Starting {settings.service_name} v{settings.service_version} Consumer service =========="
        )

        # Start consuming messages
        await consume_messages()

    except KeyboardInterrupt:
        logger.info("Consumer stopped by user")
    except Exception as e:
        logger.error(f"Consumer error: {e}")
        logger.exception("Consumer error details:")
        raise
    finally:
        logger.info("========== Consumer service stopped ==========")


if __name__ == "__main__":
    asyncio.run(main())
