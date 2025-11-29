"""Message queue consumer entry point for Analytics Engine."""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def consume_messages():
    """Main consumer loop."""
    logger.info("Starting Analytics Engine consumer...")
    logger.info("Consumer is running. Press Ctrl+C to stop.")

    try:
        while True:
            # Placeholder for actual message consumption
            # Will be implemented in later phases
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user")
    except Exception as e:
        logger.error(f"Consumer error: {e}")
        raise


async def main():
    """Entry point for the consumer."""
    await consume_messages()


if __name__ == "__main__":
    asyncio.run(main())
