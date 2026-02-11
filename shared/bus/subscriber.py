"""Event subscriber for the message bus.

Wraps Redis pub/sub for subscribing to topics. Falls back gracefully
if Redis is unavailable.

Usage:
    from shared.bus import EventSubscriber

    sub = EventSubscriber()
    await sub.connect()
    await sub.subscribe("arm.state", my_handler)
    await sub.listen()  # blocking loop
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Optional

from shared.config.service_registry import ServiceConfig

logger = logging.getLogger(__name__)

# Type for async message handlers
MessageHandler = Callable[[str, dict], Coroutine[Any, Any, None]]


class EventSubscriber:
    """Subscribes to events on the Redis message bus.

    Designed for async usage with FastAPI/asyncio services.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self._redis_url = redis_url or ServiceConfig.REDIS_URL
        self._redis = None
        self._pubsub = None
        self._connected = False
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._listen_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to Redis. Returns True if successful."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=2.0,
            )
            await self._redis.ping()
            self._pubsub = self._redis.pubsub()
            self._connected = True
            logger.info("EventSubscriber connected to Redis at %s", self._redis_url)
            return True
        except ImportError:
            logger.warning("redis package not installed — message bus disabled")
            return False
        except Exception as e:
            logger.warning("Failed to connect to Redis: %s — message bus disabled", e)
            self._connected = False
            return False

    async def subscribe(self, topic: str, handler: MessageHandler) -> bool:
        """Subscribe to a topic with a handler.

        Args:
            topic: Topic name or pattern (e.g., 'arm.state', 'objects.*')
            handler: Async function(topic, data_dict) to call on each message

        Returns:
            True if subscribed successfully.
        """
        if not self._connected or self._pubsub is None:
            return False

        if topic not in self._handlers:
            self._handlers[topic] = []
            try:
                if "*" in topic:
                    await self._pubsub.psubscribe(topic)
                else:
                    await self._pubsub.subscribe(topic)
            except Exception as e:
                logger.warning("Failed to subscribe to %s: %s", topic, e)
                return False

        self._handlers[topic].append(handler)
        return True

    async def listen(self) -> None:
        """Start listening for messages. Runs as a background task."""
        if not self._connected or self._pubsub is None:
            return

        async def _listen_loop():
            try:
                async for message in self._pubsub.listen():
                    if message["type"] in ("message", "pmessage"):
                        topic = message.get("channel", "")
                        try:
                            data = json.loads(message["data"])
                        except (json.JSONDecodeError, TypeError):
                            data = {"raw": message["data"]}

                        # Find matching handlers
                        for pattern, handlers in self._handlers.items():
                            if pattern == topic or ("*" in pattern and topic.startswith(pattern.replace("*", ""))):
                                for handler in handlers:
                                    try:
                                        await handler(topic, data)
                                    except Exception as e:
                                        logger.error("Handler error for %s: %s", topic, e)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Listen loop error: %s", e)

        self._listen_task = asyncio.create_task(_listen_loop())

    async def close(self):
        """Close the subscription and Redis connection."""
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
