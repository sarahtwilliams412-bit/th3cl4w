"""Event publisher for the message bus.

Wraps Redis pub/sub for publishing events to topics. Falls back gracefully
if Redis is unavailable (logs warning, does not crash).

Usage:
    from shared.bus import EventPublisher
    from shared.messages import ArmStateMessage

    pub = EventPublisher()
    await pub.connect()
    await pub.publish("arm.state", ArmStateMessage(...))
    await pub.close()
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel

from shared.config.service_registry import ServiceConfig

logger = logging.getLogger(__name__)


class EventPublisher:
    """Publishes events to the Redis message bus.

    Designed for async usage with FastAPI/asyncio services.
    Falls back gracefully if Redis is not available.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self._redis_url = redis_url or ServiceConfig.REDIS_URL
        self._redis = None
        self._connected = False

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
            self._connected = True
            logger.info("EventPublisher connected to Redis at %s", self._redis_url)
            return True
        except ImportError:
            logger.warning("redis package not installed — message bus disabled")
            return False
        except Exception as e:
            logger.warning("Failed to connect to Redis: %s — message bus disabled", e)
            self._connected = False
            return False

    async def publish(self, topic: str, message: BaseModel | dict | str) -> bool:
        """Publish a message to a topic.

        Args:
            topic: Topic name (e.g., 'arm.state')
            message: Pydantic model, dict, or JSON string

        Returns:
            True if published successfully, False otherwise.
        """
        if not self._connected or self._redis is None:
            return False

        try:
            if isinstance(message, BaseModel):
                payload = message.model_dump_json()
            elif isinstance(message, dict):
                payload = json.dumps(message)
            else:
                payload = str(message)

            await self._redis.publish(topic, payload)
            return True
        except Exception as e:
            logger.warning("Failed to publish to %s: %s", topic, e)
            return False

    async def close(self):
        """Close the Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


class SyncEventPublisher:
    """Synchronous event publisher for non-async code.

    Usage:
        pub = SyncEventPublisher()
        pub.connect()
        pub.publish("arm.state", {"joints": [0, 0, 0, 0, 0, 0]})
    """

    def __init__(self, redis_url: Optional[str] = None):
        self._redis_url = redis_url or ServiceConfig.REDIS_URL
        self._redis = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            import redis
            self._redis = redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=2.0,
            )
            self._redis.ping()
            self._connected = True
            logger.info("SyncEventPublisher connected to Redis at %s", self._redis_url)
            return True
        except ImportError:
            logger.warning("redis package not installed — message bus disabled")
            return False
        except Exception as e:
            logger.warning("Failed to connect to Redis: %s — message bus disabled", e)
            self._connected = False
            return False

    def publish(self, topic: str, message: BaseModel | dict | str) -> bool:
        """Publish a message to a topic."""
        if not self._connected or self._redis is None:
            return False

        try:
            if isinstance(message, BaseModel):
                payload = message.model_dump_json()
            elif isinstance(message, dict):
                payload = json.dumps(message)
            else:
                payload = str(message)

            self._redis.publish(topic, payload)
            return True
        except Exception as e:
            logger.warning("Failed to publish to %s: %s", topic, e)
            return False

    def close(self):
        """Close the Redis connection."""
        if self._redis:
            self._redis.close()
            self._redis = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
