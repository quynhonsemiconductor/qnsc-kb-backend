from collections import defaultdict, deque
from datetime import datetime, timedelta
from threading import Lock
import hashlib
import time
import structlog
from redis.asyncio import Redis
from src.core.config import settings

logger = structlog.get_logger()


class InMemoryRateLimiter:
    """Small-process limiter for the MVP; replace with Redis when scaled out."""

    def __init__(self, limit: int = 30, window_seconds: int = 60):
        self.limit = limit
        self.window = timedelta(seconds=window_seconds)
        self._events: dict[str, deque[datetime]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = datetime.utcnow()
        cutoff = now - self.window
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int((events[0] + self.window - now).total_seconds()))
                return False, retry_after
            events.append(now)
            return True, 0


def _is_development_environment() -> bool:
    """Same development set validate_production() admits, so the two guards agree."""
    return settings.ENVIRONMENT.lower() in {"development", "dev", "local"}


class RedisRateLimiter:
    """Fixed-window limiter shared by every API process through Redis.

    Redis is mandatory in production Compose.  If a developer runs without
    it, retain a conservative per-process fallback instead of making login
    and AI unavailable.

    ``fail_closed`` limiters refuse the request instead of degrading, because a
    per-process fallback multiplies the effective limit by the replica count and a
    Redis outage would otherwise remove the limit altogether — an open window for
    credential stuffing and invitation-token guessing.  Development keeps the
    fallback so a local stack without Redis still serves logins.
    """

    def __init__(self, namespace: str, limit: int, window_seconds: int = 60, fail_closed: bool = False):
        self.namespace = namespace
        self.limit = limit
        self.window_seconds = window_seconds
        self.fail_closed = fail_closed
        self.fallback = InMemoryRateLimiter(limit, window_seconds)

    async def allow(self, key: str) -> tuple[bool, int]:
        now = int(time.time())
        bucket = now // self.window_seconds
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        redis_key = f"qnsc:rate:{self.namespace}:{bucket}:{key_hash}"
        client: Redis | None = None
        try:
            client = Redis.from_url(settings.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
            count = await client.incr(redis_key)
            if count == 1:
                await client.expire(redis_key, self.window_seconds + 1)
            if count > self.limit:
                ttl = await client.ttl(redis_key)
                return False, max(1, int(ttl))
            return True, 0
        except Exception as exc:
            if self.fail_closed and not _is_development_environment():
                logger.error(
                    "Redis rate limit unavailable; refusing request",
                    namespace=self.namespace,
                    error=str(exc),
                )
                return False, self.window_seconds
            logger.warning("Redis rate limit unavailable; using local fallback", namespace=self.namespace, error=str(exc))
            return self.fallback.allow(key)
        finally:
            if client is not None:
                await client.aclose()


ai_rate_limiter = RedisRateLimiter("ai", limit=settings.AI_RATE_LIMIT_PER_MINUTE)
auth_rate_limiter = RedisRateLimiter("auth", limit=10, window_seconds=60, fail_closed=True)
source_upload_rate_limiter = RedisRateLimiter("source_upload", limit=10, window_seconds=60)
