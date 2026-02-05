"""
Refinement Store - Async Redis-based storage for refinement cycle.

Simple approach:
- First request: Generate result, create request_id, store in Redis
- Subsequent requests: Get stored result + user feedback, refine, overwrite
- Only stores the LAST result (no history needed for proofreading/rephrasing)
"""
import json
import uuid
import redis.asyncio as redis
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, REFINEMENT_TTL
from logs.logging_config import get_llm_logger

logger = get_llm_logger()

# Key prefix for refinement sessions
REFINEMENT_KEY_PREFIX = "refine"


@dataclass
class RefinementData:
    """Data stored for each refinement session."""
    request_id: str
    task: str                      # summary, rephrase, translate, etc.
    current_result: str            # Latest result (overwritten each refinement/regeneration)
    original_text: str             # Original input text
    model: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None  # User name for logging/display
    summary_type: Optional[str] = None  # For summary task: brief, detailed, bulletwise
    target_language: Optional[str] = None  # For translate task
    refinement_count: int = 0      # Count of refinements (without original text)
    regeneration_count: int = 0    # Count of regenerations (with original text)
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefinementData":
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "RefinementData":
        return cls.from_dict(json.loads(json_str))


class RefinementStore:
    """
    Async Redis-based store for refinement sessions.

    Stores only the latest result - overwrites on each refinement.
    """

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        ttl: int = REFINEMENT_TTL
    ):
        self._redis: Optional[redis.Redis] = None
        self._host = host
        self._port = port
        self._db = db
        self._ttl = ttl
        self._prefix = REFINEMENT_KEY_PREFIX

    async def _get_redis(self) -> redis.Redis:
        """Get or create async Redis connection."""
        if self._redis is None:
            self._redis = redis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                decode_responses=True
            )
            logger.info(f"[REFINE_STORE] Initialized | host={self._host}:{self._port} | db={self._db} | ttl={self._ttl}s")
        return self._redis

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _key(self, request_id: str) -> str:
        """Generate Redis key for a request."""
        return f"{self._prefix}:{request_id}"

    def generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())

    async def create(
        self,
        task: str,
        result: str,
        original_text: str,
        model: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        summary_type: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> RefinementData:
        """
        Create a new refinement session.

        Args:
            task: Task type (summary, rephrase, etc.)
            result: The generated result
            original_text: Original input text
            model: Model used
            user_id: Optional user identifier
            user_name: Optional user name for logging
            summary_type: Type of summary (brief, detailed, bulletwise)
            target_language: Target language for translation

        Returns:
            RefinementData with generated request_id
        """
        request_id = self.generate_request_id()

        data = RefinementData(
            request_id=request_id,
            task=task,
            current_result=result,
            original_text=original_text,
            model=model,
            user_id=user_id,
            user_name=user_name,
            summary_type=summary_type,
            target_language=target_language,
            refinement_count=0
        )

        r = await self._get_redis()
        key = self._key(request_id)
        await r.setex(key, self._ttl, data.to_json())

        user_info = f" | user_id={user_id}" if user_id else ""
        logger.info(f"[REFINE_STORE] Created | request_id={request_id} | task={task}{user_info}")

        return data

    async def get(self, request_id: str) -> Optional[RefinementData]:
        """
        Get refinement data by request_id.

        Args:
            request_id: The request identifier

        Returns:
            RefinementData or None if not found/expired
        """
        r = await self._get_redis()
        key = self._key(request_id)
        json_str = await r.get(key)

        if json_str:
            data = RefinementData.from_json(json_str)
            logger.debug(f"[REFINE_STORE] Retrieved | request_id={request_id}")
            return data

        logger.debug(f"[REFINE_STORE] Not found | request_id={request_id}")
        return None

    async def update(
        self,
        request_id: str,
        new_result: str,
        user_id: Optional[str] = None
    ) -> Optional[RefinementData]:
        """
        Update refinement with new result (overwrites previous).

        Args:
            request_id: The request identifier
            new_result: The new refined result
            user_id: Optional user identifier (for logging)

        Returns:
            Updated RefinementData or None if not found
        """
        data = await self.get(request_id)
        if not data:
            return None

        # Update fields
        data.current_result = new_result
        data.refinement_count += 1
        data.updated_at = datetime.now().isoformat()

        # Update user_id if provided and different
        if user_id and data.user_id != user_id:
            data.user_id = user_id

        # Get remaining TTL and preserve it
        r = await self._get_redis()
        key = self._key(request_id)
        ttl = await r.ttl(key)
        if ttl > 0:
            await r.setex(key, ttl, data.to_json())
        else:
            await r.setex(key, self._ttl, data.to_json())

        user_info = f" | user_id={data.user_id}" if data.user_id else ""
        logger.info(
            f"[REFINE_STORE] Updated | request_id={request_id} | "
            f"refinement_count={data.refinement_count}{user_info}"
        )

        return data

    async def update_regeneration(
        self,
        request_id: str,
        new_result: str,
        user_id: Optional[str] = None
    ) -> Optional[RefinementData]:
        """
        Update with regenerated result (uses original text).

        Args:
            request_id: The request identifier
            new_result: The new regenerated result
            user_id: Optional user identifier (for logging)

        Returns:
            Updated RefinementData or None if not found
        """
        data = await self.get(request_id)
        if not data:
            return None

        # Update fields
        data.current_result = new_result
        data.regeneration_count += 1
        data.updated_at = datetime.now().isoformat()

        # Update user_id if provided and different
        if user_id and data.user_id != user_id:
            data.user_id = user_id

        # Get remaining TTL and preserve it
        r = await self._get_redis()
        key = self._key(request_id)
        ttl = await r.ttl(key)
        if ttl > 0:
            await r.setex(key, ttl, data.to_json())
        else:
            await r.setex(key, self._ttl, data.to_json())

        user_info = f" | user_id={data.user_id}" if data.user_id else ""
        logger.info(
            f"[REFINE_STORE] Regenerated | request_id={request_id} | "
            f"regeneration_count={data.regeneration_count}{user_info}"
        )

        return data

    async def delete(self, request_id: str) -> bool:
        """
        Delete a refinement session.

        Args:
            request_id: The request identifier

        Returns:
            True if deleted, False if not found
        """
        r = await self._get_redis()
        key = self._key(request_id)
        result = await r.delete(key)

        if result:
            logger.info(f"[REFINE_STORE] Deleted | request_id={request_id}")
            return True

        logger.debug(f"[REFINE_STORE] Delete failed (not found) | request_id={request_id}")
        return False

    async def extend_ttl(self, request_id: str, ttl: Optional[int] = None) -> bool:
        """
        Extend the TTL for a refinement session.

        Args:
            request_id: The request identifier
            ttl: New TTL in seconds (uses default if not specified)

        Returns:
            True if extended, False if not found
        """
        r = await self._get_redis()
        key = self._key(request_id)
        ttl = ttl or self._ttl

        if await r.exists(key):
            await r.expire(key, ttl)
            logger.debug(f"[REFINE_STORE] Extended TTL | request_id={request_id} | ttl={ttl}s")
            return True

        return False

    async def get_ttl(self, request_id: str) -> int:
        """Get remaining TTL for a request."""
        r = await self._get_redis()
        key = self._key(request_id)
        return await r.ttl(key)

    async def exists(self, request_id: str) -> bool:
        """Check if a request exists."""
        r = await self._get_redis()
        key = self._key(request_id)
        return await r.exists(key) > 0


# Global store instance
_store: Optional[RefinementStore] = None


def get_refinement_store() -> RefinementStore:
    """Get the global refinement store instance."""
    global _store
    if _store is None:
        _store = RefinementStore()
    return _store


async def close_refinement_store():
    """Close the global refinement store connection."""
    global _store
    if _store:
        await _store.close()
        _store = None


def init_refinement_store(
    host: str = REDIS_HOST,
    port: int = REDIS_PORT,
    db: int = REDIS_DB,
    ttl: int = REFINEMENT_TTL
) -> RefinementStore:
    """Initialize the global refinement store with custom settings."""
    global _store
    _store = RefinementStore(host=host, port=port, db=db, ttl=ttl)
    return _store
