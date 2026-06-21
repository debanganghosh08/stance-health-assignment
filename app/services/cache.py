import os
import time
import logging
import string
import hashlib
import json
from typing import Any, Optional

logger = logging.getLogger("app.services.cache")

# Configurations from environment variables
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

class BaseCache:
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class MemoryCache(BaseCache):
    def __init__(self):
        self._data = {}  # key -> (value, expiry_timestamp)

    def get(self, key: str) -> Optional[Any]:
        if not CACHE_ENABLED:
            return None
        if key in self._data:
            val_str, expiry = self._data[key]
            if expiry is None or time.time() < expiry:
                try:
                    return json.loads(val_str)
                except Exception:
                    return val_str
            else:
                self.delete(key)
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not CACHE_ENABLED:
            return
        expiry = None
        ttl_val = ttl if ttl is not None else CACHE_TTL_SECONDS
        if ttl_val is not None:
            expiry = time.time() + ttl_val
        
        try:
            val_str = json.dumps(value)
        except Exception:
            val_str = value
            
        self._data[key] = (val_str, expiry)

    def delete(self, key: str) -> None:
        if key in self._data:
            del self._data[key]

    def clear(self) -> None:
        self._data.clear()


# Try to import redis package optionally
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCache(BaseCache):
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = None
        if REDIS_AVAILABLE:
            try:
                # Setup redis client with a short timeout to prevent blocking startup
                self.client = redis.from_url(
                    redis_url, 
                    socket_timeout=2.0, 
                    socket_connect_timeout=2.0,
                    decode_responses=True
                )
                self.client.ping()
                logger.info(f"✅ Connected to Redis successfully at {redis_url}")
            except Exception as e:
                logger.warning(f"⚠️ Redis connect fail: {e}. Fallback to memory cache.")
                self.client = None
        else:
            logger.info("ℹ️ 'redis' package is not installed. RedisCache is disabled.")

    @property
    def is_active(self) -> bool:
        return self.client is not None

    def get(self, key: str) -> Optional[Any]:
        if not CACHE_ENABLED or not self.is_active:
            return None
        try:
            val = self.client.get(key)
            if val is not None:
                return json.loads(val)
        except Exception as e:
            logger.warning(f"⚠️ Redis get error for key '{key}': {e}. Falling back to memory cache.")
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not CACHE_ENABLED or not self.is_active:
            return
        try:
            val_str = json.dumps(value)
            ttl_val = ttl if ttl is not None else CACHE_TTL_SECONDS
            self.client.set(key, val_str, ex=ttl_val)
        except Exception as e:
            logger.warning(f"⚠️ Redis set error for key '{key}': {e}. Falling back to memory cache.")

    def delete(self, key: str) -> None:
        if not self.is_active:
            return
        try:
            self.client.delete(key)
        except Exception as e:
            logger.warning(f"⚠️ Redis delete error for key '{key}': {e}.")

    def clear(self) -> None:
        if not self.is_active:
            return
        try:
            # Delete keys with specific project prefixes
            keys = []
            for prefix in ["triage:*", "rag:*", "llm:*"]:
                keys.extend(self.client.keys(prefix))
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            logger.warning(f"⚠️ Redis clear error: {e}.")


class CacheService(BaseCache):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CacheService, cls).__new__(cls, *args, **kwargs)
            cls._instance._init_caches()
        return cls._instance

    def _init_caches(self):
        self.memory_cache = MemoryCache()
        self.redis_cache = RedisCache(REDIS_URL)

    def get(self, key: str) -> Optional[Any]:
        if not CACHE_ENABLED:
            return None
        
        # 1. Try Redis first
        if self.redis_cache.is_active:
            val = self.redis_cache.get(key)
            if val is not None:
                return val
        
        # 2. Fallback to Memory Cache
        return self.memory_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not CACHE_ENABLED:
            return
        
        # 1. Try Redis first
        if self.redis_cache.is_active:
            try:
                self.redis_cache.set(key, value, ttl)
                # Mirror in memory cache for double-layer backup if required,
                # but to be conservative we write to both so if Redis goes down,
                # the in-memory fallback already has the data.
                self.memory_cache.set(key, value, ttl)
                return
            except Exception:
                pass
        
        # 2. Fallback / direct write to Memory Cache
        self.memory_cache.set(key, value, ttl)

    def delete(self, key: str) -> None:
        if self.redis_cache.is_active:
            self.redis_cache.delete(key)
        self.memory_cache.delete(key)

    def clear(self) -> None:
        self.redis_cache.clear()
        self.memory_cache.clear()


# Global Singleton instance of CacheService
cache_service = CacheService()


def normalize_message(message: str) -> str:
    """
    Normalizes a message string by converting it to lowercase,
    removing punctuation, and stripping all whitespaces.
    """
    if not message:
        return ""
    
    # 1. Lowercase
    msg = message.lower()
    
    # 2. Remove punctuation
    msg = msg.translate(str.maketrans("", "", string.punctuation))
    
    # 3. Strip all spaces (remove all spaces to canonicalize semantically equivalent spacing)
    msg = "".join(msg.split())
    
    return msg


def get_normalized_hash(message: str) -> str:
    """
    Computes an MD5 hash of the normalized message.
    """
    norm = normalize_message(message)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()
