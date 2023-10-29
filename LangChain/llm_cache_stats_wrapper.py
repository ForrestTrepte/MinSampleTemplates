from typing import Any, Dict, List, Optional
from langchain.cache import BaseCache, RETURN_VAL_TYPE

class LlmCacheStatsWrapper:
    """Wrapper for an LLM cache that tracks the number of cache hits and tokens used."""

    def __init__(self, inner_cache: BaseCache) -> None:
        self.inner_cache = inner_cache
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_stores = 0

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """
        Look up based on prompt and llm_string.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.

        Returns:
            The value in the cache, or None if not found.
        """
        result = self.inner_cache.lookup(prompt, llm_string)
        if result:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return result

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        """
        Update cache based on prompt and llm_string and persist to JSON text file.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.
            return_val: The value to store in the cache.
        """
        self.inner_cache.update(prompt, llm_string, return_val)
        self.cache_stores += 1

    def clear_cache_stats(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_stores = 0

    def dump_cache_stats(self):
        print(f"LLM Cache: {self.cache_hits} hits, {self.cache_misses} misses, {self.cache_stores} stores")
