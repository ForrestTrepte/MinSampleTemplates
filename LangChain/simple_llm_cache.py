import json
import os
from typing import Any, Dict, List, Optional
from langchain.cache import InMemoryCache, RETURN_VAL_TYPE
from langchain.schema import Generation

class SimpleLlmCache(InMemoryCache):
    """Cache that stores things in memory and persists to a JSON text file."""

    def __init__(self, verbose: bool) -> None:
        """
        Initialize with empty cache and filename for JSON text file.
        """
        self._cache: Dict[str, List[str]] = {}
        self._filename = "llm-cache.json"
        self.verbose = verbose
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_stores = 0
        try:
            with open(self._filename, 'r') as f:
                self._cache = json.load(f)
        except FileNotFoundError:
            pass

    def _get_key(self, prompt: str, llm_string: str) -> str:
        """
        Convert the tuple key to a string key.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.

        Returns:
            The string key.
        """
        return f"{prompt} ::: {llm_string})"
    
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """
        Look up based on prompt and llm_string.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.

        Returns:
            The value in the cache, or None if not found.
        """
        key = self._get_key(prompt, llm_string)
        generations = []
        value = self._cache.get(key, None)
        if self.verbose:
            print(f'Cache {"hit" if value else "miss"} for key {repr(key)}')
        if not value:
            self.cache_misses += 1
            return None
        for generation in value:
            generations.append(Generation(text=generation))
        self.cache_hits += 1
        return generations

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        """
        Update cache based on prompt and llm_string and persist to JSON text file.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.
            return_val: The value to store in the cache.
        """
        key = self._get_key(prompt, llm_string)
        generations = []
        for generation in return_val:
            generations.append(generation.text)
        self._cache[key] = generations
        if self.verbose:
            print(f'Cache store for key {repr(key)}')
        self.cache_stores += 1
        with open(self._filename, 'w') as f:
            json.dump(self._cache, f, indent=4)

    def clear(self) -> None:
        """
        Clear cache and remove JSON text file.
        """
        try:
            self._cache = {}
            os.remove(self._filename)
        except FileNotFoundError:
            pass

    def clear_cache_stats(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_stores = 0

    def dump_cache_stats(self):
        print(f"LLM Cache: {self.cache_hits} hits, {self.cache_misses} misses, {self.cache_stores} stores")