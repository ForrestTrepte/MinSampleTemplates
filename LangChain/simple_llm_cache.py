import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain.load.dump import dumps
from langchain.load.load import loads
from langchain.schema import Generation
from langchain_community.cache import InMemoryCache
from langchain_core.caches import RETURN_VAL_TYPE

logger = logging.getLogger(__name__)
time_per_backup = timedelta(minutes=15)


class SimpleLlmCache(InMemoryCache):
    """Cache that stores things in memory and persists to a JSON text file."""

    def __init__(self, filename: str = "llm-cache.json") -> None:
        """
        Initialize with empty cache and filename for JSON text file.
        """
        # We have changed the name of the base class _cache to _scache to avoid type errors.
        self._scache: Dict[str, List[str]] = {}
        self._trial = 0
        self._filename = filename
        self.last_backup = datetime.min
        try:
            with open(self._filename, "r") as f:
                self._scache = json.load(f)
        except FileNotFoundError:
            pass

    def set_trial(self, trial: int) -> None:
        """Set a trial index. This permits generating new results for the same prompt and llm_string.
        For example, when testing LLM prompts this can used to generate multiple different responses for the prompt,
        while still caching the results for each trial."""
        self._trial = trial

    def _get_key(self, prompt: str, llm_string: str) -> str:
        """
        Convert the tuple key to a string key.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.

        Returns:
            The string key.
        """
        return f"trial {self._trial} ::: {prompt} ::: {llm_string})"

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """
        Look up based on prompt and llm_string.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.

        Returns:
            The value in the cache, or None if not found.
        """
        logger.debug(f"> SimpleLlmCache.lookup")
        key = self._get_key(prompt, llm_string)
        value = self._scache.get(key, None)
        if not value:
            logger.debug(f"< SimpleLlmCache.lookup (miss)")
            return None
        generations = []
        for generation in value:
            generations.append(loads(generation))
        logger.debug(f"< SimpleLlmCache.lookup (hit)")
        return generations

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        """
        Update cache based on prompt and llm_string and persist to JSON text file.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.
            return_val: The value to store in the cache.
        """
        logger.debug(f"> SimpleLlmCache.update")
        key = self._get_key(prompt, llm_string)
        generations = []
        for generation in return_val:
            generations.append(dumps(generation))
        self._scache[key] = generations
        # Write it to a temp file first. In case we are interrupted while writing, we don't want a partially-written file.
        temp_filename = self._filename + ".tmp"
        with open(temp_filename, "w") as f:
            json.dump(self._scache, f, indent=4)
        if os.path.exists(self._filename):
            if datetime.now() - self.last_backup > time_per_backup:
                self.last_backup = datetime.now()
                backup_filename = self._filename + ".bak"
                shutil.move(self._filename, backup_filename)
        shutil.move(temp_filename, self._filename)
        logger.debug(f"< SimpleLlmCache.update")

    def clear(self, **kwargs: Any) -> None:
        """
        Clear cache and remove JSON text file.
        """
        try:
            self._scache = {}
            os.remove(self._filename)
        except FileNotFoundError:
            pass
