import contextvars
import hashlib
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

# Setting that can be used for debugging unexpected cache misses. Prints out the key so it can be compared with keys in the cache file.
warn_on_cache_misses = False


class SimpleLlmCache(InMemoryCache):
    """Cache that stores things in memory and persists to a JSON text file."""

    def __init__(self, filename: str = "llm-cache.json") -> None:
        """
        Initialize with empty cache and filename for JSON text file.
        """
        # We have changed the name of the base class _cache to _scache to avoid type errors.
        self._scache: Dict[str, List[str]] = {}

        # Trial is saved in a context variable so that it can be set in async methods
        # with their own context, without interfering with other async methods that are
        # also using the cache while running concurrently.
        self._trial_contextvar = contextvars.ContextVar("cache trial", default=0)

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
        self._trial_contextvar.set(trial)

    def get_trial(self) -> int:
        """Get the current trial index."""
        result = self._trial_contextvar.get()
        return result

    def _get_key(self, prompt: str, llm_string: str) -> str:
        """
        Convert the tuple key to a string key.

        Args:
            prompt: The prompt to use as the key.
            llm_string: The llm_string to use as part of the key.

        Returns:
            The string key.
        """
        # Sometimes (and sometimes not), with Gemini, the llm string contains async_client information
        # we need to ignore in order to retrieve cached results. Even if it consistently returned this info,
        # it contains a repr with a memory address that we'd need to ignore.
        canonicalized_llm_string = llm_string
        if '"async_client":' in llm_string:
            # Parse the non-json text before --- at the end of llm_string.
            llm_string_dash_split = llm_string.rsplit("---", 1)
            llm_string_json_text = llm_string_dash_split[0]
            llm_string_dash_text = (
                "---" + llm_string_dash_split[1]
                if len(llm_string_dash_split) > 1
                else ""
            )
            llm_string_json = json.loads(llm_string_json_text)
            id = llm_string_json.get("id", [])
            if "ChatVertexAI" in id:
                kwargs = llm_string_json.get("kwargs", {})
                if "async_client" in kwargs:
                    kwargs.pop("async_client")
                    canonicalized_llm_string = json.dumps(llm_string_json)
                    canonicalized_llm_string += llm_string_dash_text
        result = f"trial {self._trial_contextvar.get()} ::: {prompt} ::: {canonicalized_llm_string})"
        return result

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
            if warn_on_cache_misses:
                logger.warning(
                    f"Cache miss {self.get_key_hash(key)} (see debug log file for string literal of key)"
                )
                logger.debug(self.get_string_literal(key))
            return None
        generations = []
        for generation in value:
            generations.append(loads(generation))
        logger.debug(f"< SimpleLlmCache.lookup (hit)")
        return generations

    @classmethod
    def get_key_hash(self, key: str) -> str:
        trial_split = key.split(":::", 1)
        trial_part = trial_split[0].strip().replace("trial ", "t")
        hash_after_trial = hashlib.sha1(trial_split[1].encode("utf-8")).hexdigest()[:8]
        result = f"{trial_part}:{hash_after_trial}"
        return result

    @classmethod
    def get_string_literal(cls, s: str) -> str:
        # Repr is a string literal surrounded by single quotes.
        single_quoted = repr(s)
        assert single_quoted[0] == "'" and single_quoted[-1] == "'"
        # We need to convert it to double quotes in order to match the string literals in the cache file.
        double_quoted = single_quoted[1:-1]
        double_quoted = double_quoted.replace('"', r"\"")
        double_quoted = double_quoted.replace("\\'", "'")
        double_quoted = '"' + double_quoted + '"'
        return double_quoted

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
        if warn_on_cache_misses:
            logger.debug(f"Cache update {self.get_key_hash(key)}")
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
