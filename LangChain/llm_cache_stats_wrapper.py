from typing import Any, Dict, List, Optional
from langchain.cache import BaseCache, RETURN_VAL_TYPE
from collections import namedtuple
import tiktoken
import re

class LlmCacheStatsWrapper:
    """Wrapper for an LLM cache that tracks the number of cache hits and tokens used."""
    class Stat:
        def __init__(self):
            self.count = 0
            self.input_tokens = 0
            self.output_tokens = 0
        
        def add_lookup(self, input_tokens, output_tokens):
            self.count += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def __init__(self, inner_cache: BaseCache) -> None:
        """
        Initialize the wrapper.

        Args:
            inner_cache: The cache to wrap.
            encoding: The encoding used by the model.
        """
        self.inner_cache = inner_cache
        self.cache_hits = self.Stat()
        self.cache_misses = self.Stat()
        self.cache_stores = 0
        self.model_name_re = re.compile(r"'model_name',\s*'(?P<model_name_format_1>[^']+)'|\"model_name\":\s*\"(?P<model_name_format_2>[^\"]+)\"")
        self.encodings = {}

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

        model_name_match = self.model_name_re.search(llm_string)
        if not model_name_match:
            raise ValueError(f"Could not find model_name in llm_string: {llm_string}")
        model_name = model_name_match.group('model_name_format_1') or model_name_match.group('model_name_format_2')
        if model_name not in self.encodings:
            self.encodings[model_name] = tiktoken.encoding_for_model(model_name)
        encoding = self.encodings[model_name]

        input_tokens = len(encoding.encode(prompt))
        output_tokens = 0
        for generation in result:
            output_tokens += len(encoding.encode(generation.text))
        (self.cache_hits if result else self.cache_misses).add_lookup(input_tokens, output_tokens)
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
        self.cache_hits = self.Stat()
        self.cache_misses = self.Stat()
        self.cache_stores = 0

    def dump_cache_stats(self, input_cents_per_1k_tokens = 0, output_cents_per_1k_tokens = 0):
        total_input_tokens = self.cache_hits.input_tokens + self.cache_misses.input_tokens
        total_output_tokens = self.cache_hits.output_tokens + self.cache_misses.output_tokens
        print(f"LLM Cache: {self.cache_hits.count} hits, {self.cache_misses.count} misses, {self.cache_stores} stores")
        print(f"           {self.cache_misses.input_tokens} new input tokens, {self.cache_misses.output_tokens} new output tokens, {total_input_tokens} total input tokens, {total_output_tokens} total output tokens")
        if input_cents_per_1k_tokens > 0 or output_cents_per_1k_tokens > 0:
            new_cost = (input_cents_per_1k_tokens * self.cache_misses.input_tokens + output_cents_per_1k_tokens * self.cache_misses.output_tokens) / 1000.0
            total_cost = (input_cents_per_1k_tokens * total_input_tokens + output_cents_per_1k_tokens * total_output_tokens) / 1000.0
            print(f"           new (this run) API cost: ${new_cost:.2f}, total (including previously-cached runs) API cost: ${total_cost:.2f}")