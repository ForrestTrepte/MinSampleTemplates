import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import tiktoken
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache


class LlmCacheStatsWrapper:
    """Wrapper for an LLM cache that tracks the number of cache hits and tokens used."""

    class Stat:
        def __init__(self):
            self.count = 0
            self.input_tokens = 0
            self.output_tokens = 0

        def add_tokens(self, input_tokens, output_tokens):
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
        self.cache_hits_by_model_name = defaultdict(self.Stat)
        self.cache_misses_by_model_name = defaultdict(self.Stat)
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
        if result:
            self.add_tokens(True, prompt, llm_string, result)
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
        self.add_tokens(False, prompt, llm_string, return_val)

    def add_tokens(self, is_hit, prompt, llm_string, result):
        model_name = self.get_model_name_from_llm_string(llm_string)

        input_tokens = self.count_tokens(model_name, prompt)
        output_tokens = 0
        for generation in result:
            generation_tokens = self.count_tokens(model_name, generation.text)
            output_tokens += generation_tokens

        (
            self.cache_hits_by_model_name[model_name]
            if is_hit
            else self.cache_misses_by_model_name[model_name]
        ).add_tokens(input_tokens, output_tokens)

    def count_tokens(self, model_name, text):
        if model_name.startswith("gpt-"):
            return self.count_tokens_gpt(model_name, text)
        elif model_name.startswith("claude-"):
            # Anothropic doesn't seem to provide a tokenizer. As a hack approximation, we'll use the gpt tokenizer to count tokens.
            return self.count_tokens_gpt("gpt-4", text)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def count_tokens_gpt(self, model_name, text):
        if model_name not in self.encodings:
            self.encodings[model_name] = tiktoken.encoding_for_model(model_name)
        encoding = self.encodings[model_name]
        tokens = len(encoding.encode(text))
        return tokens

    def get_model_name_from_llm_string(self, llm_string):
        try:
            json_end_index = llm_string.rfind("}") + 1
            llm_json_string = llm_string[:json_end_index]
            llm_json = json.loads(llm_json_string)
            kwargs = llm_json["kwargs"]
            model_name = kwargs.get("model_name", None)
            if model_name:
                return model_name
            model_name = kwargs.get("model", None)
            if model_name:
                return model_name
            raise ValueError("No model name found in kwargs")
        except Exception as e:
            raise ValueError(f"Could not find model name in llm_string: {llm_string}")
        model_name = model_name_match.group(
            "model_name_format_1"
        ) or model_name_match.group("model_name_format_2")

    def clear_cache_stats(self):
        self.cache_hits_by_model_name = defaultdict(self.Stat)
        self.cache_misses_by_model_name = defaultdict(self.Stat)

    def get_cache_stats_summary(self):
        result = ""

        all_cache_hits = self.cache_hits_by_model_name.values()
        all_cache_misses = self.cache_misses_by_model_name.values()
        result += f"LLM Cache: {sum([x.count for x in all_cache_hits])} hits, {sum([x.count for x in all_cache_misses])} misses\n"

        all_stats = list(all_cache_hits) + list(all_cache_misses)
        result += f"           {sum([x.input_tokens for x in all_cache_misses])} new input tokens, {sum([x.output_tokens for x in all_cache_misses])} new output tokens, {sum([x.input_tokens for x in all_stats])} total input tokens, {sum([x.output_tokens for x in all_stats])} total output tokens\n"

        try:
            miss_cost = 0.0
            total_cost = 0.0
            for model_name, cache_misses in self.cache_misses_by_model_name.items():
                cost = self._get_model_cost(model_name, cache_misses)
                miss_cost += cost
                total_cost += cost
            for model_name, cache_hits in self.cache_hits_by_model_name.items():
                cost = self._get_model_cost(model_name, cache_hits)
                total_cost += cost
            result += f"           new (this run) API cost: ${miss_cost:.2f}, total (including previously-cached runs) API cost: ${total_cost:.2f}\n"
        except ValueError as e:
            result += f"           Can't estimate cost: {e}\n"
        return result

    # from https://openai.com/pricing as of 5/14/24
    # (input cost, output cost) in USD per million tokens
    _token_cost_by_model = {
        "gpt-4o": (5.0, 15.0),
        "gpt-4-1106-preview": (10.0, 30.0),
        "gpt-4": (30.00, 60.00),
        "gpt-4-32k": (60.00, 120.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "gpt-3.5-turbo-instruct": (1.50, 2.00),
        "claude-3-haiku-20240307": (0.25, 1.25),
    }

    @classmethod
    def _get_model_cost(cls, model_name, cache_misses):
        if not model_name in cls._token_cost_by_model:
            raise ValueError(f"Unknown model name: {model_name}")

        token_cost = cls._token_cost_by_model[model_name]
        one_million = 1_000_000.0
        cost = (
            cache_misses.input_tokens * token_cost[0]
            + cache_misses.output_tokens * token_cost[1]
        ) / one_million
        return cost
