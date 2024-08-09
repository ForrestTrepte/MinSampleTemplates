import ast
import json
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from vertexai.generative_models import GenerativeModel  # type: ignore


class LlmCacheStatsWrapper:
    """Wrapper for an LLM cache that tracks the number of cache hits and tokens used."""

    class BillingUnit(Enum):
        TOKENS = auto()
        CHARACTERS = auto()

    class Stat:
        def __init__(self) -> None:
            self.count = 0
            self.input_units = 0
            self.output_units = 0
            self.billing_units: Set[LlmCacheStatsWrapper.BillingUnit] = set()

        def add(
            self,
            input_units: Tuple[int, "LlmCacheStatsWrapper.BillingUnit"],
            output_units: Tuple[int, "LlmCacheStatsWrapper.BillingUnit"],
        ) -> None:
            self.count += 1
            self.input_units += input_units[0]
            self.billing_units.add(input_units[1])
            self.output_units += output_units[0]
            self.billing_units.add(output_units[1])

    def __init__(self, inner_cache: BaseCache) -> None:
        """
        Initialize the wrapper.

        Args:
            inner_cache: The cache to wrap.
            encoding: The encoding used by the model.
        """
        self.inner_cache = inner_cache
        self.cache_hits_by_model_name: Dict[str, "LlmCacheStatsWrapper.Stat"] = (
            defaultdict(self.Stat)
        )
        self.cache_misses_by_model_name: Dict[str, "LlmCacheStatsWrapper.Stat"] = (
            defaultdict(self.Stat)
        )
        self.encodings: Dict[str, tiktoken.Encoding] = {}
        self.generative_models: Dict[str, GenerativeModel] = {}

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

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        result = await self.inner_cache.alookup(prompt, llm_string)
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

    async def aupdate(self, prompt: str, llm_string: str, return_val: Any) -> None:
        await self.inner_cache.aupdate(prompt, llm_string, return_val)
        self.add_tokens(False, prompt, llm_string, return_val)

    def add_tokens(
        self, is_hit: bool, prompt: str, llm_string: str, result: RETURN_VAL_TYPE
    ) -> None:
        model_name = self.get_model_name_from_llm_string(llm_string)

        input_units = self.count_units(model_name, prompt)
        output_units = (0, input_units[1])
        for generation in result:
            generation_units = self.count_units(model_name, generation.text)
            assert output_units[1] == generation_units[1]
            output_units = (output_units[0] + generation_units[0], generation_units[1])

        (
            self.cache_hits_by_model_name[model_name]
            if is_hit
            else self.cache_misses_by_model_name[model_name]
        ).add(input_units, output_units)

    def count_units(self, model_name: str, text: str) -> Tuple[int, BillingUnit]:
        if model_name.startswith("gpt-"):
            return self.count_tokens_gpt(model_name, text)
        elif model_name.startswith("claude-"):
            # Anothropic doesn't seem to provide a tokenizer. As a hack approximation, we'll use the gpt tokenizer to count tokens.
            return self.count_tokens_gpt("gpt-4", text)
        elif model_name.startswith("gemini-"):
            return self.count_characters_gemini(model_name, text)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def count_tokens_gpt(self, model_name: str, text: str) -> Tuple[int, BillingUnit]:
        if model_name not in self.encodings:
            self.encodings[model_name] = tiktoken.encoding_for_model(model_name)
        encoding = self.encodings[model_name]
        tokens = len(encoding.encode(text))
        return tokens, self.BillingUnit.TOKENS

    def count_characters_gemini(
        self, model_name: str, text: str
    ) -> Tuple[int, BillingUnit]:
        if model_name not in self.encodings:
            self.generative_models[model_name] = GenerativeModel(model_name=model_name)
        generative_model = self.generative_models[model_name]
        tokens = generative_model.count_tokens(model_name)
        return tokens.total_billable_characters, self.BillingUnit.CHARACTERS

    def get_model_name_from_llm_string(self, llm_string: str) -> str:
        try:
            json_end_index = llm_string.rfind("---")
            if json_end_index > 0:
                # OpenAI and Anthropic APIs return JSON string at the beginning of the llm_string
                llm_json_string = llm_string[:json_end_index]
                llm_json = json.loads(llm_json_string)
                kwargs = llm_json["kwargs"]
                model_name = kwargs.get("model_name", None)
                if model_name:
                    assert isinstance(model_name, str)
                    return model_name
                model_name = kwargs.get("model", None)
                if model_name:
                    assert isinstance(model_name, str)
                    return model_name
            else:
                # Google Vertex AI returns a list of tuples that can be converted to a Python dict
                llm_ast = ast.literal_eval(llm_string)
                if isinstance(llm_ast, list) and all(
                    isinstance(item, tuple) and len(item) == 2 for item in llm_ast
                ):
                    llm_dict = dict(llm_ast)
                    model_name = llm_dict.get("model_name", None)
                    if model_name:
                        assert isinstance(model_name, str)
                        return model_name

            raise ValueError("No model name found in kwargs")
        except Exception as e:
            raise ValueError(f"Could not find model name in llm_string: {llm_string}")

    def clear_cache_stats(self) -> None:
        self.cache_hits_by_model_name = defaultdict(self.Stat)
        self.cache_misses_by_model_name = defaultdict(self.Stat)

    def get_cache_stats_summary(self) -> str:
        result = ""

        all_cache_hits = self.cache_hits_by_model_name.values()
        all_cache_misses = self.cache_misses_by_model_name.values()
        result += f"LLM Cache: {sum([x.count for x in all_cache_hits])} hits, {sum([x.count for x in all_cache_misses])} misses\n"

        all_stats = list(all_cache_hits) + list(all_cache_misses)
        all_units = set.union(*[s.billing_units for s in all_stats])
        unit_name = all_units.pop().name.lower() if len(all_units) == 1 else "units"
        result += f"           {sum([x.input_units for x in all_cache_misses])} new input {unit_name}, {sum([x.output_units for x in all_cache_misses])} new output {unit_name}, {sum([x.input_units for x in all_stats])} total input {unit_name}, {sum([x.output_units for x in all_stats])} total output {unit_name}\n"

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
    # (input cost, output cost, billing unit) in USD per million billing units
    _unit_cost_by_model: Dict[str, Tuple[float, float, BillingUnit]] = {
        "gpt-4o": (5.0, 15.0, BillingUnit.TOKENS),
        "gpt-4o-2024-05-13": (5.0, 15.0, BillingUnit.TOKENS),
        "gpt-4o-2024-08-06": (2.5, 10.0, BillingUnit.TOKENS),
        "gpt-4o-mini": (0.15, 0.60, BillingUnit.TOKENS),
        "gpt-4o-mini-2024-07-18": (0.15, 0.60, BillingUnit.TOKENS),
        "gpt-4-1106-preview": (10.0, 30.0, BillingUnit.TOKENS),
        "gpt-4-turbo-2024-04-09": (10.0, 30.0, BillingUnit.TOKENS),
        "gpt-4": (30.00, 60.00, BillingUnit.TOKENS),
        "gpt-4-0613": (30.00, 60.00, BillingUnit.TOKENS),
        "gpt-3.5-turbo": (0.50, 1.50, BillingUnit.TOKENS),
        "gpt-3.5-turbo-instruct": (1.50, 2.00, BillingUnit.TOKENS),
        "claude-3-opus-20240229": (15.0, 75.0, BillingUnit.TOKENS),
        "claude-3-sonnet-20240229": (3.0, 15.0, BillingUnit.TOKENS),
        "claude-3-haiku-20240307": (0.25, 1.25, BillingUnit.TOKENS),
        "gemini-1.5-flash-preview-0514": (0.125, 0.125, BillingUnit.CHARACTERS),
        "gemini-1.5-pro-preview-0514": (1.25, 1.25, BillingUnit.CHARACTERS),
        "gemini-1.0-pro-002": (0.125, 0.125, BillingUnit.CHARACTERS),
        "vertex-claude-3-opus@20240229": (15.0, 75.0, BillingUnit.TOKENS),
        "vertex-claude-3-sonnet@20240229": (3.0, 15.0, BillingUnit.TOKENS),
        "vertex-claude-3-haiku@20240307": (0.25, 1.25, BillingUnit.TOKENS),
    }

    @classmethod
    def _get_model_cost(cls, model_name: str, cache_data: Stat) -> float:
        if not model_name in cls._unit_cost_by_model:
            raise ValueError(f"Unknown model name: {model_name}")

        unit_cost = cls._unit_cost_by_model[model_name]
        one_million = 1_000_000.0
        cost = (
            cache_data.input_units * unit_cost[0]
            + cache_data.output_units * unit_cost[1]
        ) / one_million
        return cost
