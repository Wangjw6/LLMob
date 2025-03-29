# -*- coding: utf-8 -*-
import asyncio
import time
from functools import wraps
from typing import NamedTuple

import openai
from engine.llm_configs.config import CONFIG
from engine.llm_configs.base_gpt_api import BaseGPTAPI
from engine.llm_configs.config import Singleton


def retry(max_retries):
    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return await f(*args, **kwargs)
                except Exception:
                    if i == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** i)

        return wrapper

    return decorator


class RateLimiter:
    """Rate control class, each call goes through wait_if_needed, sleep if rate control is needed"""

    def __init__(self, rpm):
        self.last_call_time = 0

        self.interval = 1.1 * 60 / rpm
        self.rpm = rpm

    def split_batches(self, batch):
        return [batch[i: i + self.rpm] for i in range(0, len(batch), self.rpm)]

    async def wait_if_needed(self, num_requests):
        current_time = time.time()
        elapsed_time = current_time - self.last_call_time

        if elapsed_time < self.interval * num_requests:
            remaining_time = self.interval * num_requests - elapsed_tim
            await asyncio.sleep(remaining_time)

        self.last_call_time = time.time()


class Costs(NamedTuple):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    total_budget: float


class CostManager(metaclass=Singleton):
    """计算使用接口的开销"""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0


    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_costs(self) -> Costs:
        """获得所有开销"""
        return Costs(self.total_prompt_tokens, self.total_completion_tokens, self.total_cost, self.total_budget)


class OpenAIGPTAPI(BaseGPTAPI, RateLimiter):
    """
    Check https://platform.openai.com/examples for examples
    """

    def __init__(self):
        self.__init_openai(CONFIG)
        self.llm = openai
        self.model = CONFIG.openai_api_model
        self.auto_max_tokens = True
        self._cost_manager = CostManager()
        RateLimiter.__init__(self, rpm=self.rpm)

    def __init_openai(self, config):
        openai.api_key = config.openai_api_key
        if config.openai_api_base:
            openai.api_base = config.openai_api_base
        if config.openai_api_type:
            openai.api_type = config.openai_api_type
            openai.api_version = config.openai_api_version
        self.rpm = int(config.get("RPM", 10))

    async def _achat_completion_stream(self, messages) -> str:
        response = await openai.ChatCompletion.acreate(**self._cons_kwargs(messages), stream=True)

        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        async for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk["choices"][0]["delta"]  # extract the message
            collected_messages.append(chunk_message)  # save the message
            if "content" in chunk_message:
                print(chunk_message["content"], end="")
        print()

        full_reply_content = "".join([m.get("content", "") for m in collected_messages])
        return full_reply_content

    def _cons_kwargs(self, messages) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "n": 1,
            "stop": None,
            "temperature": 0.8,
        }
        return kwargs

    async def _achat_completion(self, messages) -> dict:
        rsp = await self.llm.ChatCompletion.acreate(**self._cons_kwargs(messages))
        return rsp

    def chat_completion(self, messages) -> dict:
        rsp = self.llm.ChatCompletion.create(**self._cons_kwargs(messages))
        return rsp

    def completion(self, messages) -> dict:
        return self.chat_completion(messages)

