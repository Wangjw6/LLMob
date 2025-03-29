#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Abstract GPT class
@File    : base_gpt_api.py
"""
from abc import abstractmethod
from engine.llm_configs.base_chatbot import BaseChatbot


class BaseGPTAPI(BaseChatbot):
    """GPT API abstract class, requiring all inheritors to provide a series of standard capabilities"""
    system_prompt = 'Urban mobility generation.'

    def _user_msg(self, msg: str):
        return {"role": "user", "content": msg}

    def _assistant_msg(self, msg: str) :
        return {"role": "assistant", "content": msg}

    def _system_msg(self, msg: str) :
        return {"role": "system", "content": msg}

    def _system_msgs(self, msgs): # -> list[dict[str, str]]:
        return [self._system_msg(msg) for msg in msgs]

    def _default_system_msg(self):
        return self._system_msg(self.system_prompt)

    def ask(self, msg: str) -> str:
        message = [self._default_system_msg(), self._user_msg(msg)]
        rsp = self.completion(message)
        return self.get_choice_text(rsp)

    def _extract_assistant_rsp(self, context):
        return "\n".join([i["content"] for i in context if i["role"] == "assistant"])

    def ask_batch(self, msgs: list) -> str:
        context = []
        for msg in msgs:
            umsg = self._user_msg(msg)
            context.append(umsg)
            rsp = self.completion(context)
            rsp_text = self.get_choice_text(rsp)
            context.append(self._assistant_msg(rsp_text))
        return self._extract_assistant_rsp(context)

    def ask_code(self, msgs) -> str:
        """FIXME: No code segment filtering has been done here, and all results are actually displayed"""
        rsp_text = self.ask_batch(msgs)
        return rsp_text

    @abstractmethod
    def completion(self, messages):
        """All GPTAPIs are required to provide the standard OpenAI completion interface
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello, show me python hello world code"},
            # {"role": "assistant", "content": ...}, # If there is an answer in the history, also include it
        ]
        """

    def get_choice_text(self, rsp: dict) -> str:
        """Required to provide the first text of choice"""
        return rsp.get("choices")[0]["message"]["content"]

    def messages_to_prompt(self, messages):
        """[{"role": "user", "content": msg}] to user: <msg> etc."""
        return '\n'.join([f"{i['role']}: {i['content']}" for i in messages])

    def messages_to_dict(self, messages):
        """objects to [{"role": "user", "content": msg}] etc."""
        return [i.to_dict() for i in messages]
