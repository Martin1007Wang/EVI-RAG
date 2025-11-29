from __future__ import annotations

import time
from functools import partial
from typing import Any, Dict, List

import openai
from openai import OpenAI
from vllm import LLM, SamplingParams


def init_llm(
    *,
    model_name: str,
    tensor_parallel_size: int,
    max_seq_len: int,
    max_tokens: int,
    seed: int,
    temperature: float,
    frequency_penalty: float,
):
    if "gpt" not in model_name:
        client = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_seq_len_to_capture=max_seq_len,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )
        return partial(client.chat, sampling_params=sampling_params, use_tqdm=False), False
    client = OpenAI()
    return partial(
        client.chat.completions.create,
        model=model_name,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
    ), True


def run_chat(
    llm,
    messages: List[Dict[str, str]],
    *,
    is_openai: bool,
    max_retries: int = 3,
):
    retries = 0
    while True:
        try:
            output = llm(messages=messages)
            if is_openai:
                return output.choices[0].message.content
            return output[0].outputs[0].text
        except openai.RateLimitError:
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(2**retries)


__all__ = ["init_llm", "run_chat"]
