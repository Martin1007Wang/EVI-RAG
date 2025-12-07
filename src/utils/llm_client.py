from __future__ import annotations

import time
import json
import time
from functools import partial
from typing import Any, Callable, Dict, List, Tuple
from urllib import error, request

import openai
from openai import OpenAI
from vllm import LLM, SamplingParams


LLMCallable = Callable[[List[Dict[str, str]]], Any]


def init_llm(
    *,
    model_name: str,
    tensor_parallel_size: int,
    max_seq_len: int,
    max_tokens: int,
    seed: int,
    temperature: float,
    frequency_penalty: float,
    backend: str = "auto",
    ollama_base_url: str = "http://localhost:11434",
    ollama_timeout: float = 120.0,
) -> Tuple[LLMCallable, bool]:
    """
    Initialize an LLM client.

    Returns:
        (callable, is_openai)
    """
    resolved_backend = backend
    if backend == "auto":
        resolved_backend = "openai" if "gpt" in model_name else "vllm"

    if resolved_backend == "ollama":
        model = model_name.split(":", maxsplit=1)[-1] if model_name.startswith("ollama:") else model_name

        def _ollama_chat(messages: List[Dict[str, str]]) -> str:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "frequency_penalty": frequency_penalty,
                },
            }
            req = request.Request(
                url=f"{ollama_base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=ollama_timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                raise RuntimeError(f"Ollama HTTP error: {exc.code} {exc.reason}") from exc
            except error.URLError as exc:
                raise RuntimeError(f"Ollama connection failed: {exc.reason}") from exc

            message = (data.get("message") or {}).get("content")
            if message is None:
                raise ValueError("Unexpected Ollama response: missing message.content")
            return str(message)

        return _ollama_chat, False

    if resolved_backend == "vllm":
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

    # openai backend
    client = OpenAI()
    return (
        partial(
            client.chat.completions.create,
            model=model_name,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        True,
    )


def run_chat(
    llm: LLMCallable,
    messages: List[Dict[str, str]],
    *,
    is_openai: bool,
    max_retries: int = 3,
) -> str:
    """Execute a chat completion with basic retry for OpenAI rate limits."""
    retries = 0
    while True:
        try:
            output = llm(messages=messages)
            if isinstance(output, str):
                return output
            if is_openai:
                return output.choices[0].message.content
            return output[0].outputs[0].text
        except openai.RateLimitError:
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(2**retries)


__all__ = ["init_llm", "run_chat", "LLMCallable"]
