import json
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from config import API_KEY, BASE_URL, MODEL


@dataclass
class CallStats:
    elapsed: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed <= 0:
            return 0.0
        return self.completion_tokens / self.elapsed


@dataclass
class ModelCall:
    response: Any
    stats: CallStats


def create_client() -> OpenAI:
    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )


def call_model(
    client: OpenAI,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0,
) -> ModelCall:

    started = time.perf_counter()

    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(**kwargs)

    elapsed = time.perf_counter() - started

    usage = response.usage

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    completion_details = getattr(
        usage,
        "completion_tokens_details",
        None,
    )

    reasoning_tokens = 0

    if completion_details:
        reasoning_tokens = (
            getattr(completion_details, "reasoning_tokens", 0) or 0
        )

    stats = CallStats(
        elapsed=elapsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
    )

    return ModelCall(
        response=response,
        stats=stats,
    )


def print_header(test_number: str, title: str):
    print("=" * 60)
    print(f"TEST {test_number}: {title}")
    print(f"MODEL: {MODEL}")
    print("=" * 60)


def print_call_stats(stats: CallStats, title: str = "CALL"):
    print()
    print("─" * 60)
    print(title)
    print("─" * 60)
    print(f"Time:              {stats.elapsed:.2f} sec")
    print(f"Input tokens:      {stats.prompt_tokens}")
    print(f"Output tokens:     {stats.completion_tokens}")
    print(f"Reasoning tokens:  {stats.reasoning_tokens}")
    print(f"Generation speed:  {stats.tokens_per_second:.2f} tok/sec")
    print("─" * 60)


def print_total_stats(calls: list[CallStats], elapsed: float | None = None):
    if elapsed is None:
        elapsed = sum(call.elapsed for call in calls)

    input_tokens = sum(call.prompt_tokens for call in calls)
    output_tokens = sum(call.completion_tokens for call in calls)
    reasoning_tokens = sum(call.reasoning_tokens for call in calls)

    speed = output_tokens / elapsed if elapsed > 0 else 0

    print()
    print("TOTAL")
    print("─" * 60)
    print(f"API calls:         {len(calls)}")
    print(f"Time:              {elapsed:.2f} sec")
    print(f"Input tokens:      {input_tokens}")
    print(f"Output tokens:     {output_tokens}")
    print(f"Reasoning tokens:  {reasoning_tokens}")
    print(f"Overall speed:     {speed:.2f} tok/sec")
    print("─" * 60)


def get_message(response):
    return response.choices[0].message


def get_content(response) -> str:
    return get_message(response).content or ""


def get_reasoning(response) -> str:
    return getattr(
        get_message(response),
        "reasoning_content",
        None,
    ) or ""


def get_tool_calls(response):
    return get_message(response).tool_calls or []


def print_response(response):
    message = get_message(response)

    print()
    print("RESPONSE:")
    print(f"Finish reason:       {response.choices[0].finish_reason}")
    print(f"Content:             {message.content!r}")
    print(f"Reasoning content:   {get_reasoning(response)!r}")
    print(f"Tool calls:          {len(get_tool_calls(response))}")


def parse_json(text: str):
    return json.loads(text)


def check(condition: bool, description: str) -> bool:
    status = "OK" if condition else "FAIL"
    print(f"{description:<30} {status}")
    return condition


def print_status(passed: bool):
    print()
    print("=" * 60)
    print(f"STATUS: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)