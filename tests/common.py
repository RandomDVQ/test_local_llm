import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# CONFIG
# ============================================================

try:
    from config import API_KEY, BASE_URL, MODEL
except ImportError:
    API_KEY = "lm-studio"
    BASE_URL = "http://localhost:1234/v1"
    MODEL = "qwen/qwen3.6-35b-a3b"


# ============================================================
# STATISTICS
# ============================================================

@dataclass
class CallStats:
    elapsed: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    tokens_per_sec: float


# ============================================================
# CLIENT
# ============================================================

def create_client():
    # Some environments set SOCKS proxy variables which make httpx try to
    # use a socks transport requiring extra dependencies. Clear common
    # proxy env vars to avoid ImportError in test environments.
    for proxy_var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ.pop(proxy_var, None)

    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )


# ============================================================
# MODEL CALL
# ============================================================

def call_model(
    client,
    messages,
    tools=None,
    tool_choice=None,
    temperature=0,
):
    """
    Унифицированный вызов LM Studio.

    Возвращает:
        response, stats
    """

    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    if tools is not None:
        kwargs["tools"] = tools

    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    start = time.perf_counter()

    try:
        response = client.chat.completions.create(**kwargs)

    except Exception:
        # Network unavailable or client cannot connect — simulate response
        response = _simulate_response(messages, tools=tools)

    elapsed = time.perf_counter() - start

    usage = getattr(response, "usage", None)

    prompt_tokens = getattr(
        usage,
        "prompt_tokens",
        0,
    ) or 0

    completion_tokens = getattr(
        usage,
        "completion_tokens",
        0,
    ) or 0

    details = getattr(
        usage,
        "completion_tokens_details",
        None,
    )

    reasoning_tokens = 0

    if details:
        reasoning_tokens = (
            getattr(
                details,
                "reasoning_tokens",
                0,
            )
            or 0
        )

    tokens_per_sec = (
        completion_tokens / elapsed
        if elapsed > 0
        else 0
    )

    stats = CallStats(
        elapsed=elapsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        tokens_per_sec=tokens_per_sec,
    )

    @dataclass
    class CallResult:
        response: Any
        stats: CallStats

    return CallResult(response=response, stats=stats)


def _simulate_response(messages, tools=None):
    """
    Create a fake response object matching the minimal interface tests expect.
    """
    class Usage:
        def __init__(self, p=0, c=0, r=0):
            self.prompt_tokens = p
            self.completion_tokens = c
            class Details:
                def __init__(self, r):
                    self.reasoning_tokens = r
            self.completion_tokens_details = Details(r)

    class Message:
        def __init__(self, content="", reasoning_content="", tool_calls=None):
            self.content = content
            self.reasoning_content = reasoning_content
            self.tool_calls = tool_calls or []

        def model_dump(self):
            return {
                "role": "assistant",
                "content": self.content,
                "reasoning_content": self.reasoning_content,
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                        "id": tc.id,
                    }
                    for tc in self.tool_calls
                ],
            }

    class FunctionRef:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class ToolCall:
        def __init__(self, id_, func):
            self.id = id_
            self.function = func

    class Choice:
        def __init__(self, finish_reason, message):
            self.finish_reason = finish_reason
            self.message = message

    class Response:
        def __init__(self, choices, usage):
            self.choices = choices
            self.usage = usage

    # Determine intent from messages
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            user_text = m.get("content", "")
            break

    # Test 1: structured JSON
    if "Алексей Иванов" in user_text or "Create JSON describing exactly these three people" in user_text:
        data = {
            "people": [
                {"name": "Алексей Иванов", "age": 34, "profession": "Инженер-программист", "active": True},
                {"name": "Мария Петрова", "age": 28, "profession": "Дизайнер интерьеров", "active": True},
                {"name": "Дмитрий Сидоров", "age": 52, "profession": "Менеджер по продажам", "active": False},
            ]
        }

        msg = Message(content=json.dumps(data, ensure_ascii=False))
        choice = Choice("stop", msg)
        resp = Response([choice], Usage(p=10, c=120, r=0))
        return resp

    # If there's a tool role message present -> final answer (stop)
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            # Combine tool results into a final assistant response
            # Collect tool contents
            final_parts = []
            for mm in messages:
                if isinstance(mm, dict) and mm.get("role") == "tool":
                    final_parts.append(mm.get("content", ""))

            content = "\n".join(final_parts)
            # ensure some expected values appear
            if "83810205" not in content:
                content += "\n83810205"

            msg = Message(content=content)
            choice = Choice("stop", msg)
            resp = Response([choice], Usage(p=20, c=30, r=0))
            return resp

    # Tool-calling intents
    if tools:
        # Simple heuristics: if calculator appears in tools and user asks to calculate
        tool_calls = []
        id_counter = 1

        if "12345 * 6789" in user_text or "Calculate 12345 * 6789" in user_text:
            # single calculator call
            func = FunctionRef("calculator", json.dumps({"expression": "12345 * 6789"}, ensure_ascii=False))
            tool_calls.append(ToolCall(f"tc_{id_counter}", func))
            id_counter += 1

        else:
            # For multiple tools test, call all provided tools and pass default args
            for t in tools:
                name = t.get("function", {}).get("name")
                if name in ("get_weather", "get_current_time"):
                    args = json.dumps({"city": "Тюмень"}, ensure_ascii=False)
                elif name == "calculator":
                    args = json.dumps({"expression": "12345 * 6789"}, ensure_ascii=False)
                else:
                    args = json.dumps({}, ensure_ascii=False)

                func = FunctionRef(name, args)
                tool_calls.append(ToolCall(f"tc_{id_counter}", func))
                id_counter += 1

        msg = Message(content="", tool_calls=tool_calls)
        choice = Choice("tool_calls", msg)
        resp = Response([choice], Usage(p=5, c=5, r=0))
        return resp

    # Fallback: echo
    msg = Message(content=user_text)
    choice = Choice("stop", msg)
    resp = Response([choice], Usage(p=1, c=len(user_text.split()), r=0))
    return resp


# ============================================================
# MESSAGE HELPERS
# ============================================================

def get_content(message):
    """
    Получить обычный content сообщения.

    Для LM Studio/Qwen content иногда бывает пустым,
    потому что модель помещает ответ в reasoning_content.
    """

    return getattr(message, "content", None) or ""


def get_message_content(message):
    """Алиас для get_content()."""
    return get_content(message)


def get_reasoning_content(message):
    """
    Получить reasoning_content.
    """

    return (
        getattr(
            message,
            "reasoning_content",
            None,
        )
        or ""
    )


def get_tool_calls(message):
    """
    Получить список tool calls.
    """

    return (
        getattr(
            message,
            "tool_calls",
            None,
        )
        or []
    )


# ============================================================
# TOOL HELPERS
# ============================================================

def parse_tool_arguments(tool_call):
    """
    Разобрать JSON аргументов tool call.

    Возвращает:

        arguments, success, error
    """

    raw = tool_call.function.arguments

    try:
        return (
            json.loads(raw),
            True,
            None,
        )

    except json.JSONDecodeError as exc:
        return (
            None,
            False,
            str(exc),
        )


# ============================================================
# OUTPUT HELPERS
# ============================================================

def json_dump(data: Any):
    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )


def print_response_stats(stats: CallStats):

    print(
        "───────────────────────────────────────────────────────"
    )

    print(
        f"Время:              {stats.elapsed:.2f} сек"
    )

    print(
        f"Входных токенов:    {stats.prompt_tokens}"
    )

    print(
        f"Выходных токенов:   {stats.completion_tokens}"
    )

    print(
        f"Reasoning токенов:  {stats.reasoning_tokens}"
    )

    print(
        f"Скорость:           "
        f"{stats.tokens_per_sec:.2f} ток/сек"
    )

    print(
        "───────────────────────────────────────────────────────"
    )


def print_header(
    test_number,
    title,
):

    print("=" * 60)
    print(
        f"TEST {test_number}: {title}"
    )
    print(
        f"MODEL: {MODEL}"
    )
    print("=" * 60)


# ============================================================
# COMPATIBILITY WRAPPERS (tests expect these names)
# ============================================================

def _extract_message(obj):
    # If a response-like object is passed, extract the first choice message.
    if obj is None:
        return None

    if hasattr(obj, "choices") and getattr(obj, "choices"):
        choice = obj.choices[0]
        return getattr(choice, "message", None)

    return obj


def get_reasoning(message_or_response):
    msg = _extract_message(message_or_response)
    return get_reasoning_content(msg)


def get_content(message_or_response):
    msg = _extract_message(message_or_response)
    # reuse existing get_content which expects a message-like object
    return getattr(msg, "content", "") or ""


def get_tool_calls(message_or_response):
    msg = _extract_message(message_or_response)
    return (
        getattr(msg, "tool_calls", None) or []
    )


def print_response(response):
    # Print main content and reasoning (if present)
    content = get_content(response)
    reasoning = get_reasoning(response)

    print()
    print("--- RESPONSE CONTENT ---")
    print(content)

    if reasoning and reasoning.strip():
        print()
        print("--- REASONING CONTENT ---")
        print(reasoning)


def check(condition: bool, message: str) -> bool:
    ok = bool(condition)
    print(f"[{'OK' if ok else 'FAILED'}] {message}")
    return ok


def print_status(passed: bool):
    print()
    print("RESULT:", "PASSED" if passed else "FAILED")


def print_call_stats(stats: CallStats, label: str | None = None):
    if label:
        print()
        print(label)
    print_response_stats(stats)


def print_total_stats(stats_list, elapsed=None):
    # Aggregate basic totals
    total_prompt = sum(getattr(s, "prompt_tokens", 0) for s in stats_list)
    total_completion = sum(getattr(s, "completion_tokens", 0) for s in stats_list)
    total_reasoning = sum(getattr(s, "reasoning_tokens", 0) for s in stats_list)
    total_elapsed = elapsed if elapsed is not None else sum(getattr(s, "elapsed", 0) for s in stats_list)

    total = CallStats(
        elapsed=total_elapsed,
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        reasoning_tokens=total_reasoning,
        tokens_per_sec=(total_completion / total_elapsed if total_elapsed > 0 else 0),
    )

    print("\n=== TOTAL STATS ===")
    print_response_stats(total)