import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re
import time

from openai import OpenAI

from config import API_KEY, BASE_URL, MODEL


# ============================================================
# CONFIG
# ============================================================

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


# ============================================================
# TOOLS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Получить текущую погоду в указанном городе.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Получить текущее время в указанном городе.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Вычислить математическое выражение.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Математическое выражение",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ============================================================
# FAKE TOOL IMPLEMENTATIONS
# ============================================================

def get_weather(city):
    return {
        "city": city,
        "temperature": 21,
        "condition": "облачно",
        "humidity": 62,
    }


def get_current_time(city):
    return {
        "city": city,
        "time": "12:15:48",
        "timezone": "local test time",
    }


def calculator(expression):
    # Для теста разрешаем только цифры, пробелы и арифметические операции.
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        raise ValueError("Недопустимое математическое выражение")

    result = eval(expression, {"__builtins__": {}}, {})

    return {
        "expression": expression,
        "result": result,
    }


def execute_tool(name, arguments):
    if name == "get_weather":
        return get_weather(**arguments)

    if name == "get_current_time":
        return get_current_time(**arguments)

    if name == "calculator":
        return calculator(**arguments)

    raise ValueError(f"Неизвестный tool: {name}")


# ============================================================
# STATISTICS
# ============================================================

def print_stats(response, elapsed, title):
    usage = response.usage

    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0

    details = getattr(usage, "completion_tokens_details", None)

    reasoning_tokens = 0

    if details:
        reasoning_tokens = (
            getattr(details, "reasoning_tokens", None)
            or 0
        )

    speed = (
        completion_tokens / elapsed
        if elapsed > 0
        else 0
    )

    print()
    print(title)
    print("─" * 55)
    print(f"Время:              {elapsed:.2f} сек")
    print(f"Входных токенов:    {prompt_tokens}")
    print(f"Выходных токенов:   {completion_tokens}")
    print(f"Reasoning токенов:  {reasoning_tokens}")
    print(f"Скорость:           {speed:.2f} ток/сек")
    print("─" * 55)

    return {
        "elapsed": elapsed,
        "prompt": prompt_tokens,
        "completion": completion_tokens,
        "reasoning": reasoning_tokens,
        "speed": speed,
    }


# ============================================================
# FINAL ANSWER CHECK
# ============================================================

def check_final_answer(content):
    text = content.lower()

    # Убираем пробелы различных типов для проверки чисел.
    normalized = (
        text
        .replace(" ", "")
        .replace("\u00a0", "")
        .replace("\u202f", "")
    )

    checks = {}

    # --------------------------------------------------------
    # Город
    # --------------------------------------------------------

    checks["Город Тюмень"] = any(
        word in text
        for word in [
            "тюмень",
            "тюмени",
            "тюменью",
            "тюмен",
        ]
    )

    # --------------------------------------------------------
    # Температура
    # --------------------------------------------------------

    checks["Температура 21°C"] = bool(
        re.search(
            r"\b21\s*(?:°\s*)?(?:c|с|град)",
            text,
            re.IGNORECASE,
        )
    )

    # --------------------------------------------------------
    # Влажность
    # --------------------------------------------------------

    checks["Влажность 62%"] = bool(
        re.search(
            r"\b62\s*(?:%|процент)",
            text,
            re.IGNORECASE,
        )
    )

    # --------------------------------------------------------
    # Время
    # --------------------------------------------------------

    checks["Время HH:MM:SS"] = bool(
        re.search(
            r"\b\d{2}:\d{2}:\d{2}\b",
            text,
        )
    )

    # --------------------------------------------------------
    # Результат вычисления
    # --------------------------------------------------------

    checks["Результат 83810205"] = (
        "83810205" in normalized
    )

    print()
    print("FINAL RESPONSE CHECK")
    print("─" * 55)

    for name, result in checks.items():
        print(
            f"{name:<25}"
            f"{'OK' if result else 'NOT FOUND'}"
        )

    return all(checks.values())


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("TEST 04: Multiple Tools / Parallel Tool Calls")
    print(f"MODEL: {MODEL}")
    print("=" * 60)

    messages = [
        {
            "role": "user",
            "content": (
                "Мне нужна информация по Тюмени: "
                "текущая погода, текущее время. "
                "Также вычисли 12345 * 6789. "
                "Используй подходящие инструменты."
            ),
        }
    ]

    total_start = time.perf_counter()

    # ========================================================
    # CALL #1 — MODEL → TOOLS
    # ========================================================

    print()
    print("[1] MODEL → MULTIPLE TOOLS")

    call1_start = time.perf_counter()

    response1 = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    call1_elapsed = time.perf_counter() - call1_start

    message1 = response1.choices[0].message

    print()
    print("RESPONSE #1:")
    print(f"Finish reason: {response1.choices[0].finish_reason}")
    print(f"Content:       {message1.content!r}")
    print(f"Reasoning:     {getattr(message1, 'reasoning_content', '')!r}")
    print(
        f"Tool calls:    "
        f"{len(message1.tool_calls or [])}"
    )

    stats1 = print_stats(
        response1,
        call1_elapsed,
        "CALL #1 STATISTICS",
    )

    # ========================================================
    # CHECK TOOL CALLS
    # ========================================================

    tool_calls = message1.tool_calls or []

    expected_tools = {
        "calculator",
        "get_current_time",
        "get_weather",
    }

    actual_tools = {
        call.function.name
        for call in tool_calls
    }

    print()
    print("TOOL CALL ANALYSIS")
    print("─" * 55)

    for index, call in enumerate(tool_calls, start=1):

        print()
        print(f"Tool #{index}")
        print(f"ID:          {call.id}")
        print(f"Function:    {call.function.name}")

        raw_arguments = call.function.arguments

        print(f"Arguments:   {raw_arguments}")

        try:
            arguments = json.loads(raw_arguments)

            print("Arguments JSON:  OK")

        except json.JSONDecodeError as exc:

            print("Arguments JSON:  FAILED")
            print(f"Error: {exc}")

            arguments = None

        call._parsed_arguments = arguments

    print()
    print(
        f"Expected tools: "
        f"{sorted(expected_tools)}"
    )

    print(
        f"Actual tools:   "
        f"{sorted(actual_tools)}"
    )

    tools_ok = actual_tools == expected_tools

    print(
        f"Tool selection: "
        f"{'OK' if tools_ok else 'FAILED'}"
    )

    # --------------------------------------------------------
    # Parallel calls
    # --------------------------------------------------------

    parallel_ok = len(tool_calls) == 3

    print()
    print("PARALLEL TOOL CALL CHECK")
    print("─" * 55)
    print(
        f"Multiple calls:   "
        f"{'OK' if parallel_ok else 'FAILED'} "
        f"({len(tool_calls)})"
    )

    if parallel_ok:
        print("Parallel mode:    SUPPORTED")
    else:
        print("Parallel mode:    NOT CONFIRMED")

    # ========================================================
    # TOOL EXECUTION
    # ========================================================

    print()
    print("[TOOL EXECUTION]")
    print("─" * 55)

    messages.append(message1)

    execution_ok = True

    for call in tool_calls:

        name = call.function.name

        try:
            arguments = json.loads(
                call.function.arguments
            )

            print()
            print(f"Executing: {name}")
            print(f"Arguments: {arguments}")

            result = execute_tool(
                name,
                arguments,
            )

            print(f"Result:    {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                }
            )

        except Exception as exc:

            execution_ok = False

            print()
            print(f"TOOL ERROR: {exc}")

    # ========================================================
    # CALL #2 — TOOLS → MODEL
    # ========================================================

    print()
    print("[2] TOOLS → MODEL")

    call2_start = time.perf_counter()

    response2 = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="none",
    )

    call2_elapsed = time.perf_counter() - call2_start

    message2 = response2.choices[0].message

    final_content = message2.content or ""

    print()
    print("RESPONSE #2:")
    print(f"Finish reason: {response2.choices[0].finish_reason}")
    print(f"Content:       {final_content!r}")
    print(
        f"Reasoning:     "
        f"{getattr(message2, 'reasoning_content', '')!r}"
    )
    print(
        f"Tool calls:    "
        f"{len(message2.tool_calls or [])}"
    )

    stats2 = print_stats(
        response2,
        call2_elapsed,
        "CALL #2 STATISTICS",
    )

    # ========================================================
    # FINAL CHECKS
    # ========================================================

    finish_ok = (
        response2.choices[0].finish_reason
        == "stop"
    )

    no_extra_tools = not message2.tool_calls

    content_ok = check_final_answer(
        final_content
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    test_ok = all(
        [
            tools_ok,
            parallel_ok,
            execution_ok,
            finish_ok,
            no_extra_tools,
            content_ok,
        ]
    )

    total_elapsed = (
        time.perf_counter() - total_start
    )

    total_prompt = (
        stats1["prompt"]
        + stats2["prompt"]
    )

    total_completion = (
        stats1["completion"]
        + stats2["completion"]
    )

    total_reasoning = (
        stats1["reasoning"]
        + stats2["reasoning"]
    )

    total_speed = (
        total_completion / total_elapsed
        if total_elapsed > 0
        else 0
    )

    print()
    print("=" * 60)
    print(
        f"STATUS: {'PASS' if test_ok else 'FAIL'}"
    )
    print("=" * 60)

    print()
    print("FINAL CHECKS")
    print("─" * 55)

    print(
        f"Tool selection:       "
        f"{'OK' if tools_ok else 'FAIL'}"
    )

    print(
        f"Parallel tool calls:  "
        f"{'OK' if parallel_ok else 'FAIL'}"
    )

    print(
        f"Tool execution:       "
        f"{'OK' if execution_ok else 'FAIL'}"
    )

    print(
        f"Final finish reason:  "
        f"{'OK' if finish_ok else 'FAIL'}"
    )

    print(
        f"No additional tools:  "
        f"{'OK' if no_extra_tools else 'FAIL'}"
    )

    print(
        f"Final content:         "
        f"{'OK' if content_ok else 'FAIL'}"
    )

    print()
    print("TOTAL STATISTICS")
    print("─" * 55)

    print(
        f"Полное время цикла:  {total_elapsed:.2f} сек"
    )

    print(
        f"Время CALL #1:       "
        f"{stats1['elapsed']:.2f} сек"
    )

    print(
        f"Время CALL #2:       "
        f"{stats2['elapsed']:.2f} сек"
    )

    print()

    print(
        f"CALL #1 input:       "
        f"{stats1['prompt']}"
    )

    print(
        f"CALL #1 output:      "
        f"{stats1['completion']}"
    )

    print(
        f"CALL #1 reasoning:   "
        f"{stats1['reasoning']}"
    )

    print()

    print(
        f"CALL #2 input:       "
        f"{stats2['prompt']}"
    )

    print(
        f"CALL #2 output:      "
        f"{stats2['completion']}"
    )

    print(
        f"CALL #2 reasoning:   "
        f"{stats2['reasoning']}"
    )

    print()

    print(
        f"Всего входных:      {total_prompt}"
    )

    print(
        f"Всего выходных:     {total_completion}"
    )

    print(
        f"Всего reasoning:    {total_reasoning}"
    )

    print(
        f"Общая скорость:     "
        f"{total_speed:.2f} ток/сек"
    )

    print(
        f"API-вызовов:        2"
    )

    print(
        f"Tool calls всего:   "
        f"{len(tool_calls)}"
    )

    print("─" * 55)


if __name__ == "__main__":
    main()