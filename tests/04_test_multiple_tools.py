import json

from common import (
    create_client,
    call_model,
    get_content,
    get_tool_calls,
    print_header,
    print_response,
    print_call_stats,
    print_total_stats,
    print_status,
    check,
)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current local time for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
]


def execute_tool(name, arguments):
    if name == "get_weather":
        return {
            "city": arguments["city"],
            "temperature": 21,
            "condition": "облачно",
            "humidity": 62,
        }

    if name == "get_current_time":
        return {
            "city": arguments["city"],
            "time": "12:15:48",
            "timezone": "local test time",
        }

    if name == "calculator":
        return {
            "expression": arguments["expression"],
            "result": 83810205,
        }

    raise ValueError(f"Unknown tool: {name}")


messages = [
    {
        "role": "user",
        "content": (
            "Give me the current weather in Tyumen, "
            "the current time in Tyumen, "
            "and calculate 12345 * 6789."
        ),
    }
]


client = create_client()

print_header("04", "Multiple Tool Calls")

# ---------------------------------------------------------
# CALL #1 — MODEL → MULTIPLE TOOLS
# ---------------------------------------------------------

print()
print("[1] MODEL → MULTIPLE TOOLS")

call1 = call_model(
    client,
    messages,
    tools=tools,
)

response1 = call1.response

print_response(response1)
print_call_stats(call1.stats, "CALL #1")

tool_calls = get_tool_calls(response1)

print()
print("TOOL CALL CHECKS")
print("─" * 60)

passed = True

expected_tools = {
    "get_weather",
    "get_current_time",
    "calculator",
}

actual_tools = set()

parsed_calls = []

for index, tool_call in enumerate(tool_calls, start=1):

    name = tool_call.function.name
    actual_tools.add(name)

    try:
        arguments = json.loads(tool_call.function.arguments)
        arguments_ok = True
    except json.JSONDecodeError:
        arguments = {}
        arguments_ok = False

    print()
    print(f"Tool #{index}")
    print(f"Function:  {name}")
    print(f"Arguments: {tool_call.function.arguments}")

    passed &= check(
        arguments_ok,
        f"Tool #{index} arguments JSON",
    )

    parsed_calls.append(
        (tool_call, name, arguments)
    )

passed &= check(
    len(tool_calls) == 3,
    "Exactly 3 tool calls",
)

passed &= check(
    actual_tools == expected_tools,
    "All expected tools selected",
)

# ---------------------------------------------------------
# TOOL EXECUTION
# ---------------------------------------------------------

print()
print("[TOOL EXECUTION]")
print("─" * 60)

tool_results = []

for tool_call, name, arguments in parsed_calls:

    result = execute_tool(
        name,
        arguments,
    )

    print()
    print(f"{name}")
    print(f"Arguments: {arguments}")
    print(f"Result:    {result}")

    tool_results.append(
        (tool_call, result)
    )

# ---------------------------------------------------------
# CALL #2 — TOOLS → MODEL
# ---------------------------------------------------------

messages.append(
    response1.choices[0].message.model_dump()
)

for tool_call, result in tool_results:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(
                result,
                ensure_ascii=False,
            ),
        }
    )

print()
print("[2] TOOLS → MODEL")

call2 = call_model(
    client,
    messages,
)

response2 = call2.response

print_response(response2)
print_call_stats(call2.stats, "CALL #2")

final_content = get_content(response2)

# ---------------------------------------------------------
# FINAL CHECKS
# ---------------------------------------------------------

print()
print("FINAL CHECKS")
print("─" * 60)

passed &= check(
    response2.choices[0].finish_reason == "stop",
    "Final finish reason",
)

passed &= check(
    len(get_tool_calls(response2)) == 0,
    "No additional tools",
)

passed &= check(
    "Тюмень" in final_content,
    "City in final answer",
)

passed &= check(
    "21" in final_content,
    "Temperature in final answer",
)

passed &= check(
    "62" in final_content,
    "Humidity in final answer",
)

passed &= check(
    "12:15:48" in final_content,
    "Time in final answer",
)

passed &= check(
    "83810205" in final_content.replace(" ", ""),
    "Calculation result",
)

print_status(passed)

print_total_stats(
    [call1.stats, call2.stats],
    elapsed=call1.stats.elapsed + call2.stats.elapsed,
)