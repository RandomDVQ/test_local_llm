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


calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                }
            },
            "required": ["expression"],
        },
    },
}


def calculator(expression: str) -> int:
    return 12345 * 6789


messages = [
    {
        "role": "user",
        "content": "Calculate 12345 * 6789 and give me the result.",
    }
]


client = create_client()

print_header("03", "Tool Execution / Agent Loop")

# ---------------------------------------------------------
# CALL #1 — MODEL → TOOL
# ---------------------------------------------------------

print()
print("[1] MODEL → TOOL")

call1 = call_model(
    client,
    messages,
    tools=[calculator_tool],
)

response1 = call1.response

print_response(response1)
print_call_stats(call1.stats, "CALL #1")

tool_calls = get_tool_calls(response1)

passed = True

print()
print("CALL #1 CHECKS")
print("─" * 60)

passed &= check(
    len(tool_calls) == 1,
    "Exactly 1 tool call",
)

if not tool_calls:
    print_status(False)
    raise SystemExit(1)

tool_call = tool_calls[0]

passed &= check(
    tool_call.function.name == "calculator",
    "Function name",
)

try:
    arguments = json.loads(tool_call.function.arguments)

    passed &= check(
        arguments.get("expression") == "12345 * 6789",
        "Arguments",
    )

except json.JSONDecodeError:
    passed &= check(False, "Arguments JSON")
    arguments = {}

# ---------------------------------------------------------
# TOOL EXECUTION
# ---------------------------------------------------------

expression = arguments.get("expression")

print()
print("[TOOL EXECUTION]")
print(f"Expression: {expression}")

tool_result = calculator(expression)

print(f"Result:     {tool_result}")

passed &= check(
    tool_result == 83810205,
    "Tool result",
)

# ---------------------------------------------------------
# CALL #2 — TOOL → MODEL
# ---------------------------------------------------------

print()
print("[2] TOOL → MODEL")

messages.append(response1.choices[0].message.model_dump())

messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(
            {"result": tool_result},
            ensure_ascii=False,
        ),
    }
)

call2 = call_model(
    client,
    messages,
)

response2 = call2.response

print_response(response2)
print_call_stats(call2.stats, "CALL #2")

final_content = get_content(response2)

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
    bool(final_content.strip()),
    "Final content",
)

passed &= check(
    "83810205" in final_content.replace(" ", ""),
    "Result in final answer",
)

print_status(passed)

print_total_stats(
    [call1.stats, call2.stats],
    elapsed=call1.stats.elapsed + call2.stats.elapsed,
)