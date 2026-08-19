import json

from common import (
    create_client,
    call_model,
    get_tool_calls,
    print_header,
    print_response,
    print_call_stats,
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
                    "description": "Mathematical expression to calculate.",
                }
            },
            "required": ["expression"],
        },
    },
}


messages = [
    {
        "role": "user",
        "content": "Calculate 12345 * 6789.",
    }
]


client = create_client()

print_header("02", "Tool Call")

result = call_model(
    client,
    messages,
    tools=[calculator_tool],
)

response = result.response

print_response(response)
print_call_stats(result.stats)

tool_calls = get_tool_calls(response)

print()
print("CHECKS")
print("─" * 60)

passed = True

passed &= check(
    response.choices[0].finish_reason == "tool_calls",
    "Finish reason",
)

passed &= check(
    len(tool_calls) == 1,
    "Exactly 1 tool call",
)

if tool_calls:
    tool = tool_calls[0]

    passed &= check(
        tool.function.name == "calculator",
        "Function name",
    )

    try:
        arguments = json.loads(tool.function.arguments)

        passed &= check(
            isinstance(arguments, dict),
            "Arguments JSON",
        )

        passed &= check(
            arguments.get("expression") == "12345 * 6789",
            "Arguments value",
        )

    except json.JSONDecodeError:
        passed &= check(
            False,
            "Arguments JSON",
        )

print_status(passed)