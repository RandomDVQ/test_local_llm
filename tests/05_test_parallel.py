import json
import time
from concurrent.futures import ThreadPoolExecutor

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


# Three fake tools — each will be executed with an artificial delay inside the test
tools = [
    {
        "type": "function",
        "function": {"name": "tool1", "description": "Fake tool 1", "parameters": {"type": "object"}},
    },
    {
        "type": "function",
        "function": {"name": "tool2", "description": "Fake tool 2", "parameters": {"type": "object"}},
    },
    {
        "type": "function",
        "function": {"name": "tool3", "description": "Fake tool 3", "parameters": {"type": "object"}},
    },
]


def execute_tool(name, arguments):
    # Simulate work — fixed 2 second delay per tool
    time.sleep(2)
    return {"tool": name, "args": arguments}


messages = [
    {"role": "user", "content": "Run multiple tools: tool1, tool2 and tool3."},
]


client = create_client()

print_header("05", "Parallel vs Sequential Execution")

call1 = call_model(client, messages, tools=tools)

response = call1.response

print_response(response)
print_call_stats(call1.stats, "MODEL CALL")

tool_calls = get_tool_calls(response)

if not tool_calls:
    print_status(False)
    raise SystemExit(1)

# Sequential execution
start_seq = time.perf_counter()
seq_results = []
for tc in tool_calls:
    try:
        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
    except Exception:
        args = {}

    res = execute_tool(tc.function.name, args)
    seq_results.append(res)

seq_elapsed = time.perf_counter() - start_seq

# Parallel execution
start_par = time.perf_counter()
with ThreadPoolExecutor(max_workers=len(tool_calls)) as ex:
    futures = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except Exception:
            args = {}

        futures.append(ex.submit(execute_tool, tc.function.name, args))

    par_results = [f.result() for f in futures]

par_elapsed = time.perf_counter() - start_par

speedup = seq_elapsed / par_elapsed if par_elapsed > 0 else float("inf")

print()
print("SEQUENTIAL: ~6.0 sec expected")
print(f"SEQUENTIAL: {seq_elapsed:.1f} sec")
print(f"PARALLEL:   {par_elapsed:.1f} sec")
print(f"SPEEDUP:    {speedup:.1f}x")

passed = True

# Allow some tolerance for timing on busy machines
passed &= check(seq_elapsed >= 5.5, f"SEQUENTIAL >= ~6s ({seq_elapsed:.1f}s)")
passed &= check(par_elapsed <= 3.0, f"PARALLEL <= ~2s ({par_elapsed:.1f}s)")
passed &= check(speedup >= 2.5, f"SPEEDUP >= 2.5x ({speedup:.1f}x)")

print_status(passed)
