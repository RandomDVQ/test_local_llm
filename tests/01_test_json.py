import json

from common import (
    create_client,
    call_model,
    get_content,
    get_reasoning,
    print_header,
    print_response,
    print_call_stats,
    print_status,
    check,
)
from config import MODEL


TEST_DATA = {
    "people": [
        {
            "name": "Алексей Иванов",
            "age": 34,
            "profession": "Инженер-программист",
            "active": True,
        },
        {
            "name": "Мария Петрова",
            "age": 28,
            "profession": "Дизайнер интерьеров",
            "active": True,
        },
        {
            "name": "Дмитрий Сидоров",
            "age": 52,
            "profession": "Менеджер по продажам",
            "active": False,
        },
    ]
}


messages = [
    {
        "role": "system",
        "content": (
            "Return ONLY valid JSON. "
            "Do not use Markdown. "
            "Do not add explanations."
        ),
    },
    {
        "role": "user",
        "content": (
            "Create JSON describing exactly these three people:\n"
            "Алексей Иванов, 34, Инженер-программист, active=true\n"
            "Мария Петрова, 28, Дизайнер интерьеров, active=true\n"
            "Дмитрий Сидоров, 52, Менеджер по продажам, active=false\n\n"
            "Required structure:\n"
            '{"people":[{"name":"string","age":0,'
            '"profession":"string","active":true}]}'
        ),
    },
]


client = create_client()

print_header("01", "Structured JSON")

result = call_model(
    client,
    messages,
)

response = result.response

print_response(response)
print_call_stats(result.stats)

content = get_content(response)
reasoning = get_reasoning(response)

json_source = None
data = None

for source_name, source in (
    ("content", content),
    ("reasoning_content", reasoning),
):
    if not source.strip():
        continue

    try:
        data = json.loads(source)
        json_source = source_name
        break
    except json.JSONDecodeError:
        pass


print()
print("CHECKS")
print("─" * 60)

passed = True

passed &= check(
    json_source is not None,
    f"JSON source: {json_source or 'NOT FOUND'}",
)

if data is not None:
    passed &= check(
        isinstance(data, dict),
        "Root object",
    )

    passed &= check(
        "people" in data,
        "people field",
    )

    passed &= check(
        isinstance(data.get("people"), list),
        "people is array",
    )

    passed &= check(
        len(data.get("people", [])) == 3,
        "Exactly 3 people",
    )

    passed &= check(
        data == TEST_DATA,
        "Data matches expected",
    )

print_status(passed)