from openai import OpenAI

from config import API_KEY, BASE_URL, MODEL


client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


def ask_llm(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
    )

    full_answer = ""

    for chunk in response:
        content = chunk.choices[0].delta.content

        if content:
            print(content, end="", flush=True)
            full_answer += content

    print()

    return full_answer