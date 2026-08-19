import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")


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


def main():
    print("=" * 60)
    print("             TEST LOCAL LLM")
    print(f"             Model: {MODEL}")
    print("=" * 60)
    print("Введите сообщение. Для выхода: exit")
    print()

    messages = [
        {
            "role": "system",
            "content": (
                "Ты полезный и умный ассистент. "
                "Отвечай на русском языке, если пользователь пишет по-русски."
            ),
        }
    ]

    while True:
        user_message = input("Ты: ")

        if user_message.lower() in ("exit", "quit", "выход"):
            print("До встречи!")
            break

        if not user_message.strip():
            continue

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        try:
            print()
            print("Qwen: ", end="", flush=True)

            answer = ask_llm(messages)

            messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            print()

        except Exception as error:
            print()
            print(f"Ошибка: {error}")
            print()

            # Удаляем последний запрос пользователя,
            # если модель не смогла ответить.
            messages.pop()


if __name__ == "__main__":
    main()