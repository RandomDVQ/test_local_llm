from openai import OpenAI


MODEL = "qwen/qwen3.6-35b-a3b"
BASE_URL = "http://localhost:1234/v1"


client = OpenAI(
    base_url=BASE_URL,
    api_key="lm-studio",
)


def ask_llm(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    return response.choices[0].message.content


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
            answer = ask_llm(messages)

            print()
            print(f"Qwen: {answer}")
            print()

            messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as error:
            print()
            print(f"Ошибка: {error}")
            print()


if __name__ == "__main__":
    main()