from config import MODEL
from llm import ask_llm


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

            messages.pop()


if __name__ == "__main__":
    main()