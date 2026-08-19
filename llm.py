import time

from openai import OpenAI

from config import API_KEY, BASE_URL, MODEL


client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


def ask_llm(messages):
    request_start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    first_token_time = None
    full_answer = ""

    input_tokens = None
    output_tokens = None

    for chunk in response:
        # Получаем текст очередного фрагмента
        if chunk.choices:
            content = chunk.choices[0].delta.content

            if content:
                # Время получения первого токена
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                print(content, end="", flush=True)
                full_answer += content

        # Статистика токенов приходит в последнем chunk
        if chunk.usage:
            input_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens

    request_end = time.perf_counter()

    # Полное время от отправки запроса до последнего токена
    total_time = request_end - request_start

    # Время до первого токена
    if first_token_time is not None:
        time_to_first_token = first_token_time - request_start
        generation_time = request_end - first_token_time
    else:
        time_to_first_token = None
        generation_time = None

    # Скорость генерации
    if output_tokens and generation_time and generation_time > 0:
        tokens_per_second = output_tokens / generation_time
    else:
        tokens_per_second = None

    print()
    print()
    print("─" * 45)

    print(f"Время ответа:          {total_time:.2f} сек")

    if time_to_first_token is not None:
        print(f"До первого токена:     {time_to_first_token:.2f} сек")
    else:
        print("До первого токена:     неизвестно")

    if generation_time is not None:
        print(f"Время генерации:       {generation_time:.2f} сек")
    else:
        print("Время генерации:       неизвестно")

    if input_tokens is not None:
        print(f"Входных токенов:       {input_tokens}")
    else:
        print("Входных токенов:       неизвестно")

    if output_tokens is not None:
        print(f"Выходных токенов:      {output_tokens}")
    else:
        print("Выходных токенов:      неизвестно")

    if tokens_per_second is not None:
        print(f"Скорость генерации:    {tokens_per_second:.2f} ток/сек")
    else:
        print("Скорость генерации:    неизвестно")

    print("─" * 45)

    return full_answer