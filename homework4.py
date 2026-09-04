import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)


# Завдання 1
# Напишіть чат модель яка підсумовує всю розмову в
# декілька речень. Вкажіть щоб модель зберігала якомога
# більше деталей.
# Використайте цю модель для простого чат бота який
# замість trim_massages використовує модель з підсумуванням.
# Підсумовуйте повідомлення, коли їх більше 4.
# Старі повідомлення треба видалити
# НЕ ВИДАЛЯТИ SystemMessage та не використовувати
# його для підсумування

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)

summarizer = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)


def summarize_messages(msgs: list) -> list:
    system_msg = next((m for m in msgs if isinstance(m, SystemMessage)), None)
    non_system = [m for m in msgs if not isinstance(m, SystemMessage)]

    summary_prompt = SystemMessage(
        "Підсумуй наступну розмову в декілька речень. "
        "Зберігай якомога більше важливих деталей. "
        "Відповідай лише підсумком."
    )

    summary_response = summarizer.invoke([summary_prompt] + non_system)
    summary = AIMessage(f"Підсумок попередньої розмови: {summary_response.content}")

    result = []
    if system_msg:
        result.append(system_msg)
    result.append(summary)
    return result


messages = [
    SystemMessage(
        "Ти корисний асистент, який веде дружню бесіду. "
        "Відповідай детально та зберігай контекст розмови."
    )
]

while True:
    user_text = input("Ви: ")

    if user_text == '':
        break

    human_message = HumanMessage(user_text)

    messages.append(human_message)
    
    non_system_count = 0
    for m in messages:
        if not isinstance(m, SystemMessage):
            non_system_count += 1
    if non_system_count > 4:
        messages = summarize_messages(messages)

    response = llm.invoke(messages)

    print(f"AI: {response.content}")

    messages.append(response)
