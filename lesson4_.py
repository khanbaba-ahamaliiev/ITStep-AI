import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages
)


dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)


# История
messages = [
    # список для всех сообщений в чате
    # в начале всегда идет SystemMessage -- c инструкциями для чатбота
    SystemMessage("""
    Ты -- вежливый чатбот.
    Твоя задача поддерживать общении с пользователем
    
    ###ИНСТРУКЦИИ###
    1. Ответы должны быть короткими(до 2 предложений)
    """),
    # все следующие элементы списка это запросы от пользователя и ответы от llm
    HumanMessage("Привет"),
    AIMessage("привет как дела"),
    HumanMessage("Порекомендуй интересный фильм"),
]

# response = llm.invoke(messages)
# print(type(response))
# print(response.content)

# ЧАТБОТ

# Создание списка для истории сообщений
messages = [
    SystemMessage("""
    Ты -- вежливый чатбот.
    Твоя задача поддерживать общении с пользователем

    ###ИНСТРУКЦИИ###
    1. Ответы должны быть короткими(до 2 предложений)
    """)
]

# очищення истории от сообщений
trimmer = trim_messages(
    strategy = 'last', # -- оставить последние сообщения
    token_counter = len,  # -- считаем количество сообщений
    max_tokens = 5,  # --оставить максимум 5 сообщений
    start_on = 'human',  # -- история всегда начинается HumanMessage
    end_on = 'human', # -- история всегда заканчивается HumanMessage
    include_system = True # -- SystemMessage не трогать
)

while True:
    # получить сообщение от пользователя
    user_text = input("Вы: ")

    if user_text == '':
        break

    # создать HumanMessage
    human_message = HumanMessage(user_text)

    # добавить сообщение в историю
    messages.append(human_message)

    # очищение истории
    messages = trimmer.invoke(messages)

    # получить ответ модели
    response = llm.invoke(messages)

    # вывести на экран ответ
    print(f"AI: {response.content[0]['text']}")

    # добавить response в историю общению
    messages.append(response)





