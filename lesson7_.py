import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# на фоне работает цикл while True


# заголовок сайта
st.title("Сайт для чатбота")

# Обычный текст
st.markdown("Сегодня последнее занятие по работе с чатботами")

# # история сообщений
# history = []
#
# # получить уведомление от пользователя
# user_text = st.chat_input("Спросите бота что-нибудь")
#
# # глобальная память в streamlit
#
# # если истории нету создаем пустой список
# if "history" not in st.session_state:
#     st.session_state.history = [SystemMessage("""
#     Ты -- вежливый чатбот.
#     Твоя задача поддерживать общении с пользователем
#
#     ###ИНСТРУКЦИИ###
#     1. Ответы должны быть короткими(до 2 предложений)
#     """)]
#
# st.session_state.history.append(user_text)
# print(st.session_state.history)
#
# # вывод результата
# st.markdown(f"вы сказали: {user_text}")
#
#



# чат бот
api_key = st.secrets["GEMINI_API_KEY"]
# print(api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)

user_query = st.chat_input("Ваше повідомлення")

if "history" not in st.session_state:
    st.session_state.history = [
        SystemMessage("""
        Ти -- ввічливий чатбот
        Твоя задача підтримувати спілкування з користувач
        """)
    ]


if user_query:
    # переволимо повідомлення в HumanMessage
    human_message = HumanMessage(user_query)

    messages = st.session_state['history']

    # добавляємо до історії повідомлень
    messages.append(human_message)

    # запускаємо модель
    response = llm.invoke(messages)

    # response -- AIMessage
    # добавляємо до історії повідомлень
    messages.append(response)

# вивести всю історію спілкування
for message in messages:
    # пропускаємо SystemMessage
    if isinstance(message, SystemMessage):
        continue

    # отримати вміст
    text = message.content

    # отримати роль
    if isinstance(message, HumanMessage):
        role = "human"
    else:
        role = 'ai'

    # вивести повідомлення з підписом
    with st.chat_message(role):
        st.markdown(message.text)