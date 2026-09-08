import streamlit as st


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

st.title("Практична робота по чатботу")
# Завдання 1
# Напишіть додаток, який симулює спілкування з певною
# відомою людиною.
# З ким саме спілкуватись вводить користувач через
# st.text_input()


api_key = st.secrets["GEMINI_API_KEY"]
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)

user_query = st.chat_input("Ваше повідомлення")
bot_role = st.text_input("Введіть роль для бота")

if "history" not in st.session_state:
    st.session_state.history = [
        SystemMessage(f"""
        Ти -- чатбот що симулює спілкування з певною відомою людиною.
        """)
    ]

if bot_role:
    st.session_state['history'].append(SystemMessage(f"Твоя роль -- це {bot_role}"))

if user_query:
    human_message = HumanMessage(user_query)

    messages = st.session_state['history']

    messages.append(human_message)

    response = llm.invoke(messages)

    messages.append(response)

    for message in messages:
        if isinstance(message, SystemMessage):
            continue

        if isinstance(message, HumanMessage):
            role = "human"
        else:
            role = 'ai'

        with st.chat_message(role):
            st.markdown(message.text)


# Завдання 2
# Напишіть додаток, який симулює проходження
# співбесіди на певну посаду.
# Користувач може ввести назву посади через st.text_input()
# Користувач може ввести опис вакансії через
# st.file_uploader()
# Далі починається чат з спілкуванням




# Завдання 3
# Напишіть чат бота з доступом до інтернету
