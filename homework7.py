import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)


# Завдання 1
# Напишіть додаток з чат ботом по допомозі з вивченням
# англійської мови.
#  Якщо користувач просить перекласти слово або
# фразу, то вивести переклад та приклад використання
# у речені
#  Якщо користувач просить перекласти речення, то
# вивести переклад та пояснення граматики, наприклад
# структура there is/are, пасивна форма дієслова, тощо
st.title("Homework 7:")
st.markdown("Бот-помічник з вивчення англійської мови")

api_key = st.secrets["GEMINI_API_KEY"]
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)

user_query = st.chat_input("Ваше повідомлення")

if "history" not in st.session_state:
    st.session_state.history = [
        SystemMessage(f"""
        Ти -- чатбот-помічник з вивчення англійської мови.
        
        ###ІНСТРУКЦІЇ###
        1. Якщо користувач просить перекласти слово або фразу, то вивести переклад та приклад використання у речені
        2. Якщо користувач просить перекласти речення, то вивести переклад та пояснення граматики, наприклад структура there is/are, пасивна форма дієслова, тощо
        """)
    ]

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
