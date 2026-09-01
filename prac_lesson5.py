import dotenv
import os

from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages
)


dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)

serper_search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)

#Завдання 1
# Напишіть функцію яка перевіряє складність паролю:
#  кількість символів(>8)
#  наявність хоча б однієї літери\цифри\спеціального
# символу
#  наявність літер в різних регістрах
# Функція повертає тест з описом паролю(що добре, а що
# погано)
# На основі цієї функції створіть агента.

def count_char(password:str):
    total_alpha = 0
    total_digit = 0
    total_special = 0
    total_upper = 0
    total_lower = 0

    for char in password:
        if char.isalpha():
            total_alpha += 1
        elif char.isdigit():
            total_digit += 1
        else:
            total_special += 1

        if char.isupper():
            total_upper += 1
        if char.islower():
            total_lower += 1

    return total_alpha, total_digit, total_special, total_upper, total_lower


@tool
def password_check(password: str) -> str:
    """
    перевірка складності пароля
    :param password: str -- пароль
    :return: -- оцінку складності пароля
    """

    print("hi password checker")

    if len(password) < 8:
        return "довжина паролю має бути більше 8 символів"

    total_alpha, total_digit, total_special, total_upper, total_lower = count_char(password)
    print(total_alpha, total_digit, total_special, total_upper, total_lower)

    if total_alpha == 0:
        return "в паролі мають бути літери"

    if total_digit == 0:
        return "в паролі має бути хоча б 1 числе"

    if total_special == 0:
        return "в паролі мають бути спеціальні символи"

    if total_upper == 0:
        return "в паролі хоча б 1 літера має бути з великої"

    if total_lower == 0:
        return "в паролі хоча б 1 літера має бути з маленької"

    else:
        return "пароль чудовий"


agent = create_agent(
    model=llm,
    tools=[password_check]
)

# messages = [
#     SystemMessage("""
#     Ти вічливий чатбот
#     """)
# ]
#
# while True:
#     message = input("You: ")
#
#     if message == "":
#         break
#
#     user_message = HumanMessage(message)
#     messages.append(user_message)
#
#     data = {
#         "messages": messages
#     }
#
#     data = agent.invoke(data)
#
#     messages = data["messages"]
#
#     response = messages[-1]
#     print(response.text)


# Завдання 2
# Напишіть модель показує останні новини про певну
# людину. Якщо користувач вводить не ім’я людини, то вивести
# повідомлення «немає відповідної інформації»
# Скористайтесь DuckDuckGoSearchRun

@tool
def search_news(name: str):
    """
    отримує ім'я та прізвище людини та шукає останні новини про певну людину
    :param name: str -- запит на пошук останніх новин про конкретну людину
    :return: -- результат пошуку
    """

    print("hi search news")

    result = serper_search.results(f"Новини про {name}")
    return result

agent = create_agent(
        model=llm,
        tools=[password_check, search_news]
)

messages = [
    SystemMessage("""
    Ти вічливий чатбот.
    
    у тебе є доступ до інструментів
    * password_check
    * search_news
    
    ###ІНСТРУКЦІЯ###
    1. Якщо користувач вводить не ім’я людини, то вивести повідомлення «немає відповідної інформації»
    """)
]

while True:
    message = input("You: ")

    if message == "":
        break

    user_message = HumanMessage(message)
    messages.append(user_message)
    data = {
        "messages": messages
    }

    data = agent.invoke(data)

    messages = data["messages"]

    response = messages[-1]
    print(response.text)

