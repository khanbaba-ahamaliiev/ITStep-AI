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

# инструменты
@tool
def product(a: float, b: float) -> float:
    """
    Множить 2 дійсний числа між собою
    :param a: float -- перше число
    :param b: float -- друге число
    :return: float -- добуток чисел
    """
    print("hi from product tool")
    return a * b

serper_search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)

@tool
def google_search(query: str) -> str:
    """
    шукає інформацію в інтернеті

    :param query:str -- запит у пошуковик
    :return: результат пошуку
    """
    print("hi from google search")
    result = serper_search.run(query)
    return result

# создание агента
agent = create_agent(
    model=llm,
    tools=[product, google_search] # список инструментов
)


# Написать системный промпт
# вместе с ним создаем историю сообщений
messages = [
    SystemMessage("""
    Ты -- вежливый чатбот
    
    """)
]

# цикл с общением
while True:
    user_message = input("Ви: ")

    if user_message == "":
        break

    user_message = HumanMessage(user_message)
    messages.append(user_message)

    # агент сам добавляет сообщение в историю и возвращает ее

    # агенту нужна передавать словарь с ключем messages

    data = {
        "messages": messages,
    }
    data = agent.invoke(data)
    # агент так же возвращает словарь

    # достаем новую историю
    messages = data["messages"]

    # ответ модели -- поледнее сообщение в моделе
    response = messages[-1]
    print(response.text)




