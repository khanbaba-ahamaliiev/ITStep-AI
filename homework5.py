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


# Завдання 1
# Напишіть чат бота, з інструментом по рекомендації
# ресторанів.
# Для цього скористайтесь
# GoogleSerperAPIWrapper(type="places")
# Інструмент повинен отримувати запит для пошуку та
# повертати таку інформацію про ресторани:
#  назва
#  посилання на сайт(якщо є)
#  рейтинг

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)
serper_search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key, type="places")

@tool
def search_restaurants(location: str):
    """
    Отримує локацію і шукає ресторани в цій локації
    :param location: -- запит на пошук ресторанів у певній локації
    :return: -- результат пошуку
    """

    print("hi from search restaurants")
    search = serper_search.results(f"найкращі ресторани в {location}")
    places = search.get("places", [])
    result = []
    for n, restaurant in enumerate(places):
        restaurant_name = restaurant["title"]
        restaurant_address = restaurant["address"]
        restaurant_rating = restaurant["rating"]

        restaurant_link = restaurant.get("website", "no link")

        result.append({n: {"name": restaurant_name, "address": restaurant_address, "rating": restaurant_rating, "link": restaurant_link}})
    print(result)
    return result

agent = create_agent(
    model=llm,
    tools=[search_restaurants]
)

messages = [
    SystemMessage("""
    Ти -- вічливий чатбот.
    Твоя задача давати рекомендації ресторанів.
    
    
    ###ІНСТРУКЦІЯ###
    1. Якщо користувач питає про ресторани уточни локацію
    2. Якщо користувач вів локацію для пошуку ресторанів використай search_restaurants
    2. Якщо не має інформації не вигадуй
    """)
]


while True:
    user_message = input("Ви: ")

    if user_message == "":
        break

    user_message = HumanMessage(user_message)
    messages.append(user_message)

    data = {
        "messages": messages,
    }
    data = agent.invoke(data)

    messages = data["messages"]

    response = messages[-1]
    print(response.text)