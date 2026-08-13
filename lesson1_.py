# Получение api с .env
import dotenv
import os

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
# print(api_key)


# модель
import langchain
from langchain_google_genai import GoogleGenerativeAI


# llm = GoogleGenerativeAI(
#     model="gemini-3.5-flash-lite",
#     api_key=api_key
# )


# использование

# response = llm.invoke("привет")
# print(response)


# параметры

llm = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
    temperature=1.5 # температура
)

response = llm.invoke("придумай короткий рассказ про эльфа 5 предложений")
print(response)

