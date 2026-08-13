import dotenv
import os
import langchain
from langchain_google_genai import GoogleGenerativeAI


# Завдання 1
# Підключіть модель LLM за допомогою свого API key.
# Попросіть модель згенерувати:
# ● відповідь на питання у вигляді одного
# слова(наприклад яка столиця Франції?)
# ● код python
# ● коротку історію
# Підберіть параметри креативності та довжини
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# llm = GoogleGenerativeAI(
#     model='gemini-3.5-flash-lite',
#     api_key=api_key,
#     temperature=1.5,
#     top_k=10
# )

# response = llm.invoke("яка столиця Франції?(1 слово)")
# print(response)

# response = llm.invoke("згенеруй найпростіший пайтон код з циклом")
# print(response)

# response = llm.invoke("напиши коротку історію про починаючого програміста 3 речення")
# print(response)

# Завдання 2
# Прочитайте файл data\lesson9\rules.txt з правилами
# користування атракціону. Напишіть програму яка отримує
# від користувачі питання та дає відповідь на нього виходячи
# з текстового файлу.
# Для цього об’єднайте правила користування з питанням
# користувача.
# Користувач задає питання поки не введе порожній рядок.
# Змініть файл rules.txt, щоб переконатись що модель
# дійсно його читає.

# llm = GoogleGenerativeAI(
#     model='gemini-3.5-flash-lite',
#     api_key=api_key,
#     temperature=0.1,
# )
#
# with open("data/lesson9/rules.txt", "r", encoding="utf-8") as f:
#     rules = f.read()
#
# while True:
#     question = input("Question: ")
#
#     if question == '' or question == ' ':
#         break
#
#     response = llm.invoke(f"""
#     Ти працівник парку атракціонів та відповідаєш на
#     питання клієнтів на основі правил: {rules}.
#     Відповідай тільки те що написано в правилах чітко і ясно. {question}
#     """)
#     print(response)

# Завдання 3
# Створіть найпростіший чат бот. Напишіть моделі якого
# персонажа вона повинна вдавати(відомий актор, персонаж
# кіно\книги, тощо).
# Реалізуйте двома способами:
# 1. Модель отримує інструкцію в якому стилі відповідати
# та нове повідомлення.
# 2. Модель отримує інструкцію та історію попередніх
# повідомлень як від користувача, так і її власні відповіді у
# форматі
# Instruction: ….
# Human: massage1
# AI: message2
# Human: massage3
# AI: message4
# Human: massage5
# AI:

llm = GoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    api_key=api_key,
)

instruction = None
history = []
while True:
    if not instruction:
        instruction = input("Instruction: ")
        history.append(f"instruction: {instruction}")

    question = input("Question: ")
    history.append(f"human: {question}")

    if question == '' or question == ' ':
        break

    response = llm.invoke(f"""
    Слідуй інструкції: {instruction}, орієнтуйся на історію розмови: {" ".join(history)}. 
    Тобі потрібно відповісти на питання: {question}.
    """)
    print(response)
    history.append(f"ai: {response}")


